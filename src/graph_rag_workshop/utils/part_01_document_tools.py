"""Document handling and text extraction tools for a Pydantic AI agent."""

import re
from collections import deque
from pathlib import Path
from urllib.parse import urldefrag, urlparse

from loguru import logger
from markdownify import markdownify
from pdfminer.high_level import extract_text
from playwright.sync_api import Page, sync_playwright
from rapidocr import RapidOCR

# Document paths
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent.parent.parent / "data"
MY_DOCUMENTS = DATA_DIR / "my_documents"


# -----------------------------------------------------------------------------
# Document discovery and path resolution
# -----------------------------------------------------------------------------


def resolve_document_path(file_path: str) -> Path:
    """Resolve document names relative to the configured document folder.

    Relative paths are resolved from the workshop document directory.

    Args:
        file_path: The filename or path of the document to resolve.

    Returns:
        A Path object representing the resolved absolute path to the document.
    """
    document_root = MY_DOCUMENTS.resolve()
    path = Path(file_path).expanduser()
    if not path.is_absolute():
        path = document_root / path

    path = path.resolve()

    if not path.is_relative_to(document_root):
        raise ValueError(
            f"Document paths must stay inside the workshop document directory: "
            f"{document_root}"
        )
    if not path.is_file():
        raise FileNotFoundError(f"Document not found: {path.name}")

    return path


def list_my_available_documents() -> list[str]:
    """List the filenames available in the workshop document directory.

    Returns:
        A sorted list of visible document filenames.
    """
    return sorted(
        path.name
        for path in MY_DOCUMENTS.iterdir()
        if path.is_file() and not path.name.startswith(".")
    )


# -----------------------------------------------------------------------------
# Local document extraction
# -----------------------------------------------------------------------------


def extract_text_from_pdf_file(file_path: str) -> str:
    """Extract text from a PDF file.

    Args:
        file_path: The filename or path of the PDF to extract text from.

    Returns:
        The extracted PDF text.
    """
    path = resolve_document_path(file_path)
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, received: {path.name}")
    logger.info(f"Extracting PDF text from {path.name}")
    return extract_text(path)


def extract_text_from_image_file(file_path: str) -> str:
    """Extract text from an image using RapidOCR.

    Args:
        file_path: The filename or path of the image to extract text from.

    Returns:
        The extracted text content of the image.
    """
    path = resolve_document_path(file_path)
    if path.suffix.lower() not in {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}:
        raise ValueError(f"Expected a supported image file, received: {path.name}")

    engine = RapidOCR(params={"Global.log_level": "error"})
    result = engine(path)
    return (
        "\n".join(result.txts or []) if result else "Image text could not be extracted."
    )


# -----------------------------------------------------------------------------
# Website extraction
# -----------------------------------------------------------------------------


def html_to_markdown(html: str) -> str:
    """Convert HTML content to Markdown."""
    return markdownify(html, heading_style="ATX", strip=["script", "style"]).strip()


def add_source_to_markdown(markdown: str, source_url: str) -> str:
    """Place source metadata after the page's first Markdown heading."""
    source = f"Source: {source_url}"
    heading = re.search(r"^#{1,6} .+$", markdown, flags=re.MULTILINE)
    if heading:
        prefix = markdown[: heading.start()].strip()
        remainder = markdown[heading.end() :].strip()
        parts = [heading.group(), source, prefix, remainder]
        return "\n\n".join(part for part in parts if part)
    return f"{source}\n\n{markdown}"


def rendered_page_to_markdown(page: Page, url: str) -> tuple[str, list[str]]:
    """Render one page and return its Markdown and same-site links."""
    response = page.goto(url, wait_until="domcontentloaded", timeout=20_000)
    if response is None or not response.ok:
        status = response.status if response else "unknown"
        raise RuntimeError(f"Could not load {url}: HTTP {status}")

    main = page.locator("main").first
    main.wait_for(state="attached", timeout=10_000)

    final_url, _ = urldefrag(page.url)
    start_host = urlparse(url).hostname
    if urlparse(final_url).hostname != start_host:
        raise ValueError(f"Page redirected outside the source website: {final_url}")

    markdown = html_to_markdown(main.inner_html())
    links = page.locator("a[href]").evaluate_all(
        "links => links.map(link => link.href)"
    )
    internal_links = sorted(
        {
            urldefrag(link)[0]
            for link in links
            if urlparse(link).scheme in {"http", "https"}
            and urlparse(link).hostname == start_host
        }
    )
    return add_source_to_markdown(markdown, final_url), internal_links


def extract_text_from_website(url: str, max_pages: int = 30) -> str:
    """Render a website and extract its pages as structured Markdown.

    Args:
        url: The HTTP or HTTPS URL to scrape.
        max_pages: Maximum number of same-site pages to visit.

    Returns:
        The extracted page content as Markdown.
    """
    parsed_url = urlparse(url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.hostname:
        raise ValueError("A valid HTTP or HTTPS URL is required.")
    if parsed_url.username or parsed_url.password:
        raise ValueError("URLs containing credentials are not supported.")
    if max_pages < 1:
        raise ValueError("max_pages must be at least 1.")

    logger.info(f"Extracting website text from {url}")
    pending = deque([urldefrag(url)[0]])
    visited: set[str] = set()
    documents: list[str] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(
            accept_downloads=False,
            ignore_https_errors=False,
            service_workers="block",
        )
        page = context.new_page()

        try:
            while pending and len(visited) < max_pages:
                page_url = pending.popleft()
                if page_url in visited:
                    continue
                visited.add(page_url)

                try:
                    markdown, links = rendered_page_to_markdown(page, page_url)
                except Exception as error:
                    logger.warning(f"Skipping {page_url}: {error}")
                    continue

                documents.append(markdown)
                pending.extend(link for link in links if link not in visited)
        finally:
            context.close()
            browser.close()

    if not documents:
        raise ValueError(f"No content extracted from {url}")

    return "\n\n---\n\n".join(documents)


def extract_webpage_to_markdown(url: str, output_path: Path) -> str:
    """Extract a webpage to Markdown and save it locally.

    Args:
        url: The webpage URL to extract.
        output_path: The Markdown file where the extracted page content is saved.

    Returns:
        The extracted Markdown content.
    """
    markdown = extract_text_from_website(url)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")

    return markdown
