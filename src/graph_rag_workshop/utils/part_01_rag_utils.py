"""RAG preparation helpers for the Part 1 webpage exercise."""

import re
from pathlib import Path


def format_markdown_chunks(text: str, documents: list) -> list[str]:
    """Format split Markdown documents with source and structural metadata."""
    source_by_heading = {
        (len(level), heading): source
        for level, heading, source in re.findall(
            r"^(#{1,3})[ \t]+(.+?)\s*\n\s*Source:[ \t]*(\S+)",
            text,
            flags=re.MULTILINE,
        )
    }

    chunks = []
    for index, document in enumerate(documents):
        section = (
            " > ".join(
                document.metadata[key]
                for key in ("Header 1", "Header 2", "Header 3")
                if key in document.metadata
            )
            or "No header"
        )
        source = next(
            (
                source_by_heading[(level, document.metadata[f"Header {level}"])]
                for level in (3, 2, 1)
                if (level, document.metadata.get(f"Header {level}"))
                in source_by_heading
            ),
            "",
        )
        content = re.sub(
            r"^Source:[ \t]*\S+\s*",
            "",
            document.page_content,
            flags=re.MULTILINE,
        ).strip()
        chunks.append(
            f"Source: {source or 'Unknown'}\n"
            f"Chunk: {index}\n"
            f"Section: {section}\n"
            f"Start index: {document.metadata.get('start_index', 0)}\n"
            f"Content:\n{content}"
        )

    if not chunks:
        raise ValueError("The webpage extraction produced no text chunks.")

    return chunks


def save_chunks_to_markdown(chunks: list[str], output_path: Path) -> None:
    """Save documented chunks to a Markdown file."""
    chunks_markdown = "\n\n---\n\n".join(
        f"# Chunk {index}\n\n{chunk}" for index, chunk in enumerate(chunks, start=1)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(chunks_markdown, encoding="utf-8")
