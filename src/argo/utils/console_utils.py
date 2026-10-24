"""Shared Rich console formatting for the workshop scripts."""

from collections.abc import Callable
from time import perf_counter
from typing import Any, TypeVar

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

console = Console(highlight=False)

STEP_STYLE = "bold cyan"
INFO_STYLE = "green"
T = TypeVar("T")


def print_step(title: str) -> None:
    """Print a clearly separated workshop step heading."""
    separator = "=" * max(1, console.width // 2)
    console.print(f"\n{separator}\n{title}\n{separator}", style=STEP_STYLE)


def print_result(result: object) -> None:
    """Print result content between uncolored delimiter lines."""
    separator = "-" * max(1, console.width // 2)
    console.print(separator)
    console.print(result, markup=False)
    console.print(separator)


def run_with_progress(
    operation: Callable[..., T],
    /,
    *args: Any,
    description: str = "Processing the question",
    **kwargs: Any,
) -> tuple[T, float]:
    """Run an operation with an animated progress bar and elapsed-time counter."""
    start = perf_counter()
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task(description, total=None)
        result = operation(*args, **kwargs)
        elapsed = perf_counter() - start
        progress.update(
            task_id,
            description="Processing complete",
            total=1,
            completed=1,
        )

    return result, elapsed


def print_question_answer(
    question: str,
    answer: object,
    processing_time: float | None = None,
) -> None:
    """Print the original question and model answer as the final result."""
    print_step("FINAL RESULT")
    console.print("Question:", style=INFO_STYLE)
    print_result(question)
    console.print("Answer:", style=INFO_STYLE)
    print_result(answer)
    if processing_time is not None:
        console.print(
            f"Total processing time: {processing_time:.2f} seconds",
            style=INFO_STYLE,
        )
