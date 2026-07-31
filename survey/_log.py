"""Progress, warnings, and fatal errors — all to stderr, in one visual language.

stdout is reserved for machine-readable output (the plan table, results paths),
so everything human goes here and the tool composes cleanly in a pipeline.

The language is four glyphs, colored by prefix: `✓` done (green), `✗` failed
(red), `⚠` caveat (yellow), `•` item (dim). Long operations run under a
transient spinner (`working()`) that vanishes once the summary line prints.
Rich degrades to plain text when stderr is not a terminal, so CI logs and
`2>file` captures stay flat.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import NoReturn

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

console = Console(stderr=True, highlight=False, soft_wrap=True)

_PREFIX_STYLE = {
    "✓": "green",
    "✗": "red",
    "⚠": "yellow",
    "•": "dim",
}


def log(msg: str = "") -> None:
    """One line to stderr; a leading glyph picks up its color."""
    stripped = msg.lstrip()
    style = _PREFIX_STYLE.get(stripped[:1]) if stripped else None
    if style:
        pad = msg[: len(msg) - len(stripped)]
        glyph, rest = stripped[0], stripped[1:]
        escaped = rest.replace("[", "\\[")
        console.print(f"{pad}[{style}]{glyph}[/]{escaped}", emoji=False)
    else:
        console.print(msg, markup=False, emoji=False)


def warn(msg: str) -> None:
    log(f"⚠ {msg}")


def die(msg: str) -> NoReturn:
    """Fatal: the message travels in SystemExit; cli.main styles it on the way out."""
    raise SystemExit(f"✗ {msg}")


@contextmanager
def working(label: str):
    """A transient spinner + elapsed clock for a long step.

    Yields a `set(text)` callable to rephrase the line in place. Prints nothing
    when the step is over — the caller owns the summary line. Degrades to
    silence when stderr is not a terminal.
    """
    progress = Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )
    with progress:
        task = progress.add_task(label, total=None)

        def set_text(text: str) -> None:
            progress.update(task, description=text)

        yield set_text
