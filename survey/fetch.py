"""`survey fetch` — pull model artifacts from the Hugging Face Hub into
`models/<name>/<block>/`, driven by the registry (`models/models.yaml`).

For each model's backend block (`gguf`/`onnx`) we fetch `common` + every quant's
`files` in one snapshot (the Hub de-dupes). The globs go to
`snapshot_download(allow_patterns=…)` verbatim — edit them in `models.yaml`, not
here. `--only` restricts which backends are fetched. Everything lands under
`--models-dir` (default `models/`), untracked local data you own.

Progress is one bar per artifact set, not the Hub's per-file bars: total bytes
come from the repo's file metadata, progress from watching the destination
(finished files plus the Hub's `*.incomplete` partials), and the bar clears
into a single `✓` line when the set is complete. If the metadata call fails
(offline with a warm cache, a proxy), the download still runs — under a
spinner, without a total.
"""

from __future__ import annotations

import argparse
import threading
from fnmatch import fnmatch
from pathlib import Path

import yaml
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import disable_progress_bars
from rich.filesize import decimal
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TransferSpeedColumn

from ._log import console, die, log, working
from .config import REGISTRY
from .registry import BACKEND_BLOCK

HF_PREFIX = "https://huggingface.co/"


def _repo_id(repo: str) -> str:
    """`https://huggingface.co/org/name` (or a bare `org/name`) → `org/name`."""
    return repo.removeprefix(HF_PREFIX).strip("/")


def _matched_sizes(
    repo_id: str, revision: str | None, patterns: list[str]
) -> dict[str, int] | None:
    """{rfilename: bytes} for the repo files the patterns select; None if the Hub
    metadata is unreachable."""
    try:
        info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
    except Exception:
        return None
    return {
        f.rfilename: f.size or 0
        for f in info.siblings
        if not patterns or any(fnmatch(f.rfilename, p) for p in patterns)
    }


def _bytes_on_disk(dest: Path, files: dict[str, int]) -> int:
    """Finished files plus in-flight partials — what the progress bar reads."""
    done = 0
    for name, size in files.items():
        p = dest / name
        if p.exists():
            done += min(p.stat().st_size, size)
    cache = dest / ".cache" / "huggingface"
    if cache.is_dir():
        for p in cache.rglob("*.incomplete"):
            try:
                done += p.stat().st_size
            except OSError:
                pass  # the downloader renamed it mid-scan
    return done


def _snapshot(
    repo_id: str, revision: str | None, dest: Path, patterns: list[str], label: str
) -> None:
    """One artifact set: metadata → bar → download in a thread → `✓` line."""
    files = _matched_sizes(repo_id, revision, patterns)

    failure: list[BaseException] = []

    def download() -> None:
        try:
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                local_dir=str(dest),
                allow_patterns=patterns or None,
            )
        except BaseException as e:  # noqa: BLE001 — carried to the main thread
            failure.append(e)

    note = ""
    if files is None:
        with working(f"{label} — downloading (size unknown: repo metadata unreachable)"):
            download()
    elif not console.is_terminal:
        download()  # no live bar to draw; the ✓ line below still reports the size
        note = f" ({decimal(sum(files.values()))})"
    else:
        total = sum(files.values())
        already = _bytes_on_disk(dest, files)
        thread = threading.Thread(target=download, daemon=True)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=24),
            DownloadColumn(),
            TransferSpeedColumn(),
            console=console,
            transient=True,
        ) as bar:
            task = bar.add_task(label, total=total, completed=min(already, total))
            thread.start()
            while thread.is_alive():
                bar.update(task, completed=min(_bytes_on_disk(dest, files), total))
                thread.join(0.2)
        note = f" ({decimal(total)}{', cached' if already >= total else ''})"

    if failure:
        die(f"{label}: download failed — {failure[0]} (re-run to resume)")
    log(f"✓ {label}{note} → {dest}")


def cmd_fetch(args: argparse.Namespace) -> None:
    disable_progress_bars()  # the Hub's per-file tqdm bars; ours replace them
    if not REGISTRY.exists():
        die(f"no registry at {REGISTRY}")
    registry = yaml.safe_load(REGISTRY.read_text()) or {}

    blocks = list(BACKEND_BLOCK)  # which backends to fetch
    if args.only:
        blocks = [b.strip() for b in args.only.split(",")]
        bad = [b for b in blocks if b not in BACKEND_BLOCK]
        if bad:
            die(f"--only: unknown backend(s) {bad}; choose from {list(BACKEND_BLOCK)}")

    names = args.models or list(registry)  # no names → every model in the registry
    if not names:
        die(f"{REGISTRY.name} is empty — nothing to fetch")

    dest_root = args.models_dir
    try:
        for name in names:
            entry = registry.get(name)
            if entry is None:
                known = ", ".join(registry) or "(none)"
                die(f"unknown model {name!r}. Known: {known}.")
            fetched = 0
            for backend in blocks:
                block = entry.get(BACKEND_BLOCK[backend])
                if not block:
                    continue  # this model has no artifact for that backend
                if not block.get("repo"):
                    die(f"{name}.{BACKEND_BLOCK[backend]}: no `repo` set in {REGISTRY.name}")
                quants = block.get("quants")
                if not quants:
                    die(f"{name}.{BACKEND_BLOCK[backend]}: no `quants` map in {REGISTRY.name}")
                # Pull the shared files once plus every declared quant's weights; the
                # Hub de-dupes, so one snapshot per block covers all quants.
                patterns = list(block.get("common", []))
                for qspec in quants.values():
                    patterns += (qspec or {}).get("files", [])
                _snapshot(
                    _repo_id(block["repo"]),
                    args.revision,
                    dest_root / name / BACKEND_BLOCK[backend],
                    patterns,
                    f"{name} {', '.join(quants)}",
                )
                fetched += 1
            if not fetched:
                die(
                    f"{name}: nothing to fetch for {blocks} — no matching blocks in {REGISTRY.name}"
                )
    except KeyboardInterrupt:
        die("interrupted — partial files are kept; re-run `survey fetch` to resume")
