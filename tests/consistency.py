"""Shared loading and scanning for the cross-part consistency suite.

Part 8 owns no domain. What it owns is the set of invariants that hold
*between* the other seven parts, and every one of them starts from a fully
populated registry. Importing ``rytp.commands`` runs the sibling-import block
at the bottom of that module, which is what registers every command; importing
``rytp.jobs`` runs each part's registration block, which is what registers
every job kind. If either is short, the assertions below would pass on an
empty set, so the floors are checked once, here.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rytp.commands import Command, Param
    from rytp.jobs import JobKind

#: Importing these two modules is what populates both registries.
REGISTRY_MODULES: tuple[str, ...] = ("rytp.commands", "rytp.jobs")

#: Floors, not exact counts: a new command must not have to edit this file.
#: They exist so that a part which forgot its one-line sibling import fails
#: here, loudly, instead of making every test below vacuously true.
MIN_COMMANDS = 50
MIN_JOB_KINDS = 9

#: contracts §5, "Deletion — every entity gets a `remove`". Group -> the
#: command that removes one of its entities. Top-level entries have no group
#: and are keyed by their own name.
DELETION_OWNERS: dict[str, str] = {
    "channel": "channel.remove",
    "videos": "videos.remove",
    "assets": "assets.remove",
    "jobs": "jobs.cancel",
    "transcribe": "transcribe.remove",
    "index": "index.drop",
    "assemble": "assemble.remove",
    "render": "render.remove",
    "speakers": "speakers.remove",
}

#: Removals that delete files, and therefore must offer --dry-run and demand
#: --yes (contracts §5). `channel.remove` orphans videos rather than deleting
#: them, so it is not here.
FILE_DELETING_COMMANDS: frozenset[str] = frozenset(
    {"videos.remove", "assets.remove", "assemble.remove", "render.remove"}
)

#: Removals that only touch derived database rows, and so need neither.
DERIVED_ONLY_REMOVALS: frozenset[str] = frozenset({"index.drop", "transcribe.remove"})


def load_registries() -> tuple[dict[str, Command], dict[str, JobKind]]:
    """Import everything that registers, then hand back both registries."""
    for name in REGISTRY_MODULES:
        importlib.import_module(name)
    from rytp.commands import COMMANDS
    from rytp.jobs import JOB_KINDS

    assert len(COMMANDS) >= MIN_COMMANDS, (
        f"only {len(COMMANDS)} commands registered; a part is probably missing "
        f"its line in the sibling-import block at the bottom of "
        f"rytp/commands/__init__.py. Registered: {sorted(COMMANDS)}"
    )
    assert len(JOB_KINDS) >= MIN_JOB_KINDS, (
        f"only {len(JOB_KINDS)} job kinds registered; a part is probably missing "
        f"its register_job_kind block at the bottom of rytp/jobs/__init__.py. "
        f"Registered: {sorted(JOB_KINDS)}"
    )
    return COMMANDS, JOB_KINDS


def iter_params(commands: dict[str, Command]) -> Iterator[tuple[Command, Param]]:
    """Every (command, parameter) pair, in a stable order."""
    for name in sorted(commands):
        for param in commands[name].params:
            yield commands[name], param


def repo_root() -> Path:
    """The checkout root, found from this file rather than from the cwd."""
    return Path(__file__).resolve().parent.parent


def source_files(package: str = "rytp") -> list[Path]:
    """Every Python source file of one top-level directory, sorted."""
    return sorted(
        path
        for path in (repo_root() / package).rglob("*.py")
        if "__pycache__" not in path.parts
    )


def read_sources(package: str = "rytp") -> dict[Path, str]:
    """Path -> text, for the regex scans. UTF-8 because the project is Russian."""
    return {path: path.read_text(encoding="utf-8") for path in source_files(package)}


#: Every registered job kind, mapped to the registered *commands* a person can
#: run to create one. A kind with no producer is work the worker will never
#: receive — the review found exactly that on the gpu pool. The test below
#: asserts this table's keys are the registry's keys, so adding a kind without
#: adding a producer fails by name instead of going unnoticed.
PRODUCERS: dict[str, tuple[str, ...]] = {
    "download": ("ingest", "fetch-video"),
    "captions": ("ingest", "fetch-video"),
    "caption_words": ("ingest",),
    "extract_wav": ("ingest", "fetch-video"),
    "fingerprint": ("ingest", "transcribe.fingerprint"),
    "transcribe": ("ingest", "transcribe.run"),
    "align": ("ingest", "transcribe.align"),
    "index": ("index.build",),
    "diarize": ("speakers.enqueue",),
    "render": ("render.run",),
}


def enqueued_kinds_in_source() -> dict[str, list[str]]:
    """Job-kind string literals handed to ``enqueue`` anywhere under ``rytp/``.

    A regex, deliberately: the alternative is importing and calling everything,
    and a literal is exactly what goes stale. Kinds passed through a variable
    are invisible here and are covered by the ingest-chain test instead.
    """
    import re

    pattern = re.compile(r"""enqueue\(\s*db\s*,\s*["']([a-z_]+)["']""")
    found: dict[str, list[str]] = {}
    for path, text in read_sources("rytp").items():
        for kind in pattern.findall(text):
            found.setdefault(kind, []).append(str(path.name))
    return found


def write_test_wav(path: Path, *, duration_ms: int = 1000, freq_hz: int = 220) -> Path:
    """A real 16 kHz mono int16 WAV, so nothing downstream has to be faked.

    A tone rather than silence: the acoustic fingerprint measures f0 and
    spectral tilt, and those are undefined on a flat zero signal.
    """
    import math
    import wave

    from rytp import config

    sample_rate = 16_000
    frames = int(sample_rate * duration_ms / 1000)
    config.ensure_dir(path.parent)
    samples = bytearray()
    for index in range(frames):
        value = int(12000 * math.sin(2 * math.pi * freq_hz * index / sample_rate))
        samples += int(value).to_bytes(2, "little", signed=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(bytes(samples))
    return path


def cli_command_paths() -> set[tuple[str, ...]]:
    """Every argv path the generated Typer app actually answers to.

    Walks the Click command tree rather than parsing ``--help`` text, so a
    command that exists but is hidden still counts and a summary that happens
    to contain a word does not.
    """
    import typer.main

    from rytp.cli import build_app

    root = typer.main.get_command(build_app())
    found: set[tuple[str, ...]] = set()

    def walk(node: Any, prefix: tuple[str, ...]) -> None:
        children = getattr(node, "commands", None)
        if not children:
            if prefix:
                found.add(prefix)
            return
        for name, child in children.items():
            walk(child, (*prefix, name))

    walk(root, ())
    return found
