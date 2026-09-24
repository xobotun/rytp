"""One flag, one meaning, one spelling — across all fifty-odd commands.

Each test names the review finding it exists to prevent. The findings are in
`docs/superpowers/2026-09-21-review-findings.md`; the ones with no test were
found by a person reading seven plans side by side, which is not a repeatable
process.
"""

from __future__ import annotations

import re
from collections import defaultdict

import pytest

from rytp.commands import PARAM_ALIASES, PARAM_TYPES, SPEAKER_PARAMS
from tests.consistency import (
    DERIVED_ONLY_REMOVALS,
    FILE_DELETING_COMMANDS,
    iter_params,
    load_registries,
)

COMMANDS, _JOB_KINDS = load_registries()
NAMES = sorted(COMMANDS)

#: A concept that must have exactly one spelling, mapped from the spellings
#: that are *not* it. Finding 4.2 chose `--video`; `--enqueue` is the spelling
#: three of the four enqueue flags already use.
BANNED_SPELLINGS: dict[str, str] = {
    "video_id": "video",
    "videoid": "video",
    "queue": "enqueue",
    "global_speaker": "speaker",
    "limit_rows": "limit",
    "max_rows": "limit",
    # Task 15 (BUGS.md entry 25): the owner's revision settled on `--long`
    # for the detailed `videos list` view, precisely so `--short` stays free
    # rather than silently meaning something already. Reserve it.
    "short": "long",
}

#: Names whose type is fixed across the whole surface, because a flag that is
#: a string here and an integer there is two flags wearing one name.
PINNED_TYPES: dict[str, type] = {
    "video": str,  # resolve_video_id accepts a row id or an external id
    "speaker": str,
    "video_local_speaker": str,
    "limit": int,
    "priority": int,
    "seed": int,
    "pad_ms": int,
    "dry_run": bool,
    "yes": bool,
    "force": bool,
    "enqueue": bool,
    "long": bool,
}


def _params_by_name() -> dict[str, list[tuple[str, object]]]:
    grouped: dict[str, list[tuple[str, object]]] = defaultdict(list)
    for cmd, param in iter_params(COMMANDS):
        grouped[param.name].append((cmd.name, param))
    return grouped


PARAMS_BY_NAME = _params_by_name()


@pytest.mark.parametrize("banned", sorted(BANNED_SPELLINGS))
def test_no_concept_is_spelled_two_ways(banned: str) -> None:
    """Finding 4.2: `--video-id` and `--video` were the same thing, and the
    rename was applied to the prose but not to every registration."""
    users = [name for name, _ in PARAMS_BY_NAME.get(banned, [])]
    assert not users, (
        f"{users} declare --{banned.replace('_', '-')}; the agreed spelling is "
        f"--{BANNED_SPELLINGS[banned].replace('_', '-')}"
    )


@pytest.mark.parametrize("param_name", sorted(PINNED_TYPES))
def test_one_parameter_name_has_one_type(param_name: str) -> None:
    """`--video` as a `str` on one command and an `int` on another is two
    flags sharing a name: one accepts `VIDEO_A`, the other refuses it."""
    expected = PINNED_TYPES[param_name]
    wrong = [
        (name, param.type.__name__)
        for name, param in PARAMS_BY_NAME.get(param_name, [])
        if param.type is not expected
    ]
    assert not wrong, f"--{param_name} must be {expected.__name__}; wrong: {wrong}"


def test_one_parameter_name_has_one_type_everywhere_else_too() -> None:
    """The open-ended half of the rule above: any name used by two commands
    with two different types is a finding, whether or not it is pinned.

    Types, not meanings. A *positional* may legitimately mean different things
    in different commands — `videos add <target>` takes a URL or a path and
    `assemble plan <target>` takes a sentence — because a positional is read
    in the context of the verb in front of it and is never typed as a flag. An
    option's name is typed alone, so its meaning must not move; that is what
    `BANNED_SPELLINGS` and the speaker tests above enforce.
    """
    clashes = {
        name: sorted({param.type.__name__ for _, param in entries})
        for name, entries in PARAMS_BY_NAME.items()
        if len({param.type for _, param in entries}) > 1
    }
    assert not clashes, f"one name, several types: {clashes}"


def test_every_parameter_name_is_a_python_identifier() -> None:
    """`Param.name` is the handler's keyword argument, so `min-align` could
    never be passed. Whether the CLI renders it with a dash is Part 1's job."""
    bad = [
        f"{cmd.name}.{param.name}"
        for cmd, param in iter_params(COMMANDS)
        if not param.name.isidentifier() or param.name != param.name.lower()
    ]
    assert not bad, bad


def test_every_parameter_type_is_one_the_surfaces_can_render() -> None:
    bad = [
        f"{cmd.name}.{param.name}: {param.type!r}"
        for cmd, param in iter_params(COMMANDS)
        if param.type not in PARAM_TYPES
    ]
    assert not bad, bad


# --- short flags -----------------------------------------------------


def test_every_short_flag_is_a_dash_and_one_letter() -> None:
    """Part 1's validator demands `-n`; `short="n"` raises at import. Pinned
    here so the shape is stated once for every part rather than discovered by
    whoever adds the next flag."""
    bad = [
        f"{cmd.name}.{param.name}={param.short!r}"
        for cmd, param in iter_params(COMMANDS)
        if param.short is not None and not re.fullmatch(r"-[A-Za-z]", param.short)
    ]
    assert not bad, bad


def test_a_short_flag_is_unique_within_its_command() -> None:
    for name in NAMES:
        shorts = [p.short for p in COMMANDS[name].params if p.short]
        assert len(shorts) == len(set(shorts)), f"{name}: {shorts}"


def test_a_short_letter_means_one_thing_across_the_whole_surface() -> None:
    """Muscle memory does not know which command it is in. `-s` meaning
    `--search` in one place and `--speaker` in another is how a filtered list
    silently becomes an unfiltered one."""
    meanings: dict[str, set[str]] = defaultdict(set)
    for _cmd, param in iter_params(COMMANDS):
        if param.short:
            meanings[param.short].add(param.name)
    clashes = {short: sorted(names) for short, names in meanings.items() if len(names) > 1}
    assert not clashes, f"one letter, several meanings: {clashes}"


# --- the speaker flags (contracts §5) --------------------------------

SPEAKER_PARAM_NAMES = {param.name for param in SPEAKER_PARAMS}


def test_the_shared_speaker_flags_mean_the_same_thing_everywhere() -> None:
    """contracts §5 pins two identifier spaces and one resolver. A command may
    legitimately offer only some of the three — assembling from the corpus has
    no use for a per-video diarizer label — so this checks each name it *does*
    offer against the shared declaration rather than demanding all three.

    SPEAKER_PARAMS describes *options* — a filter you either ask for or don't,
    so ``None`` means "not asked for". A bare ``<video>`` positional is a
    different grammatical thing: the verb's target, required by
    construction (``videos remove``, `speakers labels`, `transcript show`...).
    Comparing a positional's default against an option's sentinel would
    demand that every one of those verbs accept a missing target, which is
    a regression, not a consistency fix — `test_one_parameter_name_has_one_type`
    already pins its type across both uses. So positionals are exempted here
    and checked only for type, same as everywhere else.
    """
    shared = {param.name: param for param in SPEAKER_PARAMS}
    for cmd, param in iter_params(COMMANDS):
        if param.name not in shared or param.positional:
            continue
        reference = shared[param.name]
        assert param.type is reference.type, f"{cmd.name}.{param.name}: type"
        assert param.default is reference.default, (
            f"{cmd.name}.{param.name} defaults to {param.default!r}; the shared "
            f"declaration uses {reference.default!r}. `None` means 'not asked "
            f"for', which is a different answer from 'asked for nothing'."
        )


def test_a_per_video_label_always_comes_with_a_video() -> None:
    """contracts §5: "`--video-local-speaker` **requires a video to be named as
    well**". A bare SPEAKER_00 would match the first-detected voice of every
    diarized video, which is not a person."""
    for name in NAMES:
        params = {p.name for p in COMMANDS[name].params}
        if "video_local_speaker" in params:
            assert "video" in params, f"{name} takes a local label but no --video"


def test_the_speaker_alias_is_declared_once_and_reaches_the_speaker_flag() -> None:
    assert PARAM_ALIASES.get("speaker") == ("--global-speaker",)


# --- conventions -----------------------------------------------------


def test_a_boolean_flag_defaults_to_false_or_says_why_not() -> None:
    """A flag reads as "turn this on". The three that default to True are
    behaviour you turn *off*, and each is named here so a fourth cannot
    appear silently."""
    on_by_default = {
        (cmd.name, param.name)
        for cmd, param in iter_params(COMMANDS)
        if param.type is bool and param.default is True
    }
    assert on_by_default == {
        ("fetch-video", "captions"),
        ("render.run", "loudnorm"),
        ("transcribe.run", "refine"),
        ("transcribe.align", "refine"),
    }, sorted(on_by_default)


@pytest.mark.parametrize("name", sorted(FILE_DELETING_COMMANDS))
def test_anything_that_deletes_a_file_plans_first_and_asks(name: str) -> None:
    """contracts §5: "Anything that deletes a file supports `--dry-run` and
    requires `--yes`"."""
    assert name in COMMANDS
    params = {p.name for p in COMMANDS[name].params}
    assert {"dry_run", "yes"} <= params, f"{name} is missing {sorted({'dry_run', 'yes'} - params)}"


@pytest.mark.parametrize("name", sorted(DERIVED_ONLY_REMOVALS))
def test_a_removal_of_derived_rows_needs_no_ceremony(name: str) -> None:
    """The same contract sentence, read the other way: `index.drop` rebuilds
    from `index build`, so demanding --yes would be noise."""
    params = {p.name for p in COMMANDS[name].params}
    assert "yes" not in params, f"{name} demands --yes for rows it can rebuild"


def test_removal_is_never_a_job() -> None:
    """contracts §5: "Removal is not a job… A half-deleted entity recovered
    from a crashed queue is worse than a slow command"."""
    removals = sorted(FILE_DELETING_COMMANDS | DERIVED_ONLY_REMOVALS)
    for name in removals:
        assert COMMANDS[name].long_running is False, f"{name} is marked long_running"
        params = {p.name for p in COMMANDS[name].params}
        assert "enqueue" not in params, f"{name} offers to queue a deletion"
