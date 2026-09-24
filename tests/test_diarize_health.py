"""What `doctor` says about speakers. No imports of engines, no network."""

from __future__ import annotations

import pytest

from rytp import constants as C
from rytp.db import Database
from rytp.diarize.health import (
    check_diarizers,
    check_hf_token,
    configured_diarizer,
    usable,
)
from tests.fake_speaker_engines import GatedDiarizer, registered


@pytest.fixture(autouse=True)
def no_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let the developer's own token decide a test's outcome."""
    for name in C.HF_TOKEN_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def set_diarizer(db: Database, name: str) -> None:
    db.conn.execute(
        "INSERT INTO settings (key, value) VALUES (?, ?) "
        "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
        (C.SETTINGS_DIARIZER, name),
    )


# -- the registry ----------------------------------------------------------


def test_both_checks_are_registered() -> None:
    import rytp.commands.speakers  # noqa: F401 - importing registers them
    from rytp.commands import HEALTH_CHECKS

    assert {"diarizers", "hf-token"} <= set(HEALTH_CHECKS)


def test_both_checks_are_advisory() -> None:
    # contracts §5: `ok` tells the truth, `required` decides the exit code.
    # A base install has no pyannote and no token, and `doctor` must still
    # exit zero — the null diarizer needs neither.
    import rytp.commands.speakers  # noqa: F401
    from rytp.commands import HEALTH_CHECKS

    assert HEALTH_CHECKS["diarizers"].required is False
    assert HEALTH_CHECKS["hf-token"].required is False


def test_usable_reads_part_threes_availability_words() -> None:
    assert usable("ready") is True
    assert usable("interpreter ok") is True
    assert usable("needs pyannote.audio") is False
    assert usable("no HF_TOKEN") is False


def test_the_configured_diarizer_is_the_setting_then_the_default(
    db: Database,
) -> None:
    assert configured_diarizer(db) == C.DEFAULT_DIARIZER
    set_diarizer(db, "pyannote")
    assert configured_diarizer(db) == "pyannote"


# -- the diarizer check ----------------------------------------------------


def test_a_missing_engine_is_reported_as_missing(db: Database) -> None:
    # The dev environment has no pyannote, so the honest answer is False.
    # `required=False` is what keeps that from failing `doctor`.
    result = check_diarizers(db)
    assert result.ok is False
    assert "pyannote" in result.detail
    assert "none" in result.detail


def test_the_remedy_says_nothing_is_broken_when_the_configured_one_works(
    db: Database,
) -> None:
    result = check_diarizers(db)
    assert "nothing is broken" in (result.remedy or "")
    assert "rytp[pyannote]" in (result.remedy or "")


def test_the_remedy_leads_with_the_configured_engine_when_that_is_the_gap(
    db: Database,
) -> None:
    # This machine is set up to use pyannote and cannot. Still advisory,
    # but the remedy should be about pyannote, not about the others.
    set_diarizer(db, "pyannote")
    result = check_diarizers(db)
    assert result.ok is False
    remedy = result.remedy or ""
    assert remedy.startswith("the configured diarizer 'pyannote'")
    assert C.SETTINGS_DIARIZER in remedy


def test_everything_usable_is_the_only_way_to_pass(db: Database) -> None:
    from rytp.diarize import health

    monkey = {"none": "ready", "pyannote": "interpreter ok"}
    original = health._states
    health._states = lambda _db: monkey          # type: ignore[assignment]
    try:
        result = check_diarizers(db)
    finally:
        health._states = original                # type: ignore[assignment]
    assert result.ok is True
    assert result.remedy is None


def test_a_configured_engine_that_is_not_registered_at_all_is_a_failure(
    db: Database,
) -> None:
    set_diarizer(db, "nonesuch")
    result = check_diarizers(db)
    assert result.ok is False
    assert "nonesuch" in result.detail


def test_the_check_never_raises_on_a_broken_registry(db: Database) -> None:
    class Exploding:
        name = "fake-exploding"
        requires_hf_token = False
        out_of_process = False

        @property
        def required_module(self) -> str:
            raise RuntimeError("engines must not be able to break doctor")

        def diarize(self, audio: object) -> list[object]:
            return []

    with registered(Exploding):
        result = check_diarizers(db)
    assert isinstance(result.ok, bool)


# -- the token check -------------------------------------------------------


def test_a_missing_token_is_reported_as_missing(db: Database) -> None:
    # Honest, and advisory: `required=False` keeps it out of the exit code.
    result = check_hf_token(db)
    assert result.ok is False
    assert "not set" in result.detail
    assert C.PYANNOTE_DIARIZATION_MODEL in (result.remedy or "")
    assert "HF_TOKEN" in (result.remedy or "")


def test_the_detail_says_when_nothing_configured_needs_the_token(
    db: Database,
) -> None:
    assert "nothing configured needs it yet" in check_hf_token(db).detail


def test_that_reassurance_disappears_once_pyannote_is_configured(
    db: Database,
) -> None:
    set_diarizer(db, "pyannote")
    assert "nothing configured needs it yet" not in check_hf_token(db).detail


def test_a_present_token_is_reported_without_being_printed(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_thisisasecret")
    result = check_hf_token(db)
    assert result.ok is True
    assert "hf_thisisasecret" not in result.detail
    assert "HF_TOKEN" in result.detail


def test_the_alternative_variable_name_counts_too(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "t")
    assert check_hf_token(db).ok is True


def test_the_token_check_names_which_engines_are_gated(db: Database) -> None:
    with registered(GatedDiarizer):
        result = check_hf_token(db)
    assert "fake-gated-diarizer" in result.detail
