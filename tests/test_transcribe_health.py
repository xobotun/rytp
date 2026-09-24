"""Which engines can actually run here, answered without importing any of them."""
from __future__ import annotations

import subprocess
import sys

import pytest

from rytp.commands import HEALTH_CHECKS
from rytp.db import Database
from rytp.transcribe import health
from rytp.transcribe.engines.gigaam import GigaAMTranscriber
from rytp.transcribe.engines.whisper import FasterWhisperTranscriber


def test_registering_adds_one_check_per_engine_plus_the_gpu(db: Database) -> None:
    health.register_transcribe_checks()
    assert "transcriber:whisper" in HEALTH_CHECKS
    assert "transcriber:gigaam" in HEALTH_CHECKS
    assert "aligner:mfa" in HEALTH_CHECKS
    assert "aligner:wav2vec2" in HEALTH_CHECKS
    assert "gpu" in HEALTH_CHECKS


def test_registering_twice_is_harmless(db: Database) -> None:
    health.register_transcribe_checks()
    before = len(HEALTH_CHECKS)
    health.register_transcribe_checks()
    assert len(HEALTH_CHECKS) == before


def test_every_part_3_check_is_advisory(db: Database) -> None:
    # Contracts §5: ok tells the truth, required=False is what makes a missing
    # optional engine non-fatal. Nothing here is needed by the base install.
    health.register_transcribe_checks()
    for name in (
        "transcriber:whisper",
        "transcriber:gigaam",
        "aligner:mfa",
        "aligner:wav2vec2",
        "gpu",
    ):
        assert HEALTH_CHECKS[name].required is False, name


def test_a_missing_engine_reports_ok_false_rather_than_pretending(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(health, "_nvidia_smi", lambda: None)
    result = health.gpu_check(db)
    assert result.ok is False


def test_a_check_never_raises_and_always_explains_itself(db: Database) -> None:
    health.register_transcribe_checks()
    for name, check in HEALTH_CHECKS.items():
        result = check.run(db)
        assert isinstance(result.ok, bool)
        assert result.detail, f"{name} reported nothing"
        if not result.ok:
            assert result.remedy, f"{name} failed without saying how to fix it"


def test_an_in_process_engine_remedy_names_its_extra() -> None:
    assert engine_remedy_of(FasterWhisperTranscriber) == "pip install rytp[whisper]"


def test_an_out_of_process_engine_remedy_names_the_interpreter_setting() -> None:
    remedy = engine_remedy_of(GigaAMTranscriber)
    assert "engine.interpreter.gigaam" in remedy


def engine_remedy_of(cls: type) -> str:
    return health.engine_remedy(cls)


def test_no_gpu_tool_is_reported_not_fatal(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(health, "_nvidia_smi", lambda: None)
    result = health.gpu_check(db)
    assert result.ok is False
    assert "CPU" in result.detail
    assert result.remedy


def test_a_visible_gpu_is_reported_with_its_name(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(health, "_nvidia_smi", lambda: "nvidia-smi")
    monkeypatch.setattr(
        health, "_query_gpu", lambda binary: (0, "NVIDIA GeForce RTX 3080 Laptop GPU, 16384 MiB")
    )
    result = health.gpu_check(db)
    assert result.ok is True
    assert "3080" in result.detail


def test_a_wedged_driver_is_reported_not_raised(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(binary: str) -> tuple[int, str]:
        raise subprocess.TimeoutExpired(cmd=[binary], timeout=1)

    monkeypatch.setattr(health, "_nvidia_smi", lambda: "nvidia-smi")
    monkeypatch.setattr(health, "_query_gpu", explode)
    result = health.gpu_check(db)
    assert result.ok is False
    assert result.remedy


def test_probing_imports_no_engine_dependency(db: Database) -> None:
    # find_spec consults the filesystem; nothing here may import torch.
    health.register_transcribe_checks()
    for check in HEALTH_CHECKS.values():
        check.run(db)
    assert not {"torch", "transformers", "faster_whisper", "gigaam"} & set(sys.modules)


# -- entries 7, 33: the probe replaces the module-blind "interpreter ok" ----


def test_an_out_of_process_engine_with_a_missing_module_is_no_longer_falsely_ready(
    db: Database,
) -> None:
    # BUGS.md entry 7, confirmed on the real machine: gigaam, wav2vec2 and
    # pyannote all said "interpreter ok" and none of them could run.
    health.register_transcribe_checks()
    result = health.engine_check(db, GigaAMTranscriber)
    assert result.ok is False
    assert "interpreter ok" not in result.detail
    assert result.remedy is not None


def test_the_cuda_fact_is_paired_with_the_module_fact(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    # BUGS.md entry 33: doctor already found the 3080; this pairs that fact
    # with the per-engine module check instead of leaving it silent.
    def fake_cuda_fact(_db: Database, _cls: object) -> dict[str, object]:
        return {
            "module_ok": True,
            "module_error": "",
            "torch": "2.4.0+cpu",
            "cuda_available": False,
            "device": "cpu",
        }

    monkeypatch.setattr(health, "_cuda_fact", fake_cuda_fact)
    monkeypatch.setattr(health, "availability", lambda db, cls: "ready")
    result = health.engine_check(db, GigaAMTranscriber)
    assert result.ok is True
    assert "no CUDA" in result.detail
    assert "cu124" in (result.remedy or "")


def test_no_cuda_fact_is_added_when_torch_is_entirely_absent(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_cuda_fact(_db: Database, _cls: object) -> dict[str, object]:
        return {
            "module_ok": False,
            "module_error": "ModuleNotFoundError: no module named 'gigaam'",
            "torch": None,
            "cuda_available": None,
            "device": "cpu",
        }

    monkeypatch.setattr(health, "_cuda_fact", fake_cuda_fact)
    result = health.engine_check(db, GigaAMTranscriber)
    assert "CUDA" not in result.detail


# -- entry 11: the Hugging Face symlink advisory ----------------------------


def test_hf_cache_check_is_registered_and_advisory() -> None:
    health.register_transcribe_checks()
    assert "hf-cache" in HEALTH_CHECKS
    assert HEALTH_CHECKS["hf-cache"].required is False


def test_hf_cache_check_reports_the_symlink_fact_honestly(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(health, "_symlinks_supported", lambda: True)
    assert health.hf_cache_check(db).ok is True

    monkeypatch.setattr(health, "_symlinks_supported", lambda: False)
    result = health.hf_cache_check(db)
    assert result.ok is False
    assert result.remedy is not None
    assert "disk" in result.detail
