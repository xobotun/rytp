"""Fakes shared by the Part 6 test modules.

Nothing here runs a binary or touches the network. ``RecordingRunner``
does the two things a fake ffmpeg must do besides recording: answer a
probe or a loudness scan the way the real one would, and create the file
the command says it writes, so the orchestrator's existence checks
behave the way they will in production.
"""

from __future__ import annotations

import json
from pathlib import Path

from rytp.render.ffmpeg import CompletedRun


def probe_json(width: int, height: int, *, fps: str = "25/1", sar: str = "1:1") -> str:
    """One ffprobe reply, the shape `probe_command` asks for."""
    return json.dumps(
        {
            "streams": [
                {
                    "width": width,
                    "height": height,
                    "r_frame_rate": fps,
                    "sample_aspect_ratio": sar,
                }
            ]
        }
    )


def loudnorm_json(
    input_i: float,
    *,
    tp: float = -3.0,
    lra: float = 7.0,
    thresh: float = -30.0,
    offset: float = 0.1,
) -> str:
    """One measuring-pass stderr, chatter line included."""
    payload = {
        "input_i": f"{input_i:.2f}",
        "input_tp": f"{tp:.2f}",
        "input_lra": f"{lra:.2f}",
        "input_thresh": f"{thresh:.2f}",
        "target_offset": f"{offset:.2f}",
    }
    return "[Parsed_loudnorm_0 @ 0x1] \n" + json.dumps(payload)


class RecordingRunner:
    """A fake :data:`rytp.render.ffmpeg.Runner`.

    Args:
        returncode: what every invocation returns.
        stdout: canned stdout for any command not matched below.
        stderr: canned stderr for any command not matched below.
        geometry: stdout for ffprobe commands — one string, or a dict
            keyed by a substring of the path being probed, so a test
            with mixed-shape sources can answer differently per source.
            A dict is scanned in order, so put the specific key first.
        loudness: stderr for loudness-measuring commands, same keying.
        fail_when: when set, only commands containing this substring
            get ``returncode``; everything else succeeds. Lets a test
            fail one stage without failing the probe that precedes it.
        make_outputs: create the last argument as a file when it looks
            like an output path (no leading dash, has a suffix).
    """

    def __init__(
        self,
        *,
        returncode: int = 0,
        stdout: str = "",
        stderr: str = "",
        geometry: str | dict[str, str] | None = None,
        loudness: str | dict[str, str] | None = None,
        fail_when: str = "",
        make_outputs: bool = True,
    ) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.geometry = geometry
        self.loudness = loudness
        self.fail_when = fail_when
        self.make_outputs = make_outputs
        self.calls: list[list[str]] = []

    @staticmethod
    def is_probe(args: list[str]) -> bool:
        return "-show_entries" in args

    @staticmethod
    def is_measurement(args: list[str]) -> bool:
        return any("print_format=json" in arg for arg in args)

    @staticmethod
    def _pick(table: str | dict[str, str], args: list[str]) -> str:
        if isinstance(table, str):
            return table
        for needle, value in table.items():
            if any(needle in arg for arg in args):
                return value
        return ""

    def __call__(self, args: list[str]) -> CompletedRun:
        self.calls.append(list(args))
        out, err = self.stdout, self.stderr
        if self.geometry is not None and self.is_probe(args):
            out = self._pick(self.geometry, args)
        if self.loudness is not None and self.is_measurement(args):
            err = self._pick(self.loudness, args)
        code = self.returncode
        if self.fail_when and not any(self.fail_when in arg for arg in args):
            code = 0
        last = args[-1]
        if self.make_outputs and code == 0 and not last.startswith("-"):
            target = Path(last)
            if target.suffix and target.suffix != ".":
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"\x00" * 64)
        return CompletedRun(args=tuple(args), returncode=code, stdout=out, stderr=err)

    def commands_containing(self, needle: str) -> list[list[str]]:
        """Every recorded call with ``needle`` in one of its arguments."""
        return [call for call in self.calls if any(needle in arg for arg in call)]

    @property
    def probes(self) -> list[list[str]]:
        return [call for call in self.calls if self.is_probe(call)]

    @property
    def measurements(self) -> list[list[str]]:
        return [call for call in self.calls if self.is_measurement(call)]
