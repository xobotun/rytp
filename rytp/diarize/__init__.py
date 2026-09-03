"""Diarizer subsystem — speaker diarization engines.

Public surface:

* :class:`Diarizer` — protocol (re-exported from :mod:`rytp.engines`).
* :class:`DiarSegment` — typed tuple (re-exported).
* :class:`NullDiarizer` — default, no HF token.
* :class:`EnergyDiarizer` — simple energy-based diarization, no HF token.
* :class:`PyannoteDiarizer` — gated on HF_TOKEN. Importing this module
  is safe; instantiation is what requires ``rytp[pyannote]``.
"""
from rytp.diarize.base import DiarSegment, Diarizer, NullDiarizer
from rytp.diarize.energy import EnergyDiarizer
from rytp.diarize.pyannote import PyannoteDiarizer

__all__ = ["DiarSegment", "Diarizer", "NullDiarizer", "EnergyDiarizer", "PyannoteDiarizer"]