"""Concrete :class:`NullDiarizer` (lives in its own module for clarity).

Re-exports :class:`NullDiarizer` from :mod:`rytp.diarize.base`.
"""
from rytp.diarize.base import NullDiarizer

__all__ = ["NullDiarizer"]