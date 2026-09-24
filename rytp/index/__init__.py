"""The backward index: words to utterances, utterances to search hits.

Design §7. This package holds the half of the product the archive exists
for — given a word or a phrase, every place it was said. Nothing here is
imported for its side effects, and the package marker stays empty on
purpose: ``rytp/jobs/__init__.py`` imports
``rytp.index.utterances.index_readiness`` at module load time to register
the ``index`` job kind, so anything heavy added here would end up on the
import path of every command.

There is no ``stem.py``: contracts §4 puts ``stem_text`` in
``rytp/models.py``, because ``words.stem`` is written by the
transcription stage and a stemmer here would invert the dependency.
"""

from __future__ import annotations
