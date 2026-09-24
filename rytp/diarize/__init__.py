"""Speaker diarization and the speaker roster.

Importing this package is what makes an engine's name resolvable: each
module registers its class at import time, so a name that is never imported
is a name that does not exist.

Only the engine modules are imported here. `store`, `pipeline`, `link` and
`mapper` are imported by their callers, because `pyannote` and `embed` are
loaded inside a *foreign* interpreter by the out-of-process seam and must
not drag the database layer along with them.
"""

from __future__ import annotations

from rytp.diarize import embed, none, pyannote  # noqa: F401
