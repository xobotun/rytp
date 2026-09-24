"""`resolve_video_id`: by row id, external id, or catalogued URL, and the two
error shapes (BUGS.md entries 1 and 2).

Split out of `tests/test_catalog_videos.py`, which Task 15 owns: this module
is about the shared resolver in `rytp/commands/__init__.py`, not about
`videos add` / `videos list`.
"""

from __future__ import annotations

import pytest

from rytp.commands import resolve_video_id
from rytp.db import Database
from rytp.models import NotFoundError
from tests.fakes import VIDEO_A_URL, make_video


def test_resolves_by_row_id(db: Database) -> None:
    video_id = make_video(db)
    assert resolve_video_id(db, str(video_id)) == video_id


def test_resolves_by_external_id(db: Database) -> None:
    video_id = make_video(db, external_id="VIDEO_B")
    assert resolve_video_id(db, "VIDEO_B") == video_id


def test_resolves_by_catalogued_url(db: Database) -> None:
    """BUGS.md entry 1: registering first works; a URL should then resolve."""
    video_id = make_video(db, url=VIDEO_A_URL, external_id="VIDEO_A")
    assert resolve_video_id(db, VIDEO_A_URL) == video_id


def test_an_unregistered_url_names_videos_add(db: Database) -> None:
    """The owner's first command was a URL that had never been catalogued;
    the old message gave no hint that a video must be registered first."""
    url = "https://example.invalid/watch/VIDEO_UNSEEN"
    with pytest.raises(NotFoundError) as excinfo:
        resolve_video_id(db, url)
    message = str(excinfo.value)
    assert url in message
    assert "rytp videos add" in message
    assert url in message.rsplit("rytp videos add", 1)[-1]


def test_an_unknown_id_gives_the_plain_message(db: Database) -> None:
    """A bare id or external id is not a URL: no registration hint applies,
    because 'rytp videos add 4242' would not fix a typo'd id."""
    with pytest.raises(NotFoundError) as excinfo:
        resolve_video_id(db, "4242")
    message = str(excinfo.value)
    assert "4242" in message
    assert "videos add" not in message


def test_an_unknown_external_id_gives_the_plain_message(db: Database) -> None:
    with pytest.raises(NotFoundError) as excinfo:
        resolve_video_id(db, "NOT-A-URL-AND-NOT-CATALOGUED")
    assert "videos add" not in str(excinfo.value)
