"""The durable artifact: one ordered slot list, written so a human can edit it."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from rytp import constants as C
from rytp.assemble.cutlist import (
    Alternative,
    CutList,
    CutlistError,
    CutlistParams,
    Slot,
    Substitution,
    cutlist_name,
    cutlist_path,
    dumps_cutlist,
    load_cutlist,
    read_cutlist,
    target_from_slots,
    validate_name,
    write_cutlist,
)
from rytp.models import Fragment, InvalidInputError

CREATED = "2026-09-21T09:00:00+00:00"

PARAMS = CutlistParams(
    consistency=0.25, seed=0, pad_ms=0, speaker="", exclude=(), min_align_score=0.0
)


def sample() -> CutList:
    """A cut list with one fragment, one alternative, one gap, one substitution."""
    return CutList(
        schema_version=C.CUTLIST_SCHEMA_VERSION,
        name="demo",
        target="мы всё исправим",
        created_at=CREATED,
        params=PARAMS,
        slots=(
            Slot(
                kind="fragment",
                target_first=0,
                target_last=1,
                text="мы всё",
                video_id=3,
                first_word_ord=1204,
                last_word_ord=1205,
                start_ms=612_340,
                end_ms=613_100,
                align_score=0.81,
                cost=1.42,
                tier="aligned",
                video_speaker_id=11,
                speaker_label="host",
                alternatives=(
                    Alternative(
                        video_id=7,
                        first_word_ord=88,
                        last_word_ord=89,
                        start_ms=10_500,
                        end_ms=11_220,
                        text="мы всё",
                        cost=1.77,
                        align_score=0.64,
                        video_speaker_id=22,
                        speaker_label="guest",
                    ),
                ),
            ),
            Slot(
                kind="gap",
                target_first=2,
                target_last=2,
                text="исправим",
                substitutions=(
                    Substitution(
                        text="исправит",
                        reason="edit",
                        distance=1,
                        occurrences=3,
                        video_id=9,
                        first_word_ord=502,
                        last_word_ord=502,
                        start_ms=220_100,
                        end_ms=220_780,
                    ),
                ),
            ),
        ),
    )


def test_the_file_opens_with_a_comment_for_the_person_editing_it() -> None:
    text = dumps_cutlist(sample())
    assert text.startswith("#")
    assert "hand-edit" in text.lower()


def test_it_parses_as_toml_and_keeps_the_top_level_fields() -> None:
    parsed = tomllib.loads(dumps_cutlist(sample()))
    assert parsed["schema_version"] == C.CUTLIST_SCHEMA_VERSION
    assert parsed["name"] == "demo"
    assert parsed["target"] == "мы всё исправим"
    assert parsed["created_at"] == CREATED
    assert parsed["params"]["consistency"] == 0.25
    assert parsed["params"]["exclude"] == []


def test_slots_are_one_ordered_array_with_a_kind() -> None:
    slots = tomllib.loads(dumps_cutlist(sample()))["slot"]
    assert [slot["kind"] for slot in slots] == ["fragment", "gap"]
    assert "index" not in slots[0]


def test_a_fragment_slot_carries_every_field_the_renderer_needs() -> None:
    fragment = tomllib.loads(dumps_cutlist(sample()))["slot"][0]
    assert fragment["video_id"] == 3
    assert fragment["first_word_ord"] == 1204
    assert fragment["last_word_ord"] == 1205
    assert fragment["start_ms"] == 612_340
    assert fragment["end_ms"] == 613_100
    assert fragment["text"] == "мы всё"
    assert fragment["speaker_label"] == "host"
    assert fragment["alternative"][0]["video_id"] == 7


def test_an_alternative_is_not_lossy() -> None:
    """A human swapping in an alternative should see what the chosen
    fragment had: alignment score and speaker, not just timings."""
    alternative = tomllib.loads(dumps_cutlist(sample()))["slot"][0]["alternative"][0]
    assert alternative["align_score"] == pytest.approx(0.64)
    assert alternative["video_speaker_id"] == 22
    assert alternative["speaker_label"] == "guest"


def test_a_gap_slot_carries_the_word_and_its_substitutions() -> None:
    gap = tomllib.loads(dumps_cutlist(sample()))["slot"][1]
    assert gap["text"] == "исправим"
    assert gap["target_first"] == gap["target_last"] == 2
    assert gap["substitution"][0]["text"] == "исправит"
    assert gap["substitution"][0]["reason"] == "edit"


def test_absent_optional_keys_are_omitted_rather_than_nulled() -> None:
    gap = tomllib.loads(dumps_cutlist(sample()))["slot"][1]
    for absent in ("video_id", "start_ms", "end_ms", "align_score", "gap_before_ms"):
        assert absent not in gap


def test_gap_before_ms_is_written_when_a_human_set_it() -> None:
    cutlist = sample()
    edited = CutList(
        schema_version=cutlist.schema_version,
        name=cutlist.name,
        target=cutlist.target,
        created_at=cutlist.created_at,
        params=cutlist.params,
        slots=(
            Slot(**{**vars(cutlist.slots[0]), "gap_before_ms": 320}),
            cutlist.slots[1],
        ),
    )
    assert tomllib.loads(dumps_cutlist(edited))["slot"][0]["gap_before_ms"] == 320


def test_quotes_and_backslashes_survive_the_round_trip() -> None:
    cutlist = sample()
    awkward = Slot(
        kind="fragment",
        target_first=0,
        target_last=0,
        text='он сказал "нет" \\ и ушёл',
        video_id=1,
        first_word_ord=0,
        last_word_ord=0,
        start_ms=0,
        end_ms=100,
    )
    edited = CutList(
        schema_version=cutlist.schema_version,
        name=cutlist.name,
        target=cutlist.target,
        created_at=cutlist.created_at,
        params=cutlist.params,
        slots=(awkward,),
    )
    parsed = tomllib.loads(dumps_cutlist(edited))
    assert parsed["slot"][0]["text"] == 'он сказал "нет" \\ и ушёл'


def test_the_same_cut_list_serializes_byte_for_byte_identically() -> None:
    assert dumps_cutlist(sample()) == dumps_cutlist(sample())


def test_write_cutlist_creates_the_directory_and_writes_utf8_with_lf(
    tmp_path: Path,
) -> None:
    target = tmp_path / "cutlists" / "demo.toml"
    written = write_cutlist(sample(), target)
    assert written == target
    raw = target.read_bytes()
    assert b"\r\n" not in raw
    assert "мы всё".encode() in raw


def test_properties_split_the_slots_and_measure_the_output() -> None:
    cutlist = sample()
    assert len(cutlist.fragments) == 1
    assert len(cutlist.gaps) == 1
    assert cutlist.duration_ms == 613_100 - 612_340


def test_a_fragment_slot_converts_to_the_contract_fragment_type() -> None:
    fragment = sample().fragments[0].as_fragment()
    assert isinstance(fragment, Fragment)
    assert fragment.video_id == 3
    assert fragment.start_ms == 612_340


def test_a_gap_slot_refuses_to_be_a_fragment() -> None:
    with pytest.raises(InvalidInputError, match="gap"):
        sample().gaps[0].as_fragment()


def test_cutlist_name_slugifies_the_target() -> None:
    assert cutlist_name("Мы всё исправим!") == "мы-все-исправим"


def test_cutlist_name_truncates_and_never_ends_in_a_dash() -> None:
    name = cutlist_name("слово " * 40)
    assert len(name) <= C.ASSEMBLE_NAME_MAX_CHARS
    assert not name.endswith("-")


def test_cutlist_name_falls_back_when_the_target_slugifies_to_nothing() -> None:
    assert cutlist_name("!!! ???") == C.ASSEMBLE_FALLBACK_NAME


def test_validate_name_rejects_anything_that_could_escape_the_directory() -> None:
    for bad in ("", "   ", "..", "a/b", "a\\b", "/abs", "C:name"):
        with pytest.raises(InvalidInputError):
            validate_name(bad)
    assert validate_name(" demo ") == "demo"


def test_cutlist_path_lands_under_the_data_tree(data_dir: Path) -> None:
    assert cutlist_path("demo") == data_dir / C.CUTLISTS_DIRNAME / "demo.toml"


MINIMAL = """
schema_version = 1
name = "demo"
target = "мы все"

[[slot]]
kind = "fragment"
target_first = 0
target_last = 1
text = "мы все"
video_id = 3
start_ms = 1000
end_ms = 2000
"""


def write(tmp_path: Path, text: str, name: str = "demo.toml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def test_a_written_cut_list_loads_back_identically(tmp_path: Path) -> None:
    path = write_cutlist(sample(), tmp_path / "demo.toml")
    assert load_cutlist(path) == sample()


def test_a_minimal_hand_written_file_loads(tmp_path: Path) -> None:
    cutlist = load_cutlist(write(tmp_path, MINIMAL))
    assert cutlist.name == "demo"
    assert len(cutlist.fragments) == 1
    assert cutlist.fragments[0].start_ms == 1000


def test_missing_provenance_defaults_rather_than_failing(tmp_path: Path) -> None:
    """Someone retyping a timing should not have to keep the ordinals."""
    cutlist = load_cutlist(write(tmp_path, MINIMAL))
    assert cutlist.fragments[0].first_word_ord == -1
    assert cutlist.fragments[0].last_word_ord == -1
    assert cutlist.fragments[0].as_fragment().video_id == 3


def test_missing_informational_fields_default(tmp_path: Path) -> None:
    text = 'schema_version = 1\n\n[[slot]]\nkind = "gap"\ntext = "х"\n'
    cutlist = load_cutlist(write(tmp_path, text, name="notes.toml"))
    assert cutlist.name == "notes"
    assert cutlist.target == ""
    assert cutlist.created_at == ""
    assert cutlist.params.consistency == C.ASSEMBLE_DEFAULT_CONSISTENCY


def test_a_missing_file_says_so(tmp_path: Path) -> None:
    with pytest.raises(CutlistError, match=r"nope\.toml"):
        load_cutlist(tmp_path / "nope.toml")


def test_broken_toml_reports_the_parser_message(tmp_path: Path) -> None:
    with pytest.raises(CutlistError) as excinfo:
        load_cutlist(write(tmp_path, 'schema_version = 1\nname = "unclosed\n'))
    assert "demo.toml" in str(excinfo.value)
    assert "line" in str(excinfo.value).lower()


def test_a_future_schema_version_is_refused_by_number(tmp_path: Path) -> None:
    with pytest.raises(CutlistError, match="schema_version"):
        load_cutlist(write(tmp_path, MINIMAL.replace("schema_version = 1", "schema_version = 99")))


def test_a_missing_schema_version_is_refused(tmp_path: Path) -> None:
    with pytest.raises(CutlistError, match="schema_version"):
        load_cutlist(write(tmp_path, MINIMAL.replace("schema_version = 1\n", "")))


def test_a_file_with_no_slots_is_refused(tmp_path: Path) -> None:
    with pytest.raises(CutlistError, match="no slots"):
        load_cutlist(write(tmp_path, "schema_version = 1\n"))


def test_an_unknown_slot_kind_lists_the_two_that_exist(tmp_path: Path) -> None:
    with pytest.raises(CutlistError, match="fragment"):
        load_cutlist(write(tmp_path, MINIMAL.replace('kind = "fragment"', 'kind = "clip"')))


def test_a_fragment_without_a_video_is_refused_by_name(tmp_path: Path) -> None:
    with pytest.raises(CutlistError) as excinfo:
        load_cutlist(write(tmp_path, MINIMAL.replace("video_id = 3\n", "")))
    assert "slot 1" in str(excinfo.value)
    assert "video_id" in str(excinfo.value)


def test_a_fragment_without_an_end_is_refused(tmp_path: Path) -> None:
    with pytest.raises(CutlistError, match="end_ms"):
        load_cutlist(write(tmp_path, MINIMAL.replace("end_ms = 2000\n", "")))


def test_an_end_before_the_start_is_refused_with_both_numbers(tmp_path: Path) -> None:
    with pytest.raises(CutlistError) as excinfo:
        load_cutlist(write(tmp_path, MINIMAL.replace("end_ms = 2000", "end_ms = 500")))
    assert "500" in str(excinfo.value)
    assert "1000" in str(excinfo.value)


def test_a_negative_timing_is_refused(tmp_path: Path) -> None:
    with pytest.raises(CutlistError, match="start_ms"):
        load_cutlist(write(tmp_path, MINIMAL.replace("start_ms = 1000", "start_ms = -5")))


def test_a_wrong_type_names_the_field_and_what_was_expected(tmp_path: Path) -> None:
    with pytest.raises(CutlistError) as excinfo:
        load_cutlist(write(tmp_path, MINIMAL.replace("start_ms = 1000", 'start_ms = "1000"')))
    assert "start_ms" in str(excinfo.value)
    assert "integer" in str(excinfo.value)


def test_an_integer_is_accepted_where_a_float_belongs(tmp_path: Path) -> None:
    """TOML distinguishes 1 from 1.0; a person editing by hand does not."""
    cutlist = load_cutlist(write(tmp_path, MINIMAL + "align_score = 1\n"))
    assert cutlist.fragments[0].align_score == 1.0


def test_a_typo_in_a_key_suggests_the_real_one(tmp_path: Path) -> None:
    with pytest.raises(CutlistError) as excinfo:
        load_cutlist(write(tmp_path, MINIMAL.replace("video_id = 3", "video-id = 3\nvideo_id = 3")))
    assert "video-id" in str(excinfo.value)
    assert "video_id" in str(excinfo.value)


def test_a_hand_added_gap_before_survives_the_round_trip(tmp_path: Path) -> None:
    cutlist = load_cutlist(write(tmp_path, MINIMAL + "gap_before_ms = 250\n"))
    assert cutlist.fragments[0].gap_before_ms == 250
    again = load_cutlist(write_cutlist(cutlist, tmp_path / "again.toml"))
    assert again.fragments[0].gap_before_ms == 250


def test_alternatives_and_substitutions_load_back(tmp_path: Path) -> None:
    path = write_cutlist(sample(), tmp_path / "demo.toml")
    cutlist = load_cutlist(path)
    assert cutlist.fragments[0].alternatives[0].video_id == 7
    assert cutlist.fragments[0].alternatives[0].align_score == pytest.approx(0.64)
    assert cutlist.fragments[0].alternatives[0].speaker_label == "guest"
    assert cutlist.gaps[0].substitutions[0].text == "исправит"


def test_a_broken_alternative_names_its_parent_slot(tmp_path: Path) -> None:
    text = MINIMAL + '\n[[slot.alternative]]\nvideo_id = 7\nstart_ms = 0\ntext = "x"\n'
    with pytest.raises(CutlistError) as excinfo:
        load_cutlist(write(tmp_path, text))
    assert "slot 1" in str(excinfo.value)
    assert "alternative 1" in str(excinfo.value)


def test_slot_order_in_the_file_is_the_timeline(tmp_path: Path) -> None:
    text = (
        'schema_version = 1\n'
        '[[slot]]\nkind = "gap"\ntext = "первое"\n'
        '[[slot]]\nkind = "fragment"\ntext = "второе"\nvideo_id = 1\n'
        'start_ms = 0\nend_ms = 10\n'
    )
    assert [slot.text for slot in load_cutlist(write(tmp_path, text)).slots] == [
        "первое",
        "второе",
    ]


def test_read_cutlist_resolves_a_name_under_the_data_tree(data_dir: Path) -> None:
    write_cutlist(sample(), cutlist_path("demo"))
    assert read_cutlist("demo").name == "demo"


def test_read_cutlist_rejects_a_traversing_name(data_dir: Path) -> None:
    with pytest.raises(InvalidInputError):
        read_cutlist("../escape")


# --- recomputing the target from the slots (owner's request) -------------


def test_target_from_slots_reconstructs_an_unedited_target() -> None:
    """Nothing has been hand-edited yet, so joining the slots' own text —
    a fragment's real spoken words, a gap's still-missing target word —
    reproduces exactly what was planned."""
    cutlist = sample()
    assert target_from_slots(cutlist.slots) == cutlist.target


def test_target_from_slots_keeps_a_gaps_word() -> None:
    """A gap slot's text is the target word the corpus never says — the
    recomputed target must still contain it, because the cut list is still
    trying to say it (the render reports it as missing)."""
    cutlist = sample()
    gap = cutlist.gaps[0]
    assert gap.text in target_from_slots(cutlist.slots).split()


def test_target_from_slots_follows_an_adopted_substitution() -> None:
    """The point of adopting a substitution: the target should say what the
    video will actually say, `каннибализм` becoming `каннибализмом`."""
    import dataclasses

    cutlist = sample()
    gap = cutlist.gaps[0]
    adopted = dataclasses.replace(gap, kind="fragment", text="исправит")
    slots = tuple(adopted if slot is gap else slot for slot in cutlist.slots)
    assert target_from_slots(slots) == "мы всё исправит"


def test_target_from_slots_drops_a_removed_fragments_word() -> None:
    """A fragment dropped from the sequence (`d`) is gone from the splice,
    so it is gone from the recomputed target too."""
    cutlist = sample()
    remaining = tuple(slot for slot in cutlist.slots if slot.kind != "fragment")
    assert target_from_slots(remaining) == "исправим"
