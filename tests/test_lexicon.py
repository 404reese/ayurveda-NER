import json

import pytest

from ayurner import Pipeline
from ayurner.labels import CORE_LABELS
from ayurner.lexicon import LexiconError, from_term_lists, load_bundled, read_csv
from ayurner.lexicon.index import LexiconIndex
from ayurner.lexicon.loader import bundled_paths, entry_from_row


def test_bundled_lexicon_is_valid():
    entries = load_bundled()
    assert len(entries) > 500
    ids = [e.id for e in entries]
    assert len(ids) == len(set(ids)), "duplicate ids in bundled lexicon"
    for e in entries:
        assert e.label in CORE_LABELS, e.id
        assert e.canonical and e.devanagari, e.id
    assert len(bundled_paths()) >= 10


def test_bundled_lexicon_has_no_cross_label_collisions_without_review(ner):
    # Collisions are allowed but must stay few; each new one should be a deliberate choice.
    assert len(ner.index.collisions) <= 5, ner.index.collision_report()


def test_missing_script_is_generated():
    e = entry_from_row({"label": "DRAVYA", "canonical_iast": "guḍūcī"})
    assert e.devanagari == "गुडूची"
    e = entry_from_row({"label": "DRAVYA", "devanagari": "गुडूची"})
    assert e.canonical == "guḍūcī"
    assert e.id == "dravya:guduci"


def test_invalid_rows_raise():
    with pytest.raises(LexiconError):
        entry_from_row({"canonical_iast": "x"})
    with pytest.raises(LexiconError):
        entry_from_row({"label": "DRAVYA", "canonical_iast": "x", "requires_context": "maybe"})


def test_weak_forms_marked_with_tilde():
    e = entry_from_row(
        {"label": "DOSHA", "canonical_iast": "samāna vāta", "synonyms": "~समान|समान वायु"}
    )
    forms = {f.text: f.weak for f in e.surface_forms()}
    assert forms["समान"] is True
    assert forms["समान वायु"] is False
    assert forms["samāna vāta"] is False


def test_term_lists_json(tmp_path):
    path = tmp_path / "terms.json"
    path.write_text(
        json.dumps({"herbs": ["गिलोय", "Neem"], "symptoms": ["खुजली"]}, ensure_ascii=False),
        encoding="utf-8",
    )
    entries = from_term_lists(path)
    assert {(e.label, e.devanagari) for e in entries} >= {("DRAVYA", "गिलोय"), ("SYMPTOM", "खुजली")}
    nlp = Pipeline.from_entries(entries)
    assert [e.label for e in nlp("giloy aur neem")] == ["DRAVYA", "DRAVYA"]


def test_custom_csv_and_custom_label(tmp_path):
    p = tmp_path / "mine.csv"
    p.write_text(
        "id,label,canonical_iast,devanagari,synonyms\n"
        "asana:vajrasana,YOGA_ASANA,vajrāsana,वज्रासन,vajrasan\n",
        encoding="utf-8",
    )
    entries = read_csv(p)
    nlp = Pipeline.from_entries(entries)
    ents = nlp("भोजन के बाद वज्रासन करें").entities
    assert [(e.label, e.entity_id) for e in ents] == [("YOGA_ASANA", "asana:vajrasana")]


def test_index_collision_report_mentions_ids():
    a = entry_from_row({"id": "a", "label": "DRAVYA", "canonical_iast": "tvak"})
    b = entry_from_row({"id": "b", "label": "ANATOMY", "canonical_iast": "tvak"})
    idx = LexiconIndex.build([a, b])
    assert idx.collisions and "a, b" in idx.collision_report()
