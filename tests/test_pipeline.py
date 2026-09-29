import pytest
from conftest import spans

import ayurner


def ids(doc):
    return [e.entity_id for e in doc.entities]


def test_same_entities_across_scripts(ner):
    deva = ner("अश्वगंधा चूर्ण वात रोगों में उपयोगी है")
    hinglish = ner("ashwagandha churna vaat rogon mein upyogi hai")
    iast = ner("aśvagandhā cūrṇa vāta roga")
    assert ids(deva) == ids(hinglish) == ids(iast)
    assert [e.label for e in deva] == ["FORMULATION", "ROGA"]
    assert deva.entities[1].entity_id == "roga:vatavyadhi"


def test_offsets_point_into_original_text(ner):
    text = "रोगी को  अड़ूसा का काढ़ा दें"
    for e in ner(text).entities:
        assert text[e.start : e.end] == e.text
    assert spans(ner(text)) == [("अड़ूसा", "DRAVYA"), ("काढ़ा", "DOSAGE_FORM")]


def test_sanskrit_compound_and_case_endings(ner):
    doc = ner("अश्वगन्धाचूर्णम् वातरोगेषु हितम्।")
    assert spans(doc) == [("अश्वगन्धाचूर्णम्", "FORMULATION"), ("वातरोगेषु", "ROGA")]
    comp = doc.entities[0]
    assert comp.match_type == "compound"
    assert [c.entity_id for c in comp.components] == ["dravya:ashvagandha", "dosage_form:curna"]


def test_multiword_formulations_and_hindi_spellings(ner):
    doc = ner("Mahanarayan tel se abhyanga karein, fir yograj guggul lein")
    assert ids(doc) == [
        "formulation:mahanarayana_taila",
        "procedure:abhyanga",
        "formulation:yogaraja_guggulu",
    ]


def test_joined_spellings(ner):
    assert ids(ner("ashokarishta aur dashmularishta")) == [
        "formulation:ashokarishta",
        "formulation:dashamularishta",
    ]


def test_composition_rule_dravya_plus_dosage_form(ner):
    doc = ner("Giloy kadha")
    assert spans(doc) == [("Giloy kadha", "FORMULATION")]
    assert [c.entity_id for c in doc.entities[0].components] == [
        "dravya:guduci",
        "dosage_form:kvatha",
    ]


def test_weak_forms_need_context(ner):
    assert spans(ner("यह समान है")) == []
    assert spans(ner("समान वायु नाभि में रहता है")) == [("समान वायु", "DOSHA")]
    assert spans(ner("मुझे गुरु जी ने कहा")) == []
    assert spans(ner("उस पर ध्यान दें")) == []


def test_disambiguation_by_context(ner):
    taste = [e for e in ner("मधुर और अम्ल रस") if "अम्ल" in e.text]
    assert taste and taste[0].label == "PROPERTY"
    assert ner("amla").entities[0].label == "PROPERTY"  # no cue: the strong sense wins
    fruit = [e for e in ner("amla churna with honey") if "amla" in e.text.lower()]
    assert fruit and fruit[0].label in ("FORMULATION", "DRAVYA")


def test_english_equivalents_toggle():
    on = ayurner.load(match_english=True)
    off = ayurner.load(match_english=False)
    assert [e.entity_id for e in on("patient has fever and piles")] == ["roga:jvara", "roga:arsha"]
    assert off("patient has fever and piles").entities == ()


def test_label_filter_and_min_confidence():
    ner = ayurner.load(labels=["DOSHA"])
    assert {e.label for e in ner("vata pitta aur triphala")} == {"DOSHA"}
    strict = ayurner.load(min_confidence=0.99)
    assert all(e.confidence >= 0.99 for e in strict("vaat rogon mein triphala"))


def test_load_is_memoized():
    assert ayurner.load() is ayurner.load()


def test_lookup_any_script(ner):
    for q in ("giloy", "गुडूची", "guḍūcī", "Amrita", "dravya:guduci"):
        assert "dravya:guduci" in [e.id for e in ner.lookup(q)], q


def test_document_serialization(ner):
    doc = ner("त्रिफला चूर्ण")
    d = doc.to_dict()
    assert d["entities"][0]["label"] == "FORMULATION"
    assert "त्रिफला" in doc.to_json()


def test_pipe_single_process_preserves_order(ner):
    texts = ["vata", "पित्त", "kapha", ""]
    docs = list(ner.pipe(texts))
    assert [d.text for d in docs] == texts
    assert [ids(d) for d in docs] == [["dosha:vata"], ["dosha:pitta"], ["dosha:kapha"], []]


def test_pipe_multi_process(ner):
    texts = ["triphala churna", "वात रोग"] * 20
    docs = list(ner.pipe(texts, batch_size=7, n_process=2))
    assert len(docs) == 40
    assert ids(docs[0]) == ids(ner("triphala churna"))
    assert ids(docs[1]) == ids(ner("वात रोग"))


def test_pipe_multi_process_requires_load_config():
    p = ayurner.Pipeline.from_entries([])
    with pytest.raises(RuntimeError):
        list(p.pipe(["x"], n_process=2))


def test_fuzzy_catches_misspelling():
    fuzzy = ayurner.load(fuzzy=True)
    got = [
        e
        for e in fuzzy("shatavaaari aur punarnwa")
        if e.match_type in ("fuzzy", "phonetic", "exact")
    ]
    assert "dravya:punarnava" in [e.entity_id for e in got]


def test_custom_recognizer_hook(ner):
    from ayurner.match.base import Candidate, Sense

    class Always:
        def recognize(self, ctx):
            e = ctx.index.get("dosha:vata")
            return [Candidate(0, 1, [Sense(e)], "phonetic", 0.5, source="custom")]

    p = ayurner.Pipeline(index=ner.index)
    p.add_recognizer(Always())
    assert [e.entity_id for e in p("xyz")] == ["dosha:vata"]
