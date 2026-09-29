import io
import json

from fastapi.testclient import TestClient

from ayurner import export
from ayurner.lexicon.nia import import_entries, parse_body, provisional_label, split_entries

# Raw text as pypdf extracts it from page 301 of TERMINOLOGIES.pdf (garbled Devanagari column).
NIA_PAGE = """300

SV0259 शुतर िस्त्र śukla vastra White cloth; alleviates vata Doṣa.
SK0186 शुक्र धातु śukra dhātu The seventh Dhātu, whose function is reproduction. Generally equated
with Semen; Present in two forms: 1. Pervading the entire body 2.
Fertilizing the ovum. It is dominant of Jala Mahābhūta.
AF2034 शुक्राबॊ भूत्र śukrābhaṁ mūtra Urine in the form of semen
DG0314 शुक्रजनन śukrajanana substances which enhances Shukra (semen / sperm)
AF2037 शुक्रप्रितशनॊ भूत्रमुत्
ऩाक् ऩश्चात् िा
śukrapravartanaṁ
mūtrayutaḥ pāk paścāt
vā
Ejaculation before with or after micturation
AF0010 आगायधूभाबभ् āgāradhῡmābhaṁ Smoky color
SR0142 घुर्टका / घुत्ण्टका ghuṭikā / ghuṇṭikā Malleolus - Bone of the middle ear
"""


def test_nia_split_and_parse():
    entries = dict(split_entries(NIA_PAGE))
    assert len(entries) == 7
    assert parse_body(entries["SK0186"])[0] == "śukra dhātu"
    assert parse_body(entries["AF2034"]) == ("śukrābhaṃ mūtra", "Urine in the form of semen")
    assert parse_body(entries["AF2037"])[0] == "śukrapravartanaṃ mūtrayutaḥ pāk paścāt vā"
    assert parse_body(entries["AF0010"])[0] == "āgāradhūmābhaṃ"


def test_nia_rows_regenerate_devanagari_and_labels():
    rows, rejects, stats = import_entries(split_entries(NIA_PAGE))
    by = {r["id"]: r for r in rows}
    assert not rejects and stats.written == 7
    assert by["nia:SK0186"]["devanagari"] == "शुक्र धातु"
    assert by["nia:SK0186"]["label"] == "DHATU"
    assert by["nia:DG0314"]["label"] == "KARMA"
    assert by["nia:AF2034"]["label"] == "SYMPTOM"
    assert by["nia:SR0142"]["canonical_iast"] == "ghuṭikā"
    assert by["nia:SR0142"]["synonyms"] == "ghuṇṭikā"
    assert by["nia:AF2034"]["english"] == ""  # five words: kept only as definition
    assert by["nia:SV0259"]["english"] == "White cloth"


def test_nia_skips_terms_already_in_core():
    rows, _, stats = import_entries(split_entries(NIA_PAGE), {("shukr", "dhatu"): {"DHATU"}})
    assert stats.skipped_existing == 1 and "nia:SK0186" not in {r["id"] for r in rows}


def test_provisional_label_rules():
    assert (
        provisional_label("DG0001", "x", "Combination of following four plants")[0] == "FORMULATION"
    )
    assert (
        provisional_label("RS0001", "x", "Melting and pouring into cold liquids")[0] == "PROCEDURE"
    )
    assert provisional_label("SR0001", "x", "Deep tendons")[0] == "ANATOMY"


def test_export_formats(ner):
    docs = [ner("त्रिफला चूर्ण और वात")]
    bio = export.to_bio(docs[0])
    assert bio == [
        ("त्रिफला", "B-FORMULATION"),
        ("चूर्ण", "I-FORMULATION"),
        ("और", "O"),
        ("वात", "B-DOSHA"),
    ]
    buf = io.StringIO()
    assert export.to_jsonl(docs, buf) == 1
    assert json.loads(buf.getvalue())["entities"][0]["label"] == "FORMULATION"
    tasks = export.to_label_studio(docs)
    assert tasks[0]["predictions"][0]["result"][0]["value"]["labels"] == ["FORMULATION"]


def test_rest_api(ner):
    from ayurner.server import create_app

    with TestClient(create_app(ner)) as client:
        r = client.post("/v1/extract", json={"text": "giloy aur tulsi ka kadha"})
        assert r.status_code == 200
        body = r.json()
        assert [e["label"] for e in body["entities"]] == ["DRAVYA", "DRAVYA", "DOSAGE_FORM"]
        assert body["entities"][0]["entity_id"] == "dravya:guduci"
        assert body["lexicon_version"]

        r = client.post("/v1/extract", json={"text": "vata pitta triphala", "labels": ["DOSHA"]})
        assert {e["label"] for e in r.json()["entities"]} == {"DOSHA"}

        r = client.post("/v1/extract/batch", json={"texts": ["vata", "कफ"]})
        assert [len(x["entities"]) for x in r.json()["results"]] == [1, 1]

        r = client.get("/v1/lookup", params={"q": "गिलोय"})
        assert r.json()["entries"][0]["id"] == "dravya:guduci"

        assert client.get("/health").json()["status"] == "ok"
        assert "DRAVYA" in client.get("/v1/labels").json()


def test_rest_api_limits(ner, monkeypatch):
    from ayurner.server import create_app

    monkeypatch.setenv("AYURNER_MAX_TEXT_CHARS", "10")
    monkeypatch.setenv("AYURNER_MAX_BATCH", "2")
    with TestClient(create_app(ner)) as client:
        assert client.post("/v1/extract", json={"text": "x" * 11}).status_code == 413
        assert client.post("/v1/extract/batch", json={"texts": ["a", "b", "c"]}).status_code == 413
