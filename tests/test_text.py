import pytest

from ayurner.text.normalize import normalize
from ayurner.text.phonetic import lexicon_token_keys, loose_key, schwa_delete_iast, strict_key
from ayurner.text.stem import stem_candidates
from ayurner.text.tokenize import tokenize
from ayurner.text.translit import transliterate


def test_normalize_keeps_offsets_into_original():
    text = "अड़ूसा और आँवला"
    norm = normalize(text)
    assert "़" not in norm.text  # nukta removed
    assert "ं" in norm.text  # chandrabindu folded to anusvara
    toks = tokenize(norm.text)
    s, e = norm.to_original(toks[0].start, toks[0].end)
    assert text[s:e] == "अड़ूसा"
    s, e = norm.to_original(toks[2].start, toks[2].end)
    assert text[s:e] == "आँवला"


def test_zero_width_joiners_removed():
    assert normalize("क्‍ष").text == "क्ष"


def test_tokenizer_segments_and_joinability():
    toks = tokenize("वात-पित्त, कफ। triphala churna")
    assert [t.text for t in toks] == ["वात", "पित्त", "कफ", "triphala", "churna"]
    assert toks[1].joinable  # hyphen keeps words joinable
    assert not toks[2].joinable  # comma breaks
    assert toks[3].segment == toks[2].segment + 1  # danda starts a new segment
    assert toks[3].is_devanagari is False and toks[0].is_devanagari


@pytest.mark.parametrize(
    "variants",
    [
        ["अश्वगन्धा", "अश्वगंधा", "aśvagandhā", "ashwagandha", "Ashvagandha", "ashvagandh"],
        ["वात", "vāta", "vaat", "vata"],
        ["कफ", "kapha", "kaf", "kafa"],
        ["पित्त", "pitta", "pitt"],
        ["शूंठी", "शुण्ठी", "shunthi", "śuṇṭhī"],
        ["छर्दी", "छर्दि", "chhardi", "chardi"],
        ["कपिकच्छु", "kapikachhu", "kapikacchu"],
        ["अड़ूसा", "adusa"],
        ["जीरा", "ज़ीरा", "jeera", "zeera"],
        ["रसः", "rasa", "रस"],
    ],
)
def test_strict_key_unifies_scripts_and_spellings(variants):
    keys = {strict_key(v) for v in variants}
    assert len(keys) == 1, keys


def test_loose_key_handles_aspiration_and_sibilants():
    assert loose_key("vatha") == loose_key("vata")
    assert loose_key("swasa") == loose_key("shvasa")
    assert strict_key("vatha") != strict_key("vata")


@pytest.mark.parametrize(
    "iast,expected",
    [
        ("dālacīnī", "dālcīnī"),
        ("ajavāina", "ajvāina"),
        ("nāgaramothā", "nāgarmothā"),
        ("yogarāja", "yogrāja"),
        ("daśamūlāriṣṭa", "daśmūlāriṣṭa"),
    ],
)
def test_hindi_schwa_deletion(iast, expected):
    assert schwa_delete_iast(iast) == expected


def test_lexicon_keys_cover_hindi_pronunciation():
    assert strict_key("dalchini") in lexicon_token_keys("दालचीनी")
    assert strict_key("guggul") in lexicon_token_keys("गुग्गुलु")
    assert strict_key("shankhpushpi") in lexicon_token_keys("शङ्खपुष्पी")


def test_stems_strip_sanskrit_and_hindi_endings():
    assert "rog" in stem_candidates(strict_key("रोगेषु", drop_final_schwa=False))
    assert "rog" in stem_candidates(strict_key("रोगों", drop_final_schwa=False))
    assert "rog" in stem_candidates(strict_key("rogon", drop_final_schwa=False))
    assert "jvar" in stem_candidates(strict_key("ज्वरेण", drop_final_schwa=False))


def test_transliterate_round_trip():
    assert transliterate("शुक्र धातु") == "śukra dhātu"
    assert transliterate("śukra dhātu", to="devanagari") == "शुक्र धातु"
    assert transliterate("saṁhitā", to="devanagari") == "संहिता"
    with pytest.raises(ValueError):
        transliterate("x", to="tamil")
