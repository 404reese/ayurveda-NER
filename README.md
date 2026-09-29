# ayurner

Named-entity recognition for **Ayurvedic terms in Sanskrit and Hindi**, in Devanagari, IAST,
or casual romanization ("ashwagandha churna", "vaat rog"). Use it as a Python library or as a
REST service.

```python
import ayurner

ner = ayurner.load()
doc = ner("अश्वगंधा चूर्ण वात रोगों में उपयोगी है")
for e in doc.entities:
    print(e.text, e.label, e.entity_id, e.canonical)
# अश्वगंधा चूर्ण   FORMULATION  composite:dravya:ashvagandha+dosage_form:curna  aśvagandhā cūrṇa
# वात रोगों       ROGA         roga:vatavyadhi                                 vātavyādhi
```

The same sentence written as `ashwagandha churna vaat rogon mein upyogi hai` or
`aśvagandhā cūrṇa vāta roga` returns the same entity IDs.

v1 is **dictionary + rules**, with no ML model and no training data needed. The pipeline has a
recognizer slot, so a trained model can be added later without changing the API.

## Install

```bash
pip install -e .                 # library
pip install -e ".[server]"       # + REST API (FastAPI, uvicorn)
pip install -e ".[fuzzy,pdf]"    # + misspelling fallback, PDF import tools
pip install -e ".[dev]"          # + tests and linting
```

Python 3.10+. The only core dependency is `indic-transliteration`.

## Entity labels

| Label | What | Examples |
|---|---|---|
| `DRAVYA` | herbs, minerals, animal products | अश्वगंधा, giloy, शिलाजीत, madhu |
| `FORMULATION` | classical preparations and groups | त्रिफला, च्यवनप्राश, yograj guggul, ashokarishta |
| `DOSAGE_FORM` | kalpana | चूर्ण, क्वाथ/काढ़ा, वटी, घृत, तैल, भस्म |
| `ROGA` | diseases | ज्वर/बुखार, आमवात, अम्लपित्त, piles |
| `SYMPTOM` | lakshana | कब्ज, उल्टी, अरुचि, headache |
| `DOSHA` | doshas and sub-doshas | वात, pitta, समान वायु, पाचक पित्त |
| `DHATU`, `MALA` | tissues and wastes | रक्त धातु, ओजस्, मूत्र |
| `PROPERTY` | rasa, guna, virya, vipaka | मधुर रस, शीत वीर्य, guru guṇa |
| `KARMA` | pharmacological actions | दीपन, पाचन, रसायन, medhya |
| `PROCEDURE` | therapies | पंचकर्म, वमन, शिरोधारा, abhyanga |
| `ANATOMY` | srotas, marma, organs | प्राणवह स्रोतस्, हृदय |
| `CONCEPT` | other concepts | प्रकृति, अग्नि, पथ्य, दिनचर्या |

You can add your own labels. Any label that appears in a lexicon file is registered automatically.

## What each entity contains

```python
Entity(text, start, end,          # span in the ORIGINAL text
       label, subtype, entity_id,  # e.g. DRAVYA / herb / dravya:guduci
       canonical, devanagari,      # guḍūcī / गुडूची
       confidence, match_type,     # exact | phonetic | stem | loose | compound | compose | fuzzy
       ambiguous, alternatives,    # other entity ids that fit the same words
       components,                 # parts of compounds / compositions
       metadata)                   # english, hindi, botanical, xrefs, source ...
```

`doc.to_json()` / `doc.to_dict()` serialize everything.

## API

```python
ner = ayurner.load(
    lexicons=["data/external/nia.csv", "my_terms.csv"],  # extra lexicons (.csv/.jsonl/.json)
    match_english=True,      # also match English equivalents ("fever", "piles")
    fuzzy=False,             # edit-distance fallback for misspellings (needs rapidfuzz)
    labels={"DRAVYA", "ROGA"},
    min_confidence=0.0,
)
ner(text)                           # -> Document
ner.extract(text)                   # -> list[Entity]
ner.pipe(texts, batch_size=64, n_process=4)   # streaming, ordered, multi-process
ner.lookup("giloy")                 # -> lexicon entries, by any name in any script
ner.index.collision_report()        # keys shared by entries with different labels

ayurner.transliterate("शुक्र धातु")              # 'śukra dhātu'
ayurner.transliterate("śukra dhātu", to="devanagari")
```

`load()` is memoized per configuration. The compiled index is cached on disk, keyed by the lexicon
contents, so loading is near-instant after the first run. Set `AYURNER_CACHE_DIR` to move the
cache.

On Windows and macOS, call `pipe(..., n_process>1)` under `if __name__ == "__main__":`.

## How matching works

1. **Normalize**: NFC; remove nukta; fold chandrabindu into anusvara; strip zero-width
   joiners. Offsets always map back to the original string.
2. **Tokenize**: script-aware tokens; segments split at `।`, `॥` and `.`.
3. **Phonetic keys**: every token and every lexicon form becomes a key.
   - Devanagari is transliterated to IAST, then folded: diacritics removed, vowel length ignored,
     nasal spellings unified (गंध = गन्ध), doubled consonants collapsed, final schwa dropped
     (vāta = vaat).
   - Casual spellings are also folded: `w/v`, `ch/c`, `ee/i`, `ph/f`.
   - Lexicon forms also get Hindi schwa-deletion variants (dālacīnī → dalchini) and
     final-u variants (guggulu → guggul).
   - A looser second tier handles inconsistent aspiration in plain-ASCII input (Vatha = Vata,
     Swasa = Shvasa).
4. **Gazetteer**: longest match over a phrase index. It handles Sanskrit case endings and Hindi
   plurals (वातरोगेषु, रोगों), joined spellings (ashokarishta, त्रिफलाचूर्ण), and compound
   splitting with dīrgha sandhi (aśoka + ariṣṭa).
5. **Rules** (`lexicon/data/rules.toml`, editable):
   - **compose**: `DRAVYA + DOSAGE_FORM → FORMULATION` (giloy kadha),
     `DOSHA + roga → ROGA`, sub-doshas.
   - **disambiguation** by cue words and neighbouring labels: रस as taste vs. tissue,
     "amla" as sour vs. āmalakī, त्वक् as skin vs. cinnamon.
   - **context requirements** for words that are also everyday Hindi: समान "same",
     गुरु "teacher", पाचक, बल. These are only tagged when there is Ayurvedic context.
6. **Resolve** overlaps: longest span first, then strong over weak, then the better match type.

## Lexicons

The bundled core lexicon (`src/ayurner/lexicon/data/*.csv`) has about 575 entries with
synonyms in Devanagari, IAST, Hindi and English. It was built from a curated term list plus
additions. **It needs review by an Ayurveda expert before production use.**

CSV columns, designed to be editable in Excel:

```
id,label,subtype,canonical_iast,devanagari,synonyms,hindi,english,definition,botanical,xrefs,requires_context,notes,source
```

- `synonyms`, `hindi`, `english`: separated by `|`, in any script. A leading `~` marks a *weak*
  form that needs context (`~समान`).
- `requires_context`: `yes` makes the canonical and Devanagari forms weak; `compose` means
  the entry is only used inside compounds (e.g. the generic word *roga*).
- A missing `devanagari` or `canonical_iast` is generated automatically.
- Single-word English equivalents are weak, except for ROGA, FORMULATION, PROCEDURE and herbs.

For quick additions, a `{"category": ["term", ...]}` JSON file also works:
`ayurner.load(lexicons=["terms.json"])`.

### NIA / WHO-APW terminology (5,000+ terms)

The National Institute of Ayurveda's
[Non-Clinical Terminologies of Ayurveda](https://www.nia.nic.in/pdf/TERMINOLOGIES.pdf) PDF has
about 5,200 entries (code, Devanagari, IAST, English). Its Devanagari column is garbled by a
legacy font, but the IAST column is clean. The importer rebuilds the Devanagari from the IAST:

```bash
python scripts/import_nia_terminologies.py TERMINOLOGIES.pdf -o data/external/nia.csv
```

Result: 5,082 entries, 0 rejected, and 104 skipped because they duplicate core terms. Labels are
provisional, derived from the discipline code (AF → SYMPTOM, SR → ANATOMY, DG → KARMA/DRAVYA/…).
The file has no explicit licence, so it is **not bundled**. Load it with `lexicons=[...]` or
`AYURNER_LEXICONS`.

### Mining other documents

```bash
python scripts/mine_pdf_terms.py book.pdf -o candidates.csv --min-freq 3
```

This lists frequent words and phrases that the lexicon does not yet recognize, with context, so
you can label them. It warns when a PDF's text layer is garbled or missing; those files need OCR
first. Only mine documents you have the right to use. For example, the Scribd copy of
*Āyurvedīyaḥ Saṃskṛtaśabdakośaḥ* cannot be scraped, and its content must not be redistributed.

## REST API

```bash
python -m ayurner serve --host 0.0.0.0 --port 8000 --workers 4
```

```bash
curl -X POST localhost:8000/v1/extract -H "Content-Type: application/json" -d "{\"text\": \"triphala churna kapha\"}"
```

| Endpoint | |
|---|---|
| `POST /v1/extract` | `{text, labels?, min_confidence?}` → `{entities, lexicon_version}` |
| `POST /v1/extract/batch` | `{texts: [...]}` → `{results: [...]}` |
| `GET /v1/lookup?q=giloy` | lexicon entries for a name |
| `GET /v1/labels`, `/health`, `/version` | |
| `GET /docs` | interactive OpenAPI docs |

The service is stateless and CPU-bound. Scale it with `--workers` and horizontally behind a load
balancer. Configure it with environment variables: `AYURNER_LEXICONS`,
`AYURNER_MAX_TEXT_CHARS` (default 100000), `AYURNER_MAX_BATCH` (default 256), `AYURNER_FUZZY`,
`AYURNER_MATCH_ENGLISH`, `AYURNER_CORS_ORIGINS`. The service has no authentication, so put it
behind your gateway.

Docker (the index is compiled at build time):

```bash
docker build -t ayurner .
docker run -p 8000:8000 -e WORKERS=4 ayurner
# with the NIA lexicon: copy nia.csv into ./lexicons, then
docker build --build-arg AYURNER_LEXICONS=/app/lexicons/nia.csv -t ayurner .
```

## Command line

```bash
python -m ayurner build-index --collisions        # compile + show key collisions
python -m ayurner extract texts.txt -o out.jsonl -j 4
```

## Export (silver training data)

```python
from ayurner import export
export.write(ner.pipe(texts), "silver.conll")     # token<TAB>BIO tag
export.write(ner.pipe(texts), "tasks.json")       # Label Studio pre-annotations
export.write(ner.pipe(texts), "out.jsonl")
```

Correct the pre-annotations in Label Studio to get training data for a future ML recognizer.
Plug that recognizer in with `ner.add_recognizer(obj)`, where `obj` has
`recognize(ctx) -> list[Candidate]`.

## Evaluation and performance

```bash
python scripts/evaluate.py tests/fixtures/gold.jsonl --errors
python scripts/benchmark.py --docs 4000 --processes 4
```

Baseline on the bundled gold set (55 sentences in Hindi, Sanskrit, IAST, Hinglish and English;
140 entities): **exact-span micro F1 0.99**. Sanskrit is the weakest at F1 0.92: long dvandva
compounds and words like कषाय (decoction vs. astringent taste) are still missed. **Caveat:** this set
was written alongside the lexicon, so it only guards against regressions and does not measure
real-world accuracy. Add sentences from your own sources to get a meaningful score.

Throughput on an entity-dense synthetic corpus (Windows laptop, Python 3.12):

- about 77k tokens/s on 1 process;
- about 148k tokens/s with `pipe(n_process=4)`, including worker start-up;
- about 50k tokens/s with the NIA lexicon loaded.

## Known limitations

- Recall is bounded by the lexicon. To find new terms, mine documents and review the results, or
  train an ML recognizer on the exported silver data.
- A compound word is returned as one entity with `components`. Sub-token offsets are not
  available yet.
- Phonetic folding is deliberately lossy. Rare collisions show up in `collision_report()` and are
  resolved by context.
- Sandhi handling covers case endings, common enclitics (-श्च, -इति) and dīrgha sandhi. It is not
  a full sandhi splitter.
- NIA labels are heuristic and provisional.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src tests scripts
```

## License

The code is MIT. Bundled lexicon data is `source=user-seed` or `seed-v0` (see the `source` column).
Third-party terminologies such as NIA are not redistributed.
