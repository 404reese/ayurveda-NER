// AyurNER one-page NLP report. Compile: typst compile report.typ
// Devanagari needs a font such as "Noto Sans Devanagari" installed (falls back otherwise).

#set page(paper: "a4", margin: (x: 1.8cm, y: 1.4cm))
#set text(font: ("Times New Roman", "Liberation Serif", "Mangal", "Noto Serif Devanagari"), size: 10pt, fill: black)
#set par(justify: true, leading: 0.5em, spacing: 0.6em)
#set heading(numbering: none)
#show heading.where(level: 2): it => [#v(0.2em) #text(weight: "bold", size: 11.5pt)[#it.body] #v(0.05em)]
#set list(spacing: 0.4em, indent: 0pt)
#show raw: set text(size: 8pt)

#let shot(label, h: 3.8cm) = rect(
  width: 100%, height: h, stroke: (dash: "dashed", paint: black, thickness: 0.7pt), radius: 3pt,
  align(center + horizon, text(fill: black, size: 8pt)[#label \ (insert screenshot here)]),
)

#align(center)[
  #text(16pt, weight: "bold")[Named Entity Recognition for Ayurvedic Terms]

  #text(10pt)[Natural Language Processing -- Course Project Report \
  Name: *Riddhesh Chaudhary* #h(1em) Roll No: *A-14*]
]

== Problem and Objective
Ayurvedic text is written in Sanskrit and Hindi, in Devanagari, IAST, or casual romanization, so one term has many
surface forms (_vāta_, _vaat_, वात). Generic NER tools do not cover this domain and no annotated corpus is available.
*AyurNER* recognizes and links Ayurvedic entities to canonical IDs regardless of script or spelling, and ships as a
Python library, REST API and web UI.

#v(1fr)
== How It Was Built
The system is *dictionary + rules* (v1), with no training data needed. The pipeline has six stages:
#table(
  columns: (auto, 1fr), stroke: 0.5pt + black, inset: 3pt, align: (left, left),
  [*1. Normalize*], [Unicode NFC, nukta/chandrabindu folding, zero-width char removal; offsets kept to the original text.],
  [*2. Tokenize*], [Script-aware tokens; sentence breaks at danda (।, ॥).],
  [*3. Phonetic key*], [Devanagari → IAST → folded key (diacritics, vowel length, final schwa), so _vāta_ = _vaat_; a looser second tier handles aspiration.],
  [*4. Gazetteer match*], [Longest-match lookup over a phrase index; handles case endings, Hindi plurals, compounds, simple sandhi; optional fuzzy fallback (rapidfuzz).],
  [*5. Rules (TOML)*], [Composition (`DRAVYA + DOSAGE_FORM → FORMULATION`), cue-word disambiguation, context gating for words like _rasa_.],
  [*6. Resolve overlaps*], [Longest span, then strong over weak match, then match-type quality.],
)
The lexicon has about *575 curated entries* across 12 CSV files (herbs, formulations, diseases, doshas, etc.). A
script imports a further *5,082* National Institute of Ayurveda terms from a PDF with a broken font by rebuilding
Devanagari from the clean IAST column. A mining script suggests unseen terms for manual labelling. Built with Python 3.10+,
`indic-transliteration`, FastAPI/uvicorn (API), Streamlit (UI), pytest, Docker.

#v(1fr)
== What It Can Do
#grid(columns: (1.15fr, 1fr), gutter: 10pt,
  [
    - Extracts *12 entity types*: `DRAVYA`, `FORMULATION`, `DOSAGE_FORM`, `ROGA`, `SYMPTOM`, `DOSHA`, `DHATU`, `MALA`, `PROPERTY`, `KARMA`, `PROCEDURE`, `ANATOMY`, `CONCEPT` (labels are extensible).
    - Script-independent linking: "ashwagandha churna", "aśvagandhā cūrṇa" and "अश्वगंधा चूर्ण" give the same entity ID.
    - Composes formulations (herb + dosage form), flags ambiguity and alternatives, returns confidence, span, canonical IAST/Devanagari and cross-references.
    - Matches English equivalents ("fever", "piles"); batch and multi-process streaming.
    - REST endpoints `/v1/extract`, `/v1/extract/batch`, `/v1/lookup`, `/v1/labels`; Docker deployable; CSV/JSON export.
  ],
  [
    ```python
    import ayurner
    ner = ayurner.load()
    doc = ner("अश्वगंधा चूर्ण वात रोगों में उपयोगी है")
    # FORMULATION  अश्वगंधा चूर्ण
    # ROGA         वात रोगों
    ```
  ],
)

#v(1fr)
== Evaluation
On a bundled gold set of *55 sentences / 140 entities* (Hindi, Sanskrit, IAST, Hinglish, English) the system reports
*exact-span micro F1 = 0.99* (Sanskrit weakest, 0.92, due to long dvandva compounds and polysemous _kaṣāya_).
Speed: about 77k tokens/s single process and 148k tokens/s with 4 processes. _Caveat:_ the gold set was built alongside
the lexicon, so it measures regression safety, not independent real-world accuracy.

#v(1fr)
== User Interface
#grid(columns: (1fr, 1fr, 1fr), gutter: 6pt,
figure(
  image("Image1.png", width: 100%),
  caption: [Roman Text input and highlighted entities ]
),
figure(
  image("Image2.png", width: 100%),
  caption: [Devnagri Text input and highlighted entities ]
),
figure(
  image("Image3.png", width: 100%),
  caption: [REST APIs interface ]
)
  // shot("Fig. 1: Text input and highlighted entities"),
  // shot("Fig. 2: Entity table (label, ID, canonical)"),
  // shot("Fig. 3: REST API /docs (Swagger)"),
)

#v(1fr)
== Limitations and Future Work
Recall is limited by lexicon coverage; compounds are returned as one span; phonetic folding is lossy and sandhi handling
is partial; imported NIA labels are heuristic; the lexicon is not yet reviewed by an Ayurveda expert. Future work: train
an ML recognizer (the pipeline already has an `add_recognizer` slot), build an independent annotated test set, and add
expert validation.