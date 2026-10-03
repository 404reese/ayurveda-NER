# AyurNER: A Dictionary- and Rule-Based Named Entity Recognition System for Ayurvedic Terminology

## Abstract

AyurNER is a named-entity recognition (NER) system for identifying Ayurvedic terms in
Sanskrit and Hindi text, including Devanagari, IAST-transliterated, and casually romanized
("Hinglish") input. Unlike conventional NER systems that rely on statistical or neural
models trained on annotated corpora, AyurNER (v1) uses a dictionary-and-rule-based
architecture, requiring no training data while achieving a reported exact-span micro F1
of 0.99 on its bundled evaluation set. The system is designed as a Python library and
REST service, and includes a pipeline extension point for a future machine-learning
recognizer.

## 1. Introduction

Ayurveda, the traditional Indian medical system, is documented primarily in Sanskrit and
Hindi, with terminology that spans multiple writing systems (Devanagari) and multiple
transliteration conventions (IAST, informal Romanization). This multiplicity of surface
forms — e.g., *vāta*, *vaat*, and *वात* all referring to the same dosha — creates a
challenging entity-linking problem distinct from NER in English biomedical text. AyurNER
addresses this by normalizing and phonetically folding input across scripts before
matching against a curated lexicon.

## 2. System Architecture

The pipeline consists of six stages:

1. **Normalization** — Unicode NFC normalization, nukta removal, chandrabindu folding,
   and zero-width joiner stripping, while preserving character offsets into the original
   text.
2. **Tokenization** — script-aware segmentation, with sentence boundaries at Devanagari
   danda punctuation (`।`, `॥`).
3. **Phonetic key generation** — Devanagari is transliterated to IAST and folded
   (diacritics removed, vowel length and nasalization normalized, final schwa dropped),
   so that *vāta* and *vaat* resolve to the same key. A looser secondary tier tolerates
   inconsistent aspiration in ASCII input.
4. **Gazetteer matching** — a longest-match lookup over a phrase index, handling
   Sanskrit case endings, Hindi pluralization, joined compounds, and sandhi splitting.
5. **Rule application** (user-editable, TOML-configured) — composition rules (e.g.
   `DRAVYA + DOSAGE_FORM → FORMULATION`), disambiguation by cue words, and context
   gating for polysemous everyday words (e.g. *rasa* as "taste" vs. "tissue").
6. **Overlap resolution** — longest span wins, then strong match over weak, then
   match-type quality.

Each extracted `Entity` carries its span, label, subtype, a canonical entity ID,
IAST/Devanagari canonical forms, a confidence score, match type, and cross-references.

## 3. Entity Schema

The system defines twelve entity categories relevant to Ayurvedic texts: `DRAVYA`
(substances), `FORMULATION`, `DOSAGE_FORM`, `ROGA` (diseases), `SYMPTOM`, `DOSHA`,
`DHATU`/`MALA` (tissues/wastes), `PROPERTY`, `KARMA` (actions), `PROCEDURE`, `ANATOMY`,
and `CONCEPT`. Labels are extensible: any label present in a lexicon file is registered
automatically.

## 4. Lexical Resources

The bundled core lexicon contains approximately 575 curated entries with multi-script
synonyms. A separate importer converts the National Institute of Ayurveda's published
terminology (~5,200 entries) from a PDF with a corrupted Devanagari font into clean
structured data by reconstructing Devanagari from the (clean) IAST column, yielding
5,082 usable, non-duplicate entries with provisional labels inferred from discipline
codes. A document-mining script supports semi-automatic lexicon growth by surfacing
frequent unrecognized terms with context for manual annotation.

## 5. Evaluation

On a bundled gold set of 55 sentences (140 entities) spanning Hindi, Sanskrit, IAST,
Hinglish, and English, the system reports an exact-span micro F1 of 0.99, with Sanskrit
as the weakest subset (F1 0.92), primarily due to long dvandva compounds and
polysemous terms such as *kaṣāya*. The authors note this gold set was constructed
alongside the lexicon and therefore measures regression safety rather than independent
real-world accuracy.

Throughput benchmarks (Windows, Python 3.12) report approximately 77,000 tokens/second
single-process and approximately 148,000 tokens/second with four-process parallel
pipelining.

## 6. Limitations

- Recall is bounded by lexicon coverage; unseen terms require mining and manual review,
  or a trained ML recognizer.
- Compound words are returned as single entities without sub-token offsets.
- Phonetic folding is intentionally lossy, producing rare key collisions that must be
  resolved by context.
- Sandhi handling is partial (case endings, common enclitics, dīrgha sandhi only).
- Labels imported from the NIA terminology are heuristic and unvalidated.
- The bundled lexicon has not yet been reviewed by a domain (Ayurveda) expert.

## 7. Future Work

The architecture reserves an extension point (`ner.add_recognizer`) for a
machine-learning-based recognizer. The system already supports exporting
pipeline output as BIO-tagged CoNLL data or Label Studio pre-annotation tasks, intended
to be corrected by human annotators to produce supervised training data for such a
model.

## 8. Conclusion

AyurNER demonstrates that a carefully engineered dictionary-and-rule pipeline —
combining script-aware normalization, phonetic folding, gazetteer matching, and
compositional rules — can achieve high precision and recall on Ayurvedic entity
extraction without any training data, while remaining extensible toward a future
statistical or neural recognition stage.

---
*This report summarizes the AyurNER project based on its repository documentation and
source layout as of the current codebase state.*
