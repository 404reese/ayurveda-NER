"""Interactive tester for ayurner.

Run:  streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

import ayurner
from ayurner.labels import CORE_LABELS

st.set_page_config(page_title="ayurner – Ayurvedic NER", page_icon=":material/spa:", layout="wide")

NIA_PATH = Path(__file__).parent / "data" / "external" / "nia.csv"

LABEL_COLORS = {
    "DRAVYA": "green",
    "FORMULATION": "blue",
    "DOSAGE_FORM": "violet",
    "ROGA": "red",
    "SYMPTOM": "orange",
    "DOSHA": "yellow",
    "DHATU": "red",
    "MALA": "gray",
    "PROPERTY": "violet",
    "KARMA": "blue",
    "PROCEDURE": "green",
    "ANATOMY": "orange",
    "CONCEPT": "gray",
}

EXAMPLES = {
    "Hindi": "अश्वगंधा चूर्ण वात रोगों में उपयोगी है। गिलोय और तुलसी का काढ़ा बुखार में दें।",
    "Sanskrit": "अश्वगन्धाचूर्णम् वातरोगेषु हितम्। हरीतकी आमलकी विभीतकश्च त्रिफला।",
    "Hinglish": "Mahanarayan tel se abhyanga karein, fir yograj guggul sandhivata mein lein.",
    "IAST": "madhura rasa, guru guṇa, śīta vīrya; prāṇavaha srotas",
    "Tricky": "यह समान है, पर समान वायु नाभि में रहता है। amla churna vs amla rasa.",
}

_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]<>()#+\-.!|~$:])")


def md_escape(text: str) -> str:
    return _MD_SPECIAL.sub(r"\\\1", text).replace("\n", "  \n")


@st.cache_resource(max_entries=4, show_spinner="Loading lexicon…")
def get_ner(use_nia: bool, match_english: bool, fuzzy: bool):
    lexicons = [str(NIA_PATH)] if use_nia else []
    return ayurner.load(lexicons=lexicons, match_english=match_english, fuzzy=fuzzy)


def highlighted(text: str, entities) -> str:
    """Markdown with each entity as a coloured background span plus a small label."""
    out, pos = [], 0
    for e in entities:
        out.append(md_escape(text[pos : e.start]))
        color = LABEL_COLORS.get(e.label, "gray")
        surface = md_escape(e.text).replace("[", "(").replace("]", ")")
        out.append(f":{color}-background[**{surface}**] :{color}[{e.label.lower()}]")
        pos = e.end
    out.append(md_escape(text[pos:]))
    return "".join(out)


def entities_frame(entities) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "text": e.text,
                "label": e.label,
                "subtype": e.subtype,
                "canonical": e.canonical,
                "devanagari": e.devanagari,
                "entity_id": e.entity_id,
                "match": e.match_type,
                "confidence": e.confidence,
                "ambiguous": e.ambiguous,
                "components": " + ".join(c.canonical for c in e.components),
                "english": ", ".join(e.metadata.get("english", [])),
                "botanical": e.metadata.get("botanical", ""),
            }
            for e in entities
        ]
    )


ENTITY_COLUMNS = {
    "confidence": st.column_config.ProgressColumn("confidence", min_value=0.0, max_value=1.0, format="%.2f"),
    "ambiguous": st.column_config.CheckboxColumn("ambiguous"),
}

# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.header(":material/tune: Settings")
    use_nia = st.toggle(
        "Include NIA terminology (5k terms)",
        value=False,
        disabled=not NIA_PATH.exists(),
        help="Generate it with scripts/import_nia_terminologies.py" if not NIA_PATH.exists() else None,
    )
    match_english = st.toggle("Match English equivalents", value=True)
    fuzzy = st.toggle("Fuzzy matching (misspellings)", value=False)
    min_conf = st.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.05)
    labels = st.multiselect("Only these labels", list(CORE_LABELS), placeholder="All labels")

ner = get_ner(use_nia, match_english, fuzzy)

with st.sidebar:
    st.caption(f"{len(ner.index):,} lexicon entries · version `{ner.lexicon_version}`")

st.title(":material/spa: ayurner")
st.caption("Named-entity recognition for Ayurvedic terms in Sanskrit and Hindi — Devanagari, IAST or romanized.")

tab_extract, tab_lookup, tab_batch = st.tabs(
    [":material/search: Extract", ":material/menu_book: Lookup", ":material/upload_file: Batch"]
)

# ----------------------------------------------------------------- extract
with tab_extract:
    if "text" not in st.session_state:
        st.session_state.text = EXAMPLES["Hindi"]

    def use_example():
        choice = st.session_state.example
        if choice:
            st.session_state.text = EXAMPLES[choice]

    st.pills("Examples", list(EXAMPLES), key="example", on_change=use_example)
    text = st.text_area("Text", key="text", height=140, placeholder="Type or paste Hindi / Sanskrit text…")

    doc = ner(text) if text.strip() else None
    ents = [
        e
        for e in (doc.entities if doc else ())
        if e.confidence >= min_conf and (not labels or e.label in labels)
    ]

    if doc:
        with st.container(horizontal=True):
            st.metric("Entities", len(ents), border=True)
            st.metric("Labels", len({e.label for e in ents}), border=True)
            st.metric("Ambiguous", sum(e.ambiguous for e in ents), border=True)
            avg = sum(e.confidence for e in ents) / len(ents) if ents else 0
            st.metric("Avg confidence", f"{avg:.2f}", border=True)

        with st.container(border=True):
            st.markdown(highlighted(text, ents) if ents else md_escape(text))

        if ents:
            st.dataframe(entities_frame(ents), hide_index=True, column_config=ENTITY_COLUMNS)
            with st.expander("Raw JSON"):
                st.json(doc.to_dict(), expanded=False)
        else:
            st.info("No Ayurvedic entities found.", icon=":material/info:")

# ------------------------------------------------------------------ lookup
with tab_lookup:
    q = st.text_input("Term in any script", placeholder="giloy, गुडूची, guḍūcī…")
    if q.strip():
        entries = ner.lookup(q)
        if not entries:
            st.warning("Not in the lexicon.", icon=":material/search_off:")
        for entry in entries:
            with st.container(border=True):
                color = LABEL_COLORS.get(entry.label, "gray")
                st.markdown(
                    f"### {md_escape(entry.canonical)} · {entry.devanagari}  \n"
                    f":{color}-badge[{entry.label}] `{entry.id}`"
                )
                d = entry.to_dict()
                if d.get("synonyms"):
                    st.markdown("**Synonyms:** " + md_escape(", ".join(d["synonyms"])))
                if d.get("hindi"):
                    st.markdown("**Hindi:** " + md_escape(", ".join(d["hindi"])))
                if d.get("english"):
                    st.markdown("**English:** " + md_escape(", ".join(d["english"])))
                if d.get("botanical"):
                    st.markdown(f"**Botanical:** *{md_escape(d['botanical'])}*")
                if d.get("definition"):
                    st.caption(d["definition"])

# ------------------------------------------------------------------- batch
with tab_batch:
    st.caption("Upload a .txt file with one text per line.")
    upload = st.file_uploader("Text file", type=["txt"])
    if upload is not None:
        lines = [line.strip() for line in upload.getvalue().decode("utf-8-sig").splitlines() if line.strip()]
        with st.spinner(f"Tagging {len(lines):,} lines…"):
            docs = list(ner.pipe(lines))
        rows = [
            {"line": i + 1, **row}
            for i, d in enumerate(docs)
            for row in entities_frame(d.entities).to_dict("records")
        ]
        st.metric("Entities found", len(rows))
        if rows:
            df = pd.DataFrame(rows)
            st.bar_chart(df["label"].value_counts(), horizontal=True)
            st.dataframe(df, hide_index=True, column_config=ENTITY_COLUMNS)
        jsonl = "\n".join(json.dumps(d.to_dict(), ensure_ascii=False) for d in docs)
        st.download_button(
            "Download JSONL",
            jsonl,
            file_name="ayurner_output.jsonl",
            mime="application/json",
            icon=":material/download:",
        )
