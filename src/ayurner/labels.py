"""Entity label registry. Labels are plain strings; users may register their own."""

from __future__ import annotations

CORE_LABELS: dict[str, str] = {
    "DRAVYA": "Medicinal substance: herb, mineral or animal product (subtype herb/mineral/animal)",
    "FORMULATION": "Compound preparation or classical drug group (yoga, gana)",
    "DOSAGE_FORM": "Pharmaceutical form (kalpana): churna, kvatha, vati, ghrita, taila, asava, bhasma...",
    "ROGA": "Disease or disorder",
    "SYMPTOM": "Symptom or clinical feature (lakshana)",
    "DOSHA": "Dosha and sub-dosha (vata, pitta, kapha...)",
    "DHATU": "Body tissue (dhatu), upadhatu, ojas",
    "MALA": "Waste product (mala): purisha, mutra, sveda",
    "PROPERTY": "Pharmacological property: rasa, guna, virya, vipaka, prabhava",
    "KARMA": "Pharmacological action (deepana, pachana, rasayana...)",
    "PROCEDURE": "Therapeutic procedure (panchakarma, abhyanga, shirodhara...)",
    "ANATOMY": "Anatomical structure: srotas, marma, organs",
    "CONCEPT": "Other Ayurvedic concept (prakriti, agni, pathya, dinacharya...)",
}

_registry: dict[str, str] = dict(CORE_LABELS)


def register_label(name: str, description: str = "") -> str:
    """Register a custom label (e.g. ``"YOGA_ASANA"``). Returns the normalized name."""
    key = name.strip().upper()
    if not key:
        raise ValueError("label name must not be empty")
    _registry.setdefault(key, description)
    return key


def is_known(label: str) -> bool:
    return label.upper() in _registry


def all_labels() -> dict[str, str]:
    return dict(_registry)
