"""Recognizers, rules and overlap resolution."""

from .base import Candidate, MatchContext, Recognizer, Sense
from .gazetteer import GazetteerRecognizer
from .resolve import resolve
from .rules import RuleEngine

__all__ = [
    "Candidate",
    "GazetteerRecognizer",
    "MatchContext",
    "Recognizer",
    "RuleEngine",
    "Sense",
    "resolve",
]
