"""Request / response models for the REST API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ExtractRequest(BaseModel):
    text: str = Field(..., description="Text in Devanagari, IAST or romanized Hindi/Sanskrit")
    labels: list[str] | None = Field(None, description="Only return these labels")
    min_confidence: float = Field(0.0, ge=0.0, le=1.0)


class BatchExtractRequest(BaseModel):
    texts: list[str]
    labels: list[str] | None = None
    min_confidence: float = Field(0.0, ge=0.0, le=1.0)


class ComponentOut(BaseModel):
    entity_id: str
    label: str
    subtype: str
    canonical: str


class EntityOut(BaseModel):
    text: str
    start: int
    end: int
    label: str
    subtype: str
    entity_id: str
    canonical: str
    devanagari: str
    confidence: float
    match_type: str
    ambiguous: bool
    alternatives: list[str]
    components: list[ComponentOut]
    metadata: dict[str, Any]


class ExtractResponse(BaseModel):
    entities: list[EntityOut]
    lexicon_version: str


class BatchExtractResponse(BaseModel):
    results: list[ExtractResponse]
    lexicon_version: str


class LookupResponse(BaseModel):
    query: str
    entries: list[dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    entries: int
    lexicon_version: str
