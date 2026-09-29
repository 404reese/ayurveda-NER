"""FastAPI service. Stateless: scale with ``--workers N`` and horizontally behind a load balancer.

Configuration (environment variables):

* ``AYURNER_LEXICONS``       extra lexicon files, os.pathsep-separated
* ``AYURNER_MAX_TEXT_CHARS`` max characters per text (default 100000)
* ``AYURNER_MAX_BATCH``      max texts per batch request (default 256)
* ``AYURNER_FUZZY``          "1" to enable fuzzy matching
* ``AYURNER_MATCH_ENGLISH``  "0" to disable English equivalents
* ``AYURNER_CORS_ORIGINS``   comma-separated allowed origins
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request

import ayurner

from ..labels import all_labels
from ..pipeline import Pipeline
from ..types import Document
from .schemas import (
    BatchExtractRequest,
    BatchExtractResponse,
    ExtractRequest,
    ExtractResponse,
    HealthResponse,
    LookupResponse,
)


def _flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


def _settings() -> dict:
    return {
        "max_text_chars": int(os.environ.get("AYURNER_MAX_TEXT_CHARS", "100000")),
        "max_batch": int(os.environ.get("AYURNER_MAX_BATCH", "256")),
    }


def create_app(pipeline: Pipeline | None = None) -> FastAPI:
    settings = _settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.ner = pipeline or ayurner.load(
            fuzzy=_flag("AYURNER_FUZZY", False),
            match_english=_flag("AYURNER_MATCH_ENGLISH", True),
        )
        yield

    app = FastAPI(
        title="ayurner",
        version=ayurner.__version__,
        description="Named-entity recognition for Ayurvedic terms in Sanskrit and Hindi.",
        lifespan=lifespan,
    )

    origins = [
        o.strip() for o in os.environ.get("AYURNER_CORS_ORIGINS", "").split(",") if o.strip()
    ]
    if origins:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    def ner(request: Request) -> Pipeline:
        return request.app.state.ner

    def check_text(text: str) -> None:
        if len(text) > settings["max_text_chars"]:
            raise HTTPException(413, f"text longer than {settings['max_text_chars']} characters")

    def filtered(doc: Document, labels, min_conf: float) -> ExtractResponse:
        wanted = {lab.upper() for lab in labels} if labels else None
        ents = [
            e.to_dict()
            for e in doc.entities
            if (wanted is None or e.label in wanted) and e.confidence >= min_conf
        ]
        return ExtractResponse(entities=ents, lexicon_version=doc.lexicon_version)

    # Sync handlers run in FastAPI's threadpool; matching is CPU-bound, so
    # throughput scales with worker processes rather than async concurrency.
    @app.post("/v1/extract", response_model=ExtractResponse)
    def extract(req: ExtractRequest, request: Request) -> ExtractResponse:
        check_text(req.text)
        return filtered(ner(request)(req.text), req.labels, req.min_confidence)

    @app.post("/v1/extract/batch", response_model=BatchExtractResponse)
    def extract_batch(req: BatchExtractRequest, request: Request) -> BatchExtractResponse:
        if len(req.texts) > settings["max_batch"]:
            raise HTTPException(413, f"batch larger than {settings['max_batch']} texts")
        for t in req.texts:
            check_text(t)
        p = ner(request)
        results = [filtered(p(t), req.labels, req.min_confidence) for t in req.texts]
        return BatchExtractResponse(results=results, lexicon_version=p.lexicon_version)

    @app.get("/v1/lookup", response_model=LookupResponse)
    def lookup(
        request: Request, q: str = Query(..., min_length=1, max_length=200)
    ) -> LookupResponse:
        return LookupResponse(query=q, entries=[e.to_dict() for e in ner(request).lookup(q)])

    @app.get("/v1/labels")
    def labels() -> dict[str, str]:
        return all_labels()

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        p = ner(request)
        return HealthResponse(status="ok", entries=len(p.index), lexicon_version=p.lexicon_version)

    @app.get("/version")
    def version(request: Request) -> dict[str, str]:
        return {"ayurner": ayurner.__version__, "lexicon_version": ner(request).lexicon_version}

    return app


def get_app() -> FastAPI:
    """Factory for ``uvicorn --factory ayurner.server.app:get_app``."""
    return create_app()
