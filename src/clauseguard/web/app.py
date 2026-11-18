"""REST API and web UI (FastAPI). Run with `clauseguard serve`.

Endpoints:
  GET  /                 the web UI
  GET  /health           which document types have a model loaded or loadable (503 if none)
  GET  /document-types   the types the user can choose from
  POST /detect           multipart form: file or text -> the router's suggested document type
  POST /analyze          multipart form: document_type (a type, or "auto") + file or text
  POST /analyze-combined multipart form: document_types (comma-separated, default all) + input
  POST /feedback         JSON: does the user agree with a flag? (no document text is stored)

Browser extensions (chrome-extension:// and moz-extension:// origins) may call the API;
more origins can be allowed with CLAUSEGUARD_CORS_ORIGINS (comma-separated).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from clauseguard import __version__, pipeline
from clauseguard.config import available_modules, load_module_config
from clauseguard.errors import InputError, ModelNotAvailableError
from clauseguard.evaluation.robustness import EXAMPLES
from clauseguard.ingestion.loader import MAX_PDF_BYTES
from clauseguard.models.runs import read_json, resolve_run
from clauseguard.monitoring import log_analysis, log_error, log_feedback
from clauseguard.paths import _CODE_ROOT
from clauseguard.schema import AnalysisResult, CombinedResult

STATIC = Path(__file__).with_name("static")
EXAMPLES_DIR = _CODE_ROOT / "examples"
PUBLIC_FIELDS = ("name", "description", "ready", "model", "unit")
MODEL_NAMES = {
    "legal-bert": "LEGAL-BERT",
    "privbert": "PrivBERT",
    "distilbert": "DistilBERT",
    "tfidf": "TF-IDF baseline",
    "majority": "majority baseline",
}


class Feedback(BaseModel):
    document_type: str
    model_run: str
    label: str
    agree: bool


EXTENSION_ORIGINS = r"^(chrome-extension|moz-extension)://[a-z0-9-]+$"


def _error(status: int, code: str, message: str, details: dict | None = None) -> JSONResponse:
    body: dict = {"code": code, "message": message}
    if details:
        body.update(details)
    return JSONResponse(status_code=status, content={"error": body})


class _BadInput(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        self.status, self.code, self.message = status, code, message


def _read_input(text: str | None, file: UploadFile | None) -> tuple[str | None, bytes | None, str]:
    """(text, pdf_bytes, source) from a form with either a file or pasted text."""
    if (file is None) == (text is None or not text.strip()):
        raise _BadInput(422, "missing_input", "Send either a file or pasted text (not both).")
    if file is None:
        return text, None, "text"
    data = file.file.read(MAX_PDF_BYTES + 1)
    if len(data) > MAX_PDF_BYTES:
        raise _BadInput(
            413, "file_too_large", f"The file is larger than {MAX_PDF_BYTES // (1024 * 1024)} MB."
        )
    if data.startswith(b"%PDF") or (file.filename or "").lower().endswith(".pdf"):
        return None, data, "pdf"
    return data.decode("utf-8", errors="replace"), None, "text-file"


def _module_status() -> dict[str, dict]:
    status = {}
    for module in available_modules():
        cfg = load_module_config(module)
        try:
            run = resolve_run(module, None, cfg.selection_metric)
            meta = read_json(run / "meta.json")
            status[module] = {
                "name": cfg.display_name,
                "description": cfg.description,
                "ready": True,
                "run": run.name,
                "model": MODEL_NAMES.get(meta["model"], meta["model"]),
                "unit": cfg.unit_type,
            }
        except ModelNotAvailableError as exc:
            status[module] = {
                "name": cfg.display_name,
                "description": cfg.description,
                "ready": False,
                "reason": str(exc),
            }
    return status


def create_app() -> FastAPI:
    app = FastAPI(
        title="ClauseGuard",
        version=__version__,
        description="Flags clauses in terms of service, privacy policies and contracts that may "
        "work against the reader. Not legal advice.",
    )
    extra = [o.strip() for o in os.environ.get("CLAUSEGUARD_CORS_ORIGINS", "").split(",")]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o for o in extra if o],
        allow_origin_regex=EXTENSION_ORIGINS,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/health")
    def health() -> JSONResponse:
        modules = _module_status()
        ready = any(m["ready"] for m in modules.values())
        return JSONResponse(
            status_code=200 if ready else 503,
            content={
                "status": "ok" if ready else "unavailable",
                "version": __version__,
                "modules": modules,
                "router": pipeline.router_ready(),
                "llm_explanations": pipeline.get_explainer().enabled,
            },
        )

    @app.get("/document-types")
    def document_types() -> list[dict]:
        return [
            {
                "id": m,
                **{
                    k: v
                    for k, v in s.items()
                    if k in ("name", "description", "ready", "model", "unit")
                },
            }
            for m, s in _module_status().items()
        ]

    @app.post("/detect")
    def detect(
        text: Annotated[str | None, Form()] = None,
        file: Annotated[UploadFile | None, File()] = None,
    ):
        """Suggest a document type. ``document_type`` is null when the router is unsure."""
        try:
            text, pdf_bytes, _ = _read_input(text, file)
            doc = pipeline.load_document(text=None if pdf_bytes else text, pdf_bytes=pdf_bytes)
        except _BadInput as exc:
            return _error(exc.status, exc.code, exc.message)
        except InputError as exc:
            return _error(422, exc.code, exc.message)
        detection = pipeline.detect(doc)
        if detection is None:
            return _error(503, "router_unavailable", "Automatic type detection isn't installed.")
        return {**detection.as_dict(), "n_words": doc.n_words}

    @app.post("/analyze", response_model=AnalysisResult)
    def analyze(
        document_type: Annotated[str | None, Form()] = None,
        text: Annotated[str | None, Form()] = None,
        file: Annotated[UploadFile | None, File()] = None,
    ):
        """Analyse a document. ``document_type``: tos, privacy, contract, or auto (the router
        chooses, and answers 422 when it is unsure)."""
        started = time.perf_counter()
        source = "file" if file is not None else "text"

        def fail(status: int, code: str, message: str, details: dict | None = None):
            log_error(
                document_type,
                code,
                source=source,
                latency_ms=1000 * (time.perf_counter() - started),
            )
            return _error(status, code, message, details)

        try:
            text, pdf_bytes, source = _read_input(text, file)
            result = pipeline.analyze(
                document_type or pipeline.AUTO,
                text=None if pdf_bytes else text,
                pdf_bytes=pdf_bytes,
            )
        except _BadInput as exc:
            return fail(exc.status, exc.code, exc.message)
        except InputError as exc:
            return fail(422, exc.code, exc.message, exc.details)
        except ModelNotAvailableError as exc:
            return fail(503, "model_unavailable", str(exc))
        log_analysis(result, source=source, latency_ms=1000 * (time.perf_counter() - started))
        return result

    @app.post("/analyze-combined", response_model=CombinedResult)
    def analyze_combined(
        document_types: Annotated[str | None, Form()] = None,
        text: Annotated[str | None, Form()] = None,
        file: Annotated[UploadFile | None, File()] = None,
    ):
        """Run several analyzers on one document (comma-separated types; default: all)."""
        started = time.perf_counter()
        types = [t.strip() for t in (document_types or "").split(",") if t.strip()] or None
        try:
            text, pdf_bytes, source = _read_input(text, file)
            combined = pipeline.analyze_combined(
                types, text=None if pdf_bytes else text, pdf_bytes=pdf_bytes
            )
        except _BadInput as exc:
            return _error(exc.status, exc.code, exc.message)
        except InputError as exc:
            return _error(422, exc.code, exc.message, exc.details)
        except ModelNotAvailableError as exc:
            return _error(503, "model_unavailable", str(exc))
        latency = 1000 * (time.perf_counter() - started)
        for result in combined.results:
            log_analysis(result, source=source, latency_ms=latency / len(combined.results))
        return combined

    @app.post("/feedback", status_code=204)
    def feedback(item: Feedback) -> None:
        log_feedback(item.document_type, item.model_run, item.label, item.agree)

    @app.get("/examples/{document_type}", include_in_schema=False)
    def example(document_type: str):
        path = EXAMPLES_DIR / EXAMPLES.get(document_type, "missing")
        if not path.is_file():
            return _error(404, "no_example", "No example for this document type.")
        return PlainTextResponse(path.read_text(encoding="utf-8"))

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


app = create_app()
