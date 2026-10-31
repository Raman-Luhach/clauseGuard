"""REST API and web UI (FastAPI). Run with `clauseguard serve`.

Endpoints:
  GET  /                 the web UI
  GET  /health           which document types have a model loaded or loadable (503 if none)
  GET  /document-types   the types the user can choose from
  POST /analyze          multipart form: document_type (required) + file (PDF/TXT) or text
  POST /feedback         JSON: does the user agree with a flag? (no document text is stored)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from clauseguard import __version__, pipeline
from clauseguard.config import available_modules, load_module_config
from clauseguard.errors import InputError, ModelNotAvailableError
from clauseguard.ingestion.loader import MAX_PDF_BYTES
from clauseguard.models.runs import resolve_run
from clauseguard.monitoring import log_analysis, log_error, log_feedback
from clauseguard.schema import AnalysisResult

STATIC = Path(__file__).with_name("static")


class Feedback(BaseModel):
    document_type: str
    model_run: str
    label: str
    agree: bool


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def _module_status() -> dict[str, dict]:
    status = {}
    for module in available_modules():
        cfg = load_module_config(module)
        try:
            run = resolve_run(module, None, cfg.selection_metric)
            status[module] = {
                "name": cfg.display_name,
                "description": cfg.description,
                "ready": True,
                "run": run.name,
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
                "llm_explanations": pipeline.get_explainer().enabled,
            },
        )

    @app.get("/document-types")
    def document_types() -> list[dict]:
        return [
            {"id": m, **{k: v for k, v in s.items() if k in ("name", "description", "ready")}}
            for m, s in _module_status().items()
        ]

    @app.post("/analyze", response_model=AnalysisResult)
    def analyze(
        document_type: Annotated[str | None, Form()] = None,
        text: Annotated[str | None, Form()] = None,
        file: Annotated[UploadFile | None, File()] = None,
    ):
        started = time.perf_counter()
        source = "file" if file is not None else "text"

        def fail(status: int, code: str, message: str) -> JSONResponse:
            log_error(
                document_type,
                code,
                source=source,
                latency_ms=1000 * (time.perf_counter() - started),
            )
            return _error(status, code, message)

        if (file is None) == (text is None or not text.strip()):
            return fail(422, "missing_input", "Send either a file or pasted text (not both).")
        pdf_bytes = None
        if file is not None:
            data = file.file.read(MAX_PDF_BYTES + 1)
            if len(data) > MAX_PDF_BYTES:
                return fail(
                    413,
                    "file_too_large",
                    f"The file is larger than {MAX_PDF_BYTES // (1024 * 1024)} MB.",
                )
            if data.startswith(b"%PDF") or (file.filename or "").lower().endswith(".pdf"):
                pdf_bytes, source = data, "pdf"
            else:
                text, source = data.decode("utf-8", errors="replace"), "text-file"
        try:
            result = pipeline.analyze(
                document_type, text=None if pdf_bytes else text, pdf_bytes=pdf_bytes
            )
        except InputError as exc:
            return fail(422, exc.code, exc.message)
        except ModelNotAvailableError as exc:
            return fail(503, "model_unavailable", str(exc))
        log_analysis(result, source=source, latency_ms=1000 * (time.perf_counter() - started))
        return result

    @app.post("/feedback", status_code=204)
    def feedback(item: Feedback) -> None:
        log_feedback(item.document_type, item.model_run, item.label, item.agree)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


app = create_app()
