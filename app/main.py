"""FastAPI entrypoint for the TCS Financial Forecast Agent.

Exposes:
- GET /health: service readiness + document availability
- POST /forecast: runs the forecast agent and logs request/response to MySQL

The app loads documents once at startup and keeps a ForecastAgent in app_state.
"""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.config import settings
from app.db.database import get_db_session, init_db
from app.db.models import RequestLog
from app.models.schemas import ForecastRequest, ForecastResponse, HealthResponse
from app.services.document_loader import load_documents
from app.services.forecast_agent import ForecastAgent

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Simple in-memory app state shared by routes/lifespan. For this single-process
# local app, a dict is enough; larger deployments would use app.state/dependencies.
app_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize filesystem, DB, documents, and agent before serving requests."""
    Path("logs").mkdir(exist_ok=True)
    settings.data_path.mkdir(parents=True, exist_ok=True)
    settings.reports_path.mkdir(parents=True, exist_ok=True)
    settings.transcripts_path.mkdir(parents=True, exist_ok=True)

    # DB is required for request logging, but we still allow app startup so /health
    # can report the issue clearly.
    try:
        init_db()
        app_state["db_ready"] = True
    except Exception as e:
        logger.exception("DB init failed: %s", e)
        app_state["db_ready"] = False

    # Load report/transcript documents once; agent tools will use these in-memory docs.
    docs = load_documents(settings.reports_path, settings.transcripts_path)
    app_state["documents"] = docs
    app_state["agent"] = ForecastAgent(docs)
    logger.info("Loaded %d documents", len(docs))
    yield


app = FastAPI(title="TCS Financial Forecast Agent", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    """Lightweight response timing for quick latency debugging."""
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = str(int((time.perf_counter() - start) * 1000))
    return response


@app.get("/health", response_model=HealthResponse)
def health():
    docs = app_state.get("documents", [])
    return HealthResponse(
        status="ok",
        database="connected" if app_state.get("db_ready") else "unavailable",
        llm_configured=settings.llm_configured,
        documents_available={
            "reports": sum(1 for d in docs if d.doc_type == "report"),
            "transcripts": sum(1 for d in docs if d.doc_type == "transcript"),
        },
    )


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    """Running the forecast agent and persist the request/response for traceability."""
    start = time.perf_counter()
    agent: ForecastAgent = app_state.get("agent")
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    # Fail fast with actionable errors before expensive LLM/RAG work.
    if not settings.llm_configured:
        raise HTTPException(status_code=400, detail="LLM not configured. Use LLM_PROVIDER=ollama with Ollama running, or set LLM_PROVIDER=openai and OPENAI_API_KEY in .env.")

    docs = app_state.get("documents", [])
    if not docs:
        raise HTTPException(
            status_code=400,
            detail="No documents found. Place TCS quarterly report PDFs in data/reports and transcript PDFs in data/transcripts.",
        )

    try:
        result = agent.run(req.query, quarters=req.quarters)
    except Exception as e:
        logger.exception("Forecast generation failed")
        # Log failures too, so MySQL has an audit trail for debugging.
        _log_request("/forecast", req.model_dump(), {"error": str(e)}, "error", start, str(e))
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {e}")

    payload = result.model_dump(mode="json")
    _log_request("/forecast", req.model_dump(), payload, "success", start)
    return JSONResponse(content=payload)


def _log_request(endpoint: str, request_payload: dict, response_payload: dict, status: str, start: float, error: str | None = None):
    """Best-effort request logging.

    Logging should never break the user-facing API response; failures are only logged.
    """
    if not app_state.get("db_ready"):
        logger.warning("DB unavailable; skipping request log")
        return
    duration_ms = int((time.perf_counter() - start) * 1000)
    try:
        with get_db_session() as session:
            session.add(
                RequestLog(
                    endpoint=endpoint,
                    request_query=request_payload.get("query", ""),
                    request_payload=request_payload,
                    response_payload=response_payload,
                    status=status,
                    error_message=error,
                    duration_ms=duration_ms,
                )
            )
    except Exception:
        logger.exception("Failed to write request log")
