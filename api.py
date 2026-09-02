"""
FastAPI wrapper around the ticket categorizer.

Serves the trained scikit-learn routing pipeline over HTTP and hosts the
routing-desk UI in static/. All routing logic still lives in
ticket_classifier.py - this module only loads the model, validates input, and
turns a routing result into JSON.

Run:
    uvicorn api:app --reload
Then open http://127.0.0.1:8000
"""

from contextlib import asynccontextmanager
from itertools import count
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import ticket_classifier as tc

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

# Slip numbers continue the series the offline demo prints (TCK-3001...).
_slip_numbers = count(3001)


def load_or_train():
    """Load the saved pipeline; train once and save it if it isn't there yet."""
    if Path(tc.MODEL_PATH).exists():
        return joblib.load(tc.MODEL_PATH)
    print("No saved model found - training one now...")
    return tc.train_and_evaluate()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = load_or_train()
    app.state.model_name = type(app.state.pipeline.named_steps["clf"]).__name__
    app.state.categories = sorted(str(c) for c in app.state.pipeline.classes_)
    yield


app = FastAPI(
    title="Auto Ticket Categorizer",
    description="Routes a support ticket to BILLING, TECHNICAL, HR or GENERAL.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class TicketIn(BaseModel):
    ticket_text: str = Field(
        min_length=1,
        max_length=5000,
        description="Raw ticket text (subject + body).",
        examples=["My card was charged twice for the same order, please refund"],
    )


class BatchIn(BaseModel):
    tickets: list[str] = Field(min_length=1, max_length=100)


class RunnerUp(BaseModel):
    department: str
    confidence: float


class Routing(BaseModel):
    id: str
    ticket_text: str
    department: str
    model_prediction: str
    confidence: float
    needs_review: bool
    review_reason: str | None
    runner_up: RunnerUp
    priority: str
    distribution: dict[str, float]


class Health(BaseModel):
    status: str
    model: str
    categories: list[str]
    confidence_threshold: float
    ambiguity_threshold: float


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api/health", response_model=Health, summary="Model status and thresholds")
def health() -> Health:
    return Health(
        status="ready",
        model=app.state.model_name,
        categories=app.state.categories,
        confidence_threshold=tc.CONFIDENCE_THRESHOLD,
        ambiguity_threshold=tc.AMBIGUITY_THRESHOLD,
    )


@app.get("/api/samples", response_model=list[str], summary="Unseen example tickets")
def samples() -> list[str]:
    return tc.SAMPLE_TICKETS


@app.post("/api/classify", response_model=Routing, summary="Route one ticket")
def classify(payload: TicketIn) -> Routing:
    text = payload.ticket_text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Ticket text is empty.")
    return _route(text)


@app.post("/api/classify/batch", response_model=list[Routing], summary="Route many tickets")
def classify_batch(payload: BatchIn) -> list[Routing]:
    texts = [t.strip() for t in payload.tickets if t.strip()]
    if not texts:
        raise HTTPException(status_code=422, detail="No non-empty tickets supplied.")
    return [_route(t) for t in texts]


def _route(text: str) -> Routing:
    result = tc.route_ticket(app.state.pipeline, text)
    return Routing(id=f"TCK-{next(_slip_numbers)}", ticket_text=text, **result)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
