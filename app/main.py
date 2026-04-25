from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.model_service import service


ROOT_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT_DIR / "frontend"

app = FastAPI(title="Webpage Detection Platform", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR / "static"), name="static")


class PredictionRequest(BaseModel):
    features: dict[str, float]


class SourceAnalysisRequest(BaseModel):
    url: str = ""
    text: str = ""
    html: str = ""
    overrides: dict[str, float] = {}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/summary")
def get_summary() -> dict:
    return service.get_project_summary()


@app.get("/api/schema")
def get_schema() -> dict:
    return {"fields": service.input_columns}


@app.post("/api/predict")
def predict(request: PredictionRequest) -> dict:
    result = service.predict(request.features)
    return _serialize_prediction(result)


@app.post("/api/analyze")
def analyze(request: SourceAnalysisRequest) -> dict:
    result = service.analyze_source(
        url=request.url,
        text=request.text,
        html=request.html,
        overrides=request.overrides,
    )
    return _serialize_prediction(result)


def _serialize_prediction(result) -> dict:
    return {
        "classification_label": result.classification_label,
        "classification_probability": result.classification_probability,
        "web_traffic_prediction": result.web_traffic_prediction,
        "cluster_id": result.cluster_id,
        "similar_webpages": result.similar_webpages,
        "extracted_features": result.extracted_features,
        "findings": result.findings,
        "notes": result.notes,
        "source_meta": result.source_meta,
    }
