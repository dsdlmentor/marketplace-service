from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request

from app.schemas.predict import PredictRequest, PredictResponse

MODEL_PATH = Path("models/model.pkl")

# bundle собран в обучающем ноутбуке: модель + порог + порядок признаков + категории штатов
_bundle = joblib.load(MODEL_PATH)
_MODEL = _bundle["model"]
_THRESHOLD = float(_bundle["threshold"])
_FEATURES = _bundle["features"]
_STATE_CATEGORIES = _bundle["state_categories"]


def _featurize(req: PredictRequest) -> pd.DataFrame:
    row = {
        "total_price": req.price,
        "total_freight": req.freight,
        "freight_ratio": req.freight / max(req.price, 0.01),
        "installments": req.installments,
        "n_items": req.n_items,
        "promised_lead_time": req.promised_lead_time_days,
        "purchase_month": req.purchase_month,
        "customer_state": req.customer_state,
    }
    df = pd.DataFrame([row])[_FEATURES]
    # категории штата переиспользуем из обучения, иначе коды LightGBM разъедутся
    df["customer_state"] = pd.Categorical(df["customer_state"], categories=_STATE_CATEGORIES)
    return df


def _score(req: PredictRequest) -> PredictResponse:
    proba = float(_MODEL.predict_proba(_featurize(req))[0, 1])
    return PredictResponse(
        negative_review_probability=round(proba, 4),
        is_high_risk=proba >= _THRESHOLD,
        threshold=round(_THRESHOLD, 4),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = _MODEL
    yield


app = FastAPI(title="Marketplace satisfaction service", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request) -> PredictResponse:
    return _score(req)