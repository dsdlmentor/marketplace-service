from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request

import gradio as gr

from app.schemas.predict import BR_STATES, PredictRequest, PredictResponse

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

def _build_gradio_demo() -> gr.Blocks:
    states = sorted(BR_STATES)

    def predict_ui(price, freight, installments, n_items, state, lead_time, month):
        req = PredictRequest(
            price=price, freight=freight, installments=int(installments),
            n_items=int(n_items), customer_state=state,
            promised_lead_time_days=int(lead_time), purchase_month=int(month),
        )
        res = _score(req)
        label = "Высокий риск негатива" if res.is_high_risk else "Скорее доволен"
        return {label: res.negative_review_probability,
                "—": 1 - res.negative_review_probability}

    with gr.Blocks(title="Риск негативного отзыва") as demo:
        gr.Markdown(
            "# 📦 Риск негативного отзыва\n"
            "По параметрам заказа модель оценивает вероятность оценки 1-2 звезды."
        )
        with gr.Row():
            with gr.Column(scale=2):
                price = gr.Number(label="Сумма товаров, R$", value=120.0)
                freight = gr.Number(label="Доставка, R$", value=20.0)
                installments = gr.Number(label="Платежей по рассрочке", value=1)
                n_items = gr.Number(label="Позиций в заказе", value=1)
                state = gr.Dropdown(choices=states, label="Штат покупателя", value="SP")
                lead_time = gr.Number(label="Обещанный срок доставки, дней", value=15)
                month = gr.Number(label="Месяц заказа (1-12)", value=6)
                submit = gr.Button("Оценить риск", variant="primary")
            with gr.Column(scale=1):
                output = gr.Label(label="Вероятность негативного отзыва")
        gr.Examples(
            examples=[
                [49.9, 7.8, 1, 1, "SP", 12, 5],
                [180.0, 45.0, 6, 2, "BA", 45, 1],
                [29.9, 23.5, 1, 1, "AM", 60, 11],
            ],
            inputs=[price, freight, installments, n_items, state, lead_time, month],
            label="Примеры заказов (клик заполняет форму)",
        )
        submit.click(
            fn=predict_ui,
            inputs=[price, freight, installments, n_items, state, lead_time, month],
            outputs=output,
        )
    return demo


app = gr.mount_gradio_app(app, _build_gradio_demo(), path="/")