from pydantic import BaseModel, Field, field_validator

# 27 штатов Бразилии — ровно те категории customer_state, что видела модель на обучении
BR_STATES = frozenset({
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS", "MT",
    "PA", "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO",
})


class PredictRequest(BaseModel):
    """Один заказ маркетплейса в момент оформления."""

    price: float = Field(..., ge=0, description="Сумма товаров в заказе, R$")
    freight: float = Field(..., ge=0, description="Стоимость доставки, R$")
    installments: int = Field(..., ge=1, le=24, description="Число платежей по рассрочке")
    n_items: int = Field(..., ge=1, description="Количество позиций в заказе")
    customer_state: str = Field(..., description="Штат покупателя, 2 буквы")
    promised_lead_time_days: int = Field(..., ge=0, le=180, description="Дней до обещанной доставки")
    purchase_month: int = Field(..., ge=1, le=12, description="Месяц заказа, 1-12")

    @field_validator("customer_state")
    @classmethod
    def state_must_be_known(cls, value: str) -> str:
        normalized = value.strip().upper()
        if normalized not in BR_STATES:
            raise ValueError(f"unknown customer_state {value!r}")
        return normalized


class PredictResponse(BaseModel):
    """Оценка риска негативного отзыва по заказу."""

    negative_review_probability: float = Field(..., ge=0, le=1)
    is_high_risk: bool
    threshold: float