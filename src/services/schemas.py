from pydantic import BaseModel, Field

class CustomerFeaturesInput(BaseModel):
    balance_to_max_ratio: float = Field(..., description="Ratio of current balance to historical peak balance", ge=0.0, le=1.5)
    tx_velocity_drop_ratio: float = Field(..., description="Proxy drop ratio of transaction frequency windows", ge=0.0, le=1.0)
    failed_tx_count_7d: int = Field(..., description="Count of failed transaction events over the last 7 days", ge=0)
    support_tickets_30d: int = Field(..., description="Count of client support disputes filed in 30 days", ge=0)
    is_active_credit_card_user: int = Field(..., description="Binary flag indicating active credit card usage (0 or 1)", ge=0, le=1)

    class Config:
        json_schema_extra = {
            "example": {
                "balance_to_max_ratio": 0.45,
                "tx_velocity_drop_ratio": 0.82,
                "failed_tx_count_7d": 3,
                "support_tickets_30d": 1,
                "is_active_credit_card_user": 1
            }
        }