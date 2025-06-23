
from pydantic import BaseModel

class PredictionResponse(BaseModel):
    final_class: str
    final_confidence: float

    class Config:
        schema_extra = {
            "example": {
                "final_class": "YakiNaran",
                "final_confidence": 0.905681848526001
            }
        }