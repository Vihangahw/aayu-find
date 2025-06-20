from pydantic import BaseModel

class PredictionResponse(BaseModel):
    resnet_class: str
    resnet_confidence: float
    effnet_class: str
    effnet_confidence: float
    cnn_class: str
    cnn_confidence: float
    final_class: str
    final_confidence: float