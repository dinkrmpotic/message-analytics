from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import joblib
import os
from pathlib import Path

app = FastAPI(title="Spam Detector API", version="1.0.0")

encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

model_path = os.environ.get('MODEL_PATH', str(Path(__file__).parent.parent.parent / 'models' / 'spam_classifier.joblib'))
clf = joblib.load(model_path)


class MessageRequest(BaseModel):
    message: str


class PredictionResponse(BaseModel):
    message: str
    spam_probability: float
    is_spam: bool


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: MessageRequest):
    embedding = encoder.encode([request.message])
    prob = clf.predict_proba(embedding)[0, 1]
    
    return PredictionResponse(
        message=request.message,
        spam_probability=round(prob * 100, 2),
        is_spam=prob > 0.5
    )
