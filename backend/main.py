# backend/main.py

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
# import torch
# from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from utils.synonyms import normalize_text
from fastapi.responses import JSONResponse  # ⭐️ added for graceful error

# Initialize FastAPI
app = FastAPI(title="Triager+ API", description="Predicts helpdesk ticket category and priority.")

# ⭐️ Configure CORS middleware
origins = ["https://3wh.dev"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,  # Important: must be False if origins = ["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models and encoders
category_encoder = joblib.load("models/category_encoder.pkl")
priority_encoder = joblib.load("models/priority_encoder.pkl")

vectorizer = joblib.load("models/vectorizer_v2.pkl")
category_model = joblib.load("models/category_model_v2.pkl")
priority_model = joblib.load("models/priority_model_v2.pkl")

# Commented out DistilBERT-related loading for memory saving
# bert_model_path = "models/bert_category"
# bert_model = DistilBertForSequenceClassification.from_pretrained(bert_model_path)
# tokenizer = DistilBertTokenizerFast.from_pretrained(bert_model_path)

# Input schema
class TicketRequest(BaseModel):
    ticket_text: str
    model_choice: str = "DistilBERT"  # or "Naive Bayes"

@app.post("/predict")
async def predict(request: TicketRequest):
    ticket_text = request.ticket_text
    model_choice = request.model_choice

    print(f"Received request — model_choice: {model_choice}")
    print(f"Ticket text: {ticket_text[:50]}...")

    norm_text = normalize_text(ticket_text)

    if model_choice == "Naive Bayes":
        print("Using Naive Bayes")
        try:
            X_priority = vectorizer.transform([norm_text])
            category_pred = category_model.predict(X_priority)[0]
            category_conf = np.max(category_model.predict_proba(X_priority))
            priority_pred = priority_model.predict(X_priority)[0]
            priority_conf = np.max(priority_model.predict_proba(X_priority))
        except Exception as e:
            print("Naive Bayes prediction failed:", e)
            raise
    else:
        print("Using DistilBERT — Not available")
        return JSONResponse(
            status_code=503,
            content={
                "error": "DistilBERT model is currently disabled due to memory limits.",
                "category": None,
                "category_confidence": None,
                "priority": None,
                "priority_confidence": None,
                "model_used": "DistilBERT (unavailable)"
            }
        )

    return {
        "category": category_pred,
        "category_confidence": round(category_conf, 4),
        "priority": priority_pred,
        "priority_confidence": round(priority_conf, 4),
        "model_used": model_choice
    }