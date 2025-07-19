# backend/main.py

from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import numpy as np
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from utils.synonyms import normalize_text

# Initialize FastAPI
app = FastAPI(title="Triager+ API", description="Predicts helpdesk ticket category and priority.")

# Load models and encoders
category_encoder = joblib.load("models/category_encoder.pkl")
priority_encoder = joblib.load("models/priority_encoder.pkl")

vectorizer = joblib.load("models/vectorizer_v2.pkl")
category_model = joblib.load("models/category_model_v2.pkl")
priority_model = joblib.load("models/priority_model_v2.pkl")

bert_model_path = "models/bert_category"
bert_model = DistilBertForSequenceClassification.from_pretrained(bert_model_path)
tokenizer = DistilBertTokenizerFast.from_pretrained(bert_model_path)

# Input schema
class TicketRequest(BaseModel):
    ticket_text: str
    model_choice: str = "DistilBERT"  # or "Naive Bayes"

@app.post("/predict")
async def predict(request: TicketRequest):
    ticket_text = request.ticket_text
    model_choice = request.model_choice

    norm_text = normalize_text(ticket_text)

    if model_choice == "Naive Bayes":
        X_priority = vectorizer.transform([norm_text])
        category_pred = category_model.predict(X_priority)[0]
        category_conf = np.max(category_model.predict_proba(X_priority))
        priority_pred = priority_model.predict(X_priority)[0]
        priority_conf = np.max(priority_model.predict_proba(X_priority))
    else:  # DistilBERT
        inputs = tokenizer(ticket_text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = bert_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            category_idx = torch.argmax(probs, dim=1).item()
            category_pred = category_encoder.inverse_transform([category_idx])[0]
            category_conf = probs[0][category_idx].item()

        # Priority still uses Naive Bayes
        X_priority = vectorizer.transform([norm_text])
        priority_pred = priority_model.predict(X_priority)[0]
        priority_conf = np.max(priority_model.predict_proba(X_priority))

    return {
        "category": category_pred,
        "category_confidence": round(category_conf, 4),
        "priority": priority_pred,
        "priority_confidence": round(priority_conf, 4),
        "model_used": model_choice
    }