import csv
from datetime import datetime
import os

LOG_FILE = "tracking/user_logs.csv"

def log_user_ticket(name, email, ticket_text, predicted_category, predicted_priority, model_used, category_confidence=None, priority_confidence=None):
    """
    Logs the user-submitted ticket with optional user info, predicted labels, model used,
    and optional confidence scores.
    """
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.now().isoformat(),
            name or "N/A",
            email or "N/A",
            ticket_text,
            predicted_category,
            predicted_priority,
            model_used,
            f"{category_confidence:.4f}" if category_confidence is not None else "N/A",
            f"{priority_confidence:.4f}" if priority_confidence is not None else "N/A",
            "N/A"  # Placeholder for corrected category
        ])

def log_wrong_prediction(ticket_text, corrected_category):
    """
    Logs a correction made by the user for retraining later.
    """
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.now().isoformat(),
            "N/A",         # Name
            "N/A",         # Email
            ticket_text,
            "N/A",         # Predicted category
            "N/A",         # Predicted priority
            "N/A",         # Model used
            "N/A",         # Category confidence
            "N/A",         # Priority confidence
            corrected_category
        ])
