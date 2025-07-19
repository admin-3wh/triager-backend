import re
import string

# Synonym map to unify terminology
synonym_map = {
    "wifi": "internet",
    "wi-fi": "internet",
    "wireless": "internet",
    "connection": "internet",
    "printer": "printing",
    "laptop": "computer",
    "pc": "computer",
    "desktop": "computer",
    "email": "outlook",
    "mail": "outlook",
    "sign in": "authentication",
    "sign-in": "authentication",
    "login": "authentication",
    "log in": "authentication",
    "password": "authentication",
    "access": "authentication",
    "blue screen": "crash",
    "freeze": "crash",
    "not responding": "crash",
    "slow": "performance",
    "lag": "performance",
    "sluggish": "performance",
    "account": "user",
    "profile": "user",
    "folder": "file",
    "document": "file",
    "spreadsheet": "file",
    "presentation": "file",
    "teams": "communication",
    "zoom": "communication",
    "meeting": "communication",
    "camera": "hardware",
    "microphone": "hardware",
    "speaker": "hardware",
    # Add more synonyms here as needed
}

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()

    # Apply synonym replacement
    for synonym, canonical in synonym_map.items():
        text = re.sub(rf"\b{synonym}\b", canonical, text)

    return text
