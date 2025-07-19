import re

# utils/synonym_map.py

synonym_map = {
    "wifi": "internet",
    "wi-fi": "internet",
    "wireless": "internet",
    "printer": "printing",
    "laptop": "computer",
    "pc": "computer",
    "desktop": "computer",
    "email": "outlook",
    "mail": "outlook",
    "login": "authentication",
    "sign in": "authentication",
    "sign-in": "authentication",
    # Add more synonyms as needed
}

def replace_synonyms(text):
    for synonym, canonical in synonym_map.items():
        if synonym in text:
            text = text.replace(synonym, canonical)
    return text
