import re

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def extract_symptoms(user_message, all_symptoms):

    # Lowercase everything for easier matching
    msg = user_message.lower()
    found = []
    
    for symptom in all_symptoms:
        pattern = r'\b' + re.escape(symptom.lower()) + r'\b'
        if re.search(pattern, msg):
            found.append(symptom)
    return found
