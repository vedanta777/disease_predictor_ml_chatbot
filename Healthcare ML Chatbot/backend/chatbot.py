from backend.utility import get_all_symptoms
from backend.nlp import extract_symptoms

class HealthChatbot:
    def __init__(self, predictor, disease_to_treatment, all_symptoms):
        self.predictor = predictor
        self.disease_to_treatment = disease_to_treatment
        self.all_symptoms = all_symptoms

    def handle_message(self, user_message):
        # 1. Extract symptoms from message
        symptoms_found = extract_symptoms(user_message, self.all_symptoms)

        # 2. If no symptoms found, ask user to clarify
        if not symptoms_found:
            return "I'm sorry, I couldn't detect any valid symptoms in your message. Could you please describe how you're feeling in more detail?"

        # 3. Predict disease
        disease = self.predictor.predict(symptoms_found)

        # 4. Find treatment, if available
        treatment = self.disease_to_treatment.get(disease, None)

        # 5. Build a friendly response
        response = f"Based on your symptoms ({', '.join(symptoms_found)}), the most likely condition is: {disease}."
        if treatment:
            response += f" Standard advice: {treatment}. However, please consult a medical professional for a proper diagnosis."
        else:
            response += " Unfortunately, I don't have specific treatment advice for this condition. Please consult a doctor."

        return response
