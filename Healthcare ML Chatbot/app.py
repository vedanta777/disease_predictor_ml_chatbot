from flask import Flask, request, jsonify
from backend.predict import DiseasePredictor
from backend.utility import load_symptom_disease_data, load_treatments_data, get_all_symptoms
from backend.chatbot import HealthChatbot

# Initialize Flask app
app = Flask(__name__)

# Load datasets
symptom_df, _ = load_symptom_disease_data('data/disease_symptoms.csv')
treatments = load_treatments_data('data/disease_treatment.csv')

# Initialize ML predictor
predictor = DiseasePredictor('data/symptoms_diseases.csv')
predictor.train()

# Get all known symptoms
all_symptoms = get_all_symptoms(symptom_df)

# Constructing the chatbot
chatbot = HealthChatbot(predictor, treatments, all_symptoms)

# Define a simple API endpoint for testing
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    reply = chatbot.handle_message(user_message)
    return jsonify({ 'reply': reply })

if __name__ == '__main__':
    app.run(debug=True)
