from backend.predict import DiseasePredictor
from backend.utility import load_symptom_disease_data, load_treatments_data, get_all_symptoms
from backend.chatbot import HealthChatbot

# Load datasets
symptom_df, _ = load_symptom_disease_data('data/disease_symptoms.csv')
treatments = load_treatments_data('data/disease_treatment.csv')

# Train model
predictor = DiseasePredictor('data/disease_symptoms.csv')
predictor.train()

# Get all symptoms
global_symptoms = get_all_symptoms(symptom_df)

# Create chatbot
chatbot = HealthChatbot(predictor, treatments, global_symptoms)

# Command-line loop
def main():
    print("Welcome to the Disease Prediction Chatbot! Type 'exit' to quit.")
    while True:
        user_input = input('You: ').strip()
        if user_input.lower() in ('exit', 'quit'): break
        response = chatbot.handle_message(user_input)
        print('Bot:', response)

if __name__ == '__main__':
    main()
