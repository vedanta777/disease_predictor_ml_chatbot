This is a basic walkthrough of the tech kit and know-how of what has been used in this project.

Coding Language – Python  
(Used for reading datasets, encoding symptoms, training the Random Forest model, and running the chatbot backend logic.)

Database / Data Source – CSV files  
(Symptom–disease mappings and treatment/precaution info are stored in CSV files instead of a traditional database for simplicity and portability.)

Libraries Required  
1. pandas – Reads and manipulates CSV files, filters records, and prepares clean tabular data for ML.
2. numpy – Handles numerical arrays and low-level vector operations for the feature matrix given to the model.
3. scikit-learn (sklearn) – Provides `RandomForestClassifier` for disease prediction and `MultiLabelBinarizer` (or similar encoders) for converting symptom lists into model-friendly vectors.
4. Flask (or any minimal web framework used) – Exposes the ML model as HTTP routes so the chatbot or UI can send symptoms and receive predictions.
5. pickle / joblib – Saves trained models and encoders to disk and reloads them without retraining every time.

How the code works  
1. When the app runs, it loads the symptom–disease dataset, converts symptom lists into binary vectors, and either trains or loads a pre-trained Random Forest classifier.
2. For each user interaction, the chatbot or API takes the user’s symptom input (often free text or selected options), parses it into a symptom list, encodes it using the same encoder, and sends it to the model for prediction.
3. The model predicts the most likely disease, and the backend then fetches the corresponding description, precautions, and basic treatment guidance from the CSV data to build a response.
4. A lightweight Flask server (or equivalent) listens for requests (from a web form or chat UI), passes symptoms to the ML pipeline, and returns JSON or rendered HTML with the prediction and advice.

What the code creates and does
1.Creates an end-to-end pipeline that:  
  - Loads curated healthcare datasets.
  - Trains and serves a multiclass disease prediction model using Random Forest.
  - Provides a chatbot-style or form-based interface for users to enter symptoms and receive instant predicted diseases plus simple precautions.
2. Can be extended to support more diseases, richer descriptions, authentication, and additional safety checks before showing any recommendation.


