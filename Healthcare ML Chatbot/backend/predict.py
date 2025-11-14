import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import pickle

class DiseasePredictor:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

        # Split symptoms column: from 'cough|fever' to ['cough', 'fever']
        self.df['Symptoms'] = self.df['Symptoms'].apply(
            lambda x: [s.strip() for s in x.split('|')])
        self.encoder = MultiLabelBinarizer()

        # Fit the encoder on all symptoms found in the CSV.
        self.X = self.encoder.fit_transform(self.df['Symptoms'])
        self.y = self.df['Disease']
        self.model = None

    def train(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(self.X, self.y)

    def predict(self, symptoms_list):
        input_encoded = self.encoder.transform([symptoms_list])
        disease_pred = self.model.predict(input_encoded)[0]
        return disease_pred

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump((self.model, self.encoder), f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.model, self.encoder = pickle.load(f)
