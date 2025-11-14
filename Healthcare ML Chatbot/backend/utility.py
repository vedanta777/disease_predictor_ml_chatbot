import pandas as pd  

# Loading the datasets
def load_symptom_disease_data(filepath):
    df = pd.read_csv(filepath)
    
    # Now, let's convert the symptoms column from a string to a list for each row.
    df['Symptoms'] = df['Symptoms'].apply(lambda x: [s.strip() for s in x.split('|')])
    
    # Building a quick lookup dictionary: disease --> list of symptoms
    disease_to_symptoms = dict(zip(df['Disease'], df['Symptoms']))
    return df, disease_to_symptoms

# Load the treatments dataset
def load_treatments_data(filepath):
    df = pd.read_csv(filepath)
    
    # Simple mapping: disease --> treatment
    disease_to_treatment = dict(zip(df['Disease'], df['Treatment']))
    return disease_to_treatment

# Getting a set of all known symptoms
def get_all_symptoms(df):
    all_symptoms = set()
    for symptoms_list in df['Symptoms']:
        all_symptoms.update(symptoms_list)
    return all_symptoms

