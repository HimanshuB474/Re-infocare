# Import necessary libraries
import streamlit as st
import pandas as pd
from stable_baselines3 import PPO
import numpy as np

# Load necessary data files
diets = pd.read_csv(r'C:\Users\himan\OneDrive\Desktop\Database ML\diets.csv')
medications = pd.read_csv(r'C:\Users\himan\OneDrive\Desktop\Database ML\medications.csv')
workouts = pd.read_csv(r'C:\Users\himan\OneDrive\Desktop\Database ML\workout_df.csv')
symptoms_df = pd.read_csv(r'C:\Users\himan\OneDrive\Desktop\Database ML\symtoms_df.csv')

# Load the saved model
model = PPO.load("healthcare_recommendation_model")

# Preprocess the symptoms for MultiLabelBinarizer encoding
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
symptoms_df['symptoms'] = symptoms_df[['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4']].apply(
    lambda x: [s.strip() for s in x.tolist() if isinstance(s, str)], axis=1)
mlb.fit(symptoms_df['symptoms'])

# Streamlit app code
st.title("Healthcare Recommendation System")
st.write("Please enter your symptoms to receive a personalized health recommendation.")

# Input field for user symptoms
user_input = st.text_input("Enter symptoms (comma-separated):")
if user_input:
    user_symptoms = [symptom.strip() for symptom in user_input.split(",")]

    # Transform symptoms to match model input format
    user_symptoms_encoded = pd.DataFrame(mlb.transform([user_symptoms]), columns=mlb.classes_)

    # Load environment and set user state
    env_state = user_symptoms_encoded.values.flatten()
    action, _ = model.predict(env_state)

    # Determine recommendation type
    if action < len(diets):
        recommendation_type = "Diet"
        recommendation = diets.iloc[action]['Diet']  # Adjust column name as per dataset
    elif action < len(diets) + len(medications):
        recommendation_type = "Medication"
        recommendation = medications.iloc[action - len(diets)]['Medication']  # Adjust column name
    else:
        recommendation_type = "Workout"
        recommendation = workouts.iloc[action - len(diets) - len(medications)]['Workout']  # Adjust column name

    # Display recommendation
    st.write(f"### Recommended {recommendation_type}:")
    st.write(recommendation)
