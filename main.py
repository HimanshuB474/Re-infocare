from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sklearn.preprocessing import MultiLabelBinarizer
import joblib
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import uvicorn

# Load the trained PPO model and encoder
model = PPO.load(r'C:\Users\himan\OneDrive\Desktop\Database ML\healthcare_recommendation_model.zip')
mlb = joblib.load(r'C:\Users\himan\OneDrive\Desktop\Database ML\mlb.joblib')

# Load necessary data files
diets = pd.read_csv(r'C:\Users\himan\OneDrive\Desktop\Database ML\diets.csv')
medications = pd.read_csv(r'C:\Users\himan\OneDrive\Desktop\Database ML\medications.csv')
workouts = pd.read_csv(r'C:\Users\himan\OneDrive\Desktop\Database ML\workout_df.csv')

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/recommend", response_class=HTMLResponse)
async def recommend(request: Request, symptoms: str = Form(...)):
    try:
        # Preprocess user symptoms
        print("Received Symptoms:", symptoms)
        user_symptoms = [symptom.strip() for symptom in symptoms.split(",")]
        print("Processed Symptoms:", user_symptoms)

        # Ensure all symptoms are mapped to the classes
        valid_symptoms = [symptom for symptom in user_symptoms if symptom in mlb.classes_]
        unknown_symptoms = [symptom for symptom in user_symptoms if symptom not in mlb.classes_]

        if unknown_symptoms:
            print("Unknown Symptoms:", unknown_symptoms)

        # Check if we have valid symptoms for encoding
        if valid_symptoms:
            user_symptoms_encoded = pd.DataFrame(mlb.transform([valid_symptoms]), columns=mlb.classes_)
            print("Encoded Symptoms:\n", user_symptoms_encoded)
        else:
            print("No valid symptoms found in the MLB classes. Returning empty encoding.")
            user_symptoms_encoded = pd.DataFrame(np.zeros((1, len(mlb.classes_))), columns=mlb.classes_)
        
        # Check if the encoded data has the correct number of features
        if user_symptoms_encoded.shape[1] != 86:
            print(f"Warning: Adjusting encoded data to match 86 features. Current shape: {user_symptoms_encoded.shape[1]}")
            user_symptoms_encoded = user_symptoms_encoded.iloc[:, :86]  # Trim to the first 86 features
            print("Adjusted Encoded Symptoms:\n", user_symptoms_encoded)

        # Log the shape and values of the environment state
        env_state = user_symptoms_encoded.values.flatten()
        print("Environment State Shape:", env_state.shape)
        print("Environment State:", env_state)

        # Predict recommendation using the PPO model
        action, _ = model.predict(env_state)
        print("Predicted Action:", action)

        # Perform the same checks and recommendation as before
        total_diets = len(diets)
        total_medications = len(medications)
        total_workouts = len(workouts)

        if action < total_diets:
            recommendation_type = "Diet"
            recommendation = diets.iloc[action]['Diet']
        elif action < total_diets + total_medications:
            recommendation_type = "Medication"
            recommendation = medications.iloc[action - total_diets]['Medication']
        else:
            recommendation_type = "Workout"
            recommendation = workouts.iloc[action - total_diets - total_medications]['Workout']

        print(f"Recommendation Type: {recommendation_type}, Recommendation: {recommendation}")

        # Show a message about unknown symptoms
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "recommendation_type": recommendation_type,
                "recommendation": recommendation,
                "unknown_symptoms": unknown_symptoms
            }
        )

    except Exception as e:
        print(f"Recommendation Error: {e}")
        return f"Internal Server Error: {str(e)}"


# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
