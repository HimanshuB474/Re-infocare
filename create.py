from sklearn.preprocessing import MultiLabelBinarizer
import joblib

# List of symptoms
symptoms = [
    "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing",
    "shivering", "chills", "joint_pain", "stomach_pain", "acidity", 
    "ulcers_on_tongue", "muscle_wasting", "vomiting", "burning_micturition",
    "spotting_urination", "fatigue", "weight_gain", "anxiety", "cold_hands_and_feets",
    "mood_swings", "weight_loss", "restlessness", "lethargy", "patches_in_throat",
    "irregular_sugar_level", "cough", "high_fever", "sunken_eyes", "breathlessness",
    "sweating", "dehydration", "indigestion", "headache", "yellowish_skin",
    "dark_urine", "nausea", "loss_of_appetite", "pain_behind_the_eyes", "back_pain",
    "constipation", "abdominal_pain", "diarrhoea", "mild_fever", "yellow_urine",
    "yellowing_of_eyes", "acute_liver_failure", "fluid_overload", "swelling_of_stomach",
    "swelled_lymph_nodes", "malaise", "blurred_and_distorted_vision", "phlegm",
    "throat_irritation", "redness_of_eyes", "sinus_pressure", "runny_nose",
    "congestion", "chest_pain", "weakness_in_limbs", "fast_heart_rate",
    "pain_during_bowel_movements", "pain_in_anal_region", "bloody_stool",
    "irritation_in_anus", "neck_pain", "dizziness", "cramps", "bruising", "obesity",
    "swollen_legs", "swollen_blood_vessels", "puffy_face_and_eyes", "enlarged_thyroid",
    "brittle_nails", "swollen_extremeties", "excessive_hunger", "extra_marital_contacts",
    "drying_and_tingling_lips", "slurred_speech", "knee_pain", "hip_joint_pain",
    "muscle_weakness", "stiff_neck", "swelling_joints", "movement_stiffness",
    "spinning_movements", "loss_of_balance", "unsteadiness", "weakness_of_one_body_side",
    "loss_of_smell", "bladder_discomfort", "foul_smell_ofurine", "continuous_feel_of_urine",
    "passage_of_gases", "internal_itching", "toxic_look_(typhos)", "depression",
    "irritability", "muscle_pain", "altered_sensorium", "red_spots_over_body",
    "belly_pain", "abnormal_menstruation", "dischromic_patches", "watering_from_eyes",
    "increased_appetite", "polyuria", "family_history", "mucoid_sputum", "rusty_sputum",
    "lack_of_concentration", "visual_disturbances", "receiving_blood_transfusion",
    "receiving_unsterile_injections", "coma", "stomach_bleeding", "distention_of_abdomen",
    "history_of_alcohol_consumption", "fluid_overload", "blood_in_sputum",
    "prominent_veins_on_calf", "palpitations", "painful_walking", "pus_filled_pimples",
    "blackheads", "scurring", "skin_peeling", "silver_like_dusting", "small_dents_in_nails",
    "inflammatory_nails", "blister", "red_sore_around_nose", "yellow_crust_ooze", "prognosis"
]

# Initialize the MultiLabelBinarizer
mlb = MultiLabelBinarizer()

# Fit the binarizer with the list of symptoms
mlb.fit([symptoms])

# Save the encoder
joblib.dump(mlb, r'C:\Users\himan\OneDrive\Desktop\Database ML\mlb.joblib')
import numpy as np

# List of all symptoms (classes) the model was trained on
classes = ['abdominal_pain', 'abnormal_menstruation', 'acidity', 'acute_liver_failure', 
           'altered_sensorium', 'anxiety', 'back_pain', 'belly_pain', 'blackheads',
           'bladder_discomfort', 'blister', 'blood_in_sputum', 'bloody_stool', 
           'blurred_and_distorted_vision', 'breathlessness', 'brittle_nails', 
           'bruising', 'burning_micturition', 'chest_pain', 'chills', 
           'cold_hands_and_feets', 'coma', 'congestion', 'constipation',
           'continuous_feel_of_urine', 'continuous_sneezing', 'cough', 'cramps', 
           'dark_urine', 'dehydration', 'depression', 'diarrhoea', 'dischromic_patches', 
           'distention_of_abdomen', 'dizziness', 'drying_and_tingling_lips', 
           'enlarged_thyroid', 'excessive_hunger', 'extra_marital_contacts', 
           'family_history', 'fast_heart_rate', 'fatigue', 'fluid_overload', 
           'foul_smell_ofurine', 'headache', 'high_fever', 'hip_joint_pain', 
           'history_of_alcohol_consumption', 'increased_appetite', 'indigestion', 
           'inflammatory_nails', 'internal_itching', 'irregular_sugar_level', 
           'irritability', 'irritation_in_anus', 'itching', 'joint_pain', 'knee_pain', 
           'lack_of_concentration', 'lethargy', 'loss_of_appetite', 'loss_of_balance', 
           'loss_of_smell', 'malaise', 'mild_fever', 'mood_swings', 'movement_stiffness', 
           'mucoid_sputum', 'muscle_pain', 'muscle_wasting', 'muscle_weakness', 'nausea', 
           'neck_pain', 'nodal_skin_eruptions', 'obesity', 'pain_behind_the_eyes', 
           'pain_during_bowel_movements', 'pain_in_anal_region', 'painful_walking', 
           'palpitations', 'passage_of_gases', 'patches_in_throat', 'phlegm', 'polyuria', 
           'prognosis', 'prominent_veins_on_calf', 'puffy_face_and_eyes', 
           'pus_filled_pimples', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 
           'red_sore_around_nose', 'red_spots_over_body', 'redness_of_eyes', 'restlessness', 
           'runny_nose', 'rusty_sputum', 'scurring', 'shivering', 'silver_like_dusting', 
           'sinus_pressure', 'skin_peeling', 'skin_rash', 'slurred_speech', 'small_dents_in_nails', 
           'spinning_movements', 'spotting_urination', 'stiff_neck', 'stomach_bleeding', 
           'stomach_pain', 'sunken_eyes', 'sweating', 'swelled_lymph_nodes', 'swelling_joints', 
           'swelling_of_stomach', 'swollen_blood_vessels', 'swollen_extremeties', 'swollen_legs', 
           'throat_irritation', 'toxic_look_(typhos)', 'ulcers_on_tongue', 'unsteadiness', 
           'visual_disturbances', 'vomiting', 'watering_from_eyes', 'weakness_in_limbs', 
           'weakness_of_one_body_side', 'weight_gain', 'weight_loss', 'yellow_crust_ooze', 
           'yellow_urine', 'yellowing_of_eyes', 'yellowish_skin']

# Save it as a numpy file
np.save('mlb_classes.npy', np.array(classes))
