import pandas as pd
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
from sklearn.preprocessing import OneHotEncoder

# Prepare symptoms data
diets = pd.read_csv(r'C:\Users\himan\OneDrive\Desktop\Database ML\diets.csv')
medications = pd.read_csv(r'C:\Users\himan\OneDrive\Desktop\Database ML\medications.csv')
precautions = pd.read_csv(r'C:\Users\himan\OneDrive\Desktop\Database ML\precautions_df.csv')
symptom_severity = pd.read_csv(r'C:\Users\himan\OneDrive\Desktop\Database ML\Symptom-severity.csv')
symptoms = pd.read_csv(r'C:\Users\himan\OneDrive\Desktop\Database ML\symtoms_df.csv')
workouts = pd.read_csv(r'C:\Users\himan\OneDrive\Desktop\Database ML\workout_df.csv')

# Prepare symptoms data
symptoms_long = symptoms.melt(id_vars=['Disease'], value_vars=[f'Symptom_{i}' for i in range(1, 5)], var_name='SymptomType', value_name='Symptom')
symptoms_severity = pd.merge(symptoms_long, symptom_severity, on='Symptom')

# One-hot encode symptoms
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_symptoms = encoder.fit_transform(symptoms_severity[['Symptom']])
encoded_symptoms_df = pd.DataFrame(encoded_symptoms, columns=encoder.get_feature_names_out(['Symptom']))
symptoms_severity_encoded = pd.concat([symptoms_severity.drop(['Symptom'], axis=1), encoded_symptoms_df], axis=1)

# Define custom healthcare environment class
class HealthcareEnv(gym.Env):
    def __init__(self):
        super(HealthcareEnv, self).__init__()

        # Define action space: actions based on treatments (diets, medications, workouts, precautions)
        self.n_actions = len(diets) + len(medications) + len(workouts)
        self.action_space = spaces.Discrete(self.n_actions)

        # Define observation space (features from the encoded symptom severity data)
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(symptoms_severity_encoded.columns),), dtype=np.float32)

        # Initialize state (random initial state based on symptoms severity)
        self.state = symptoms_severity_encoded.sample(1).iloc[0].values
        self.done = False

    def reset(self):
        # Reset the state to a new random sample of symptom severities
        self.state = symptoms_severity_encoded.sample(1).iloc[0].values
        self.done = False
        return np.array(self.state, dtype=np.float32)

    def step(self, action):
        # Simplified logic: reduce severity based on action (could be expanded)
        if action < len(diets):  # Diet action
            self.state = np.clip(self.state - np.random.rand(len(self.state)) * 0.1, 0, 1)
        elif action < len(diets) + len(medications):  # Medication action
            self.state = np.clip(self.state - np.random.rand(len(self.state)) * 0.2, 0, 1)
        else:  # Workout action
            self.state = np.clip(self.state - np.random.rand(len(self.state)) * 0.15, 0, 1)

        # Reward: lower symptom severity should give a higher reward
        reward = -np.sum(self.state)

        # End the episode if all symptoms are near zero (treatment success)
        self.done = np.all(self.state < 0.1)

        return np.array(self.state, dtype=np.float32), reward, self.done, {}

    def render(self, mode='human'):
        print(f"Current symptom severities: {self.state}")

# Initialize the healthcare environment
env = HealthcareEnv()

# Initialize the PPO model and start training
model = PPO('MlpPolicy', env, verbose=1)

from sklearn.preprocessing import LabelEncoder

# Assuming your state has a categorical column like 'disease'
label_encoder = LabelEncoder()

# Fit and transform the disease column
symptoms_severity_encoded['Disease'] = label_encoder.fit_transform(symptoms_severity_encoded['Disease'])


# Import necessary libraries
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv  # Corrected import
from sklearn.preprocessing import MultiLabelBinarizer
import gym

# Load your data
symptoms_df = pd.read_csv(r'C:\Users\himan\OneDrive\Desktop\Database ML\symtoms_df.csv')

# Check for any leading/trailing spaces in column names
symptoms_df.columns = symptoms_df.columns.str.strip()

# Preprocess the symptoms: convert symptom columns to lists of symptoms
symptoms_df['symptoms'] = symptoms_df[['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4']].apply(lambda x: [s.strip() for s in x.tolist() if isinstance(s, str)], axis=1)

# One-hot encode the symptoms using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
symptom_encoded = mlb.fit_transform(symptoms_df['symptoms'])

# Create a new DataFrame with encoded symptoms
encoded_symptoms_df = pd.DataFrame(symptom_encoded, columns=mlb.classes_)

# Concatenate encoded symptoms with the disease column
final_data = pd.concat([symptoms_df[['Disease']], encoded_symptoms_df], axis=1)

# Print the final data to verify encoding
print(final_data)

# Define the custom environment
class HealthcareEnv(gym.Env):
    def __init__(self):
        super(HealthcareEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(2)  # Example: Two possible actions
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(encoded_symptoms_df.shape[1],), dtype=np.float32)
        self.state = None
        self.done = False

    def reset(self):
        # Sample a random state (numeric representation)
        self.state = encoded_symptoms_df.sample(1).values.flatten()  # Flatten to 1D array
        self.done = False
        return np.array(self.state, dtype=np.float32)
         def step(self, action):
        # Your logic to determine the next state and reward goes here
        reward = 1 if action == 1 else -1  # Example: reward based on the action
        self.done = True  # End the episode after one step
        return self.state, reward, self.done, {}

# Create the environment
env = DummyVecEnv([lambda: HealthcareEnv()])

# Initialize the PPO model
model = PPO('MlpPolicy', env, verbose=1)

# Train the model for 10,000 timesteps
model.learn(total_timesteps=10000)

# Test the model with 10 steps
state = env.reset()
for _ in range(10):
    action, _states = model.predict(state)
    next_state, reward, done, _ = env.step(action)
    print(f"Action: {action}, Next State: {next_state}, Reward: {reward}")
    if done:
        state = env.reset()
    else:
        state = next_state

# Save the trained model
model.save("healthcare_recommendation_model")

# Load and test the saved model
model = PPO.load("healthcare_recommendation_model")

# Test again after loading the model
state = env.reset()
for _ in range(10):
    action, _states = model.predict(state)
    next_state, reward, done, _ = env.step(action)
    print(f"Action: {action}, Next State: {next_state}, Reward: {reward}")
    if done:
        state = env.reset()
    else:
        state = next_state
def get_health_recommendation():
    # Ask the user for their symptoms
    print("Please enter your symptoms (comma-separated):")
    user_input = input()
    
    # Preprocess the input to match the expected format
    user_symptoms = [symptom.strip() for symptom in user_input.split(",")]
    
    # Create a DataFrame to hold the user's symptoms
    user_symptoms_df = pd.DataFrame(columns=mlb.classes_)
    user_symptoms_df.loc[0] = mlb.transform([user_symptoms])[0]

    # Set the environment state to the user's symptoms
    env.state = user_symptoms_df.values.flatten()
    
    # Use the model to predict the best action
    action, _ = model.predict(env.state)
    
    # Interpret the action
    if action < len(diets):
        recommendation_type = "Diet"
        diet_list = eval(diets.iloc[action]['Diet'])  # Convert string representation of list to actual list
        recommendation = ", ".join(diet_list)  # Join the diets into a string for display
    elif action < len(diets) + len(medications):
        recommendation_type = "Medication"
        recommendation = medications.iloc[action - len(diets)]['Correct_Column_Name']  # Replace with actual column name
    else:
        recommendation_type = "Workout"
        recommendation = workouts.iloc[action - len(diets) - len(medications)]['Correct_Column_Name']  # Replace with actual column name

    # Print the recommendation
    print(f"We recommend the following {recommendation_type}: {recommendation}")

# Run the recommendation function
get_health_recommendation()

print(diets.head())  # Display the first few rows of the DataFrame
print(diets.columns)  # Print the names of the columns in the DataFrame