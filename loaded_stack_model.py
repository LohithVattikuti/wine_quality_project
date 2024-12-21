import os
import joblib

# Define the path to the model file
local_model_path = "./stacked_model.joblib"  # Replace this with the actual path if it's in a subdirectory

# Verify the file exists and load it
if os.path.exists(local_model_path):
    model = joblib.load(local_model_path)
    print("Model loaded successfully.")
else:
    print(f"Model file not found at {local_model_path}. Please verify the path.")