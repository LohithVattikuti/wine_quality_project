import joblib

# Define the path to the stacked model
model_path = "/Users/lohithvattikuti/Desktop/Wine_Quality_project/stacked_model.joblib"

# Load the model
try:
    model = joblib.load(model_path)
    print("Model loaded successfully!")
except FileNotFoundError:
    print(f"Model file not found at path: {model_path}")
except Exception as e:
    print(f"An error occurred while loading the model: {e}")


