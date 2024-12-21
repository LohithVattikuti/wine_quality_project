import joblib
from xgboost import XGBClassifier

# Path to the model file in the outputs directory
model_path = "./outputs/stacked_model.joblib"

try:
    # Load the model
    model = joblib.load(model_path)
    print("Model loaded successfully!")
    
    # Create test input data (example input for prediction)
    sample_data = [[7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]]
    
    # Make a prediction using the loaded model
    prediction = model.predict(sample_data)
    print(f"Prediction: {prediction}")
except Exception as e:
    print(f"Error: {e}")