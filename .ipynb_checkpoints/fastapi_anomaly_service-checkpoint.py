from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load the saved model
model_path = "refined_isolation_forest_model.joblib"
model = joblib.load(model_path)

# Initialize FastAPI app
app = FastAPI()

# Define input data model
class ProcessFeatures(BaseModel):
    process_duration: float
    activity_count: int
    unique_activities: int

# Define prediction endpoint
@app.post("/predict")
def predict(features: ProcessFeatures):
    # Convert input features to a format suitable for the model
    input_data = [[
        features.process_duration,
        features.activity_count,
        features.unique_activities
    ]]
    
    # Make prediction
    prediction = model.predict(input_data)
    result = "Anomaly" if prediction[0] == -1 else "Normal"
    
    return {"prediction": result}