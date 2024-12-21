from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
from typing import List

# Load the new stacked model (without xgboost)
model_path = "./outputs/stacked_model_no_xgboost.joblib"  # Updated model path
try:
    model = joblib.load(model_path)
    print("Model loaded successfully!")
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

# Initialize FastAPI app
app = FastAPI(
    title="Wine Quality Prediction API",
    description="API to predict wine quality based on input features",
    version="1.0.0",
)

# Define input schema
class WineFeatures(BaseModel):
    fixed_acidity: float = Field(..., example=7.4)
    volatile_acidity: float = Field(..., example=0.7)
    citric_acid: float = Field(..., example=0.0)
    residual_sugar: float = Field(..., example=1.9)
    chlorides: float = Field(..., example=0.076)
    free_sulfur_dioxide: float = Field(..., example=11.0)
    total_sulfur_dioxide: float = Field(..., example=34.0)
    density: float = Field(..., example=0.9978)
    pH: float = Field(..., example=3.51)
    sulphates: float = Field(..., example=0.56)
    alcohol: float = Field(..., example=9.4)

    class Config:
        schema_extra = {
            "example": {
                "fixed_acidity": 7.4,
                "volatile_acidity": 0.7,
                "citric_acid": 0.0,
                "residual_sugar": 1.9,
                "chlorides": 0.076,
                "free_sulfur_dioxide": 11.0,
                "total_sulfur_dioxide": 34.0,
                "density": 0.9978,
                "pH": 3.51,
                "sulphates": 0.56,
                "alcohol": 9.4,
            }
        }

class BatchWineFeatures(BaseModel):
    data: List[WineFeatures]

# Define a health check endpoint
@app.get("/health")
def health_check():
    return {"status": "API is running!"}

# Define single prediction endpoint
@app.post("/predict")
def predict(features: WineFeatures):
    """
    Predict wine quality based on input features.
    """
    try:
        # Convert features to numpy array for prediction
        feature_values = np.array([
            features.fixed_acidity,
            features.volatile_acidity,
            features.citric_acid,
            features.residual_sugar,
            features.chlorides,
            features.free_sulfur_dioxide,
            features.total_sulfur_dioxide,
            features.density,
            features.pH,
            features.sulphates,
            features.alcohol,
        ]).reshape(1, -1)

        # Use the model to predict
        prediction = model.predict(feature_values)
        return {"prediction": int(prediction[0])}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {e}")

# Define batch prediction endpoint
@app.post("/predict_batch")
def predict_batch(batch: BatchWineFeatures):
    """
    Predict wine quality for a batch of inputs.
    """
    try:
        # Convert batch data into numpy array
        batch_features = np.array([
            [
                item.fixed_acidity,
                item.volatile_acidity,
                item.citric_acid,
                item.residual_sugar,
                item.chlorides,
                item.free_sulfur_dioxide,
                item.total_sulfur_dioxide,
                item.density,
                item.pH,
                item.sulphates,
                item.alcohol,
            ] for item in batch.data
        ])

        predictions = model.predict(batch_features)
        return {"predictions": predictions.tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making batch predictions: {e}")