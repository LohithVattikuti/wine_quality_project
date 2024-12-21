import joblib
import pandas as pd

# Load the model
model_path = "best_model_xgboost.joblib"
model = joblib.load(model_path)
print("Model loaded successfully.")

# Load test dataset
test_data_path = "data/wine_features.csv"  # Update path if needed
test_labels_path = "data/wine_labels.csv"

features = pd.read_csv(test_data_path)
labels = pd.read_csv(test_labels_path)

# Combine features and labels
test_data = pd.concat([features, labels], axis=1)

# Prepare test inputs
X_test = test_data.drop("quality", axis=1)
y_test = test_data["quality"]

# Predict using the loaded model
predictions = model.predict(X_test)

print(f"Predictions: {predictions[:10]}")  # Display the first 10 predictions