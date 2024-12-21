import os
import pandas as pd
import numpy as np
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import mlflow
import joblib

# Set credentials for DagsHub
os.environ["MLFLOW_TRACKING_USERNAME"] = "vattikutilohith"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "2a8c53e990014589f57319be4396a9d4d10b33c0"

# Set MLFlow tracking URI
mlflow.set_tracking_uri("https://dagshub.com/vattikutilohith/wine_quality_project.mlflow")

# Define the experiment name
experiment_name = "Custom Experiment - Stacked Model #6"
mlflow.set_experiment(experiment_name)

# Load the dataset
features_path = "./data/wine_features.csv"
labels_path = "./data/wine_labels.csv"
features = pd.read_csv(features_path)
labels = pd.read_csv(labels_path)

# Combine features and labels
data = pd.concat([features, labels], axis=1)

# Define features and target
X = data.drop("quality", axis=1)
y = data["quality"]

# Normalize the target variable
y = y - y.min()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define the base models
base_models = [
    ("rf", RandomForestClassifier(random_state=42)),
    ("xgb", XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="mlogloss"))
]

# Define the meta-classifier
meta_model = LogisticRegression(max_iter=1000, random_state=42)

# Define the Stacked Classifier
stacked_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("stacked", StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5))
])

# Start the MLFlow run
with mlflow.start_run(run_name="Stacked Model - Experiment #6"):
    # Perform cross-validation
    scores = cross_val_score(stacked_pipeline, X_train, y_train, cv=5, scoring="f1_macro")
    mean_score = np.mean(scores)
    std_score = np.std(scores)

    # Train the model
    stacked_pipeline.fit(X_train, y_train)

    # Test set evaluation
    y_pred = stacked_pipeline.predict(X_test)
    test_f1 = f1_score(y_test, y_pred, average="macro")
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Log metrics to MLFlow
    mlflow.log_param("Base Models", "Random Forest, XGBoost")
    mlflow.log_param("Meta Model", "Logistic Regression")
    mlflow.log_metric("CV Mean F1 Score", mean_score)
    mlflow.log_metric("CV Std F1 Score", std_score)
    mlflow.log_metric("Test F1 Score", test_f1)
    mlflow.log_text(str(conf_matrix), "confusion_matrix.txt")

    # Save the model
    model_path = "./outputs/stacked_model.joblib"
    joblib.dump(stacked_pipeline, model_path)
    mlflow.log_artifact(model_path, artifact_path="models")

    print(f"Cross-Validation F1 Scores: {scores}")
    print(f"Mean CV F1-Score: {mean_score:.4f}, Std: {std_score:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print("Confusion Matrix:\n", conf_matrix)

print(f"Experiment '{experiment_name}' completed.")