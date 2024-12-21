import os
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, confusion_matrix
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Set credentials for DagsHub
os.environ["MLFLOW_TRACKING_USERNAME"] = "vattikutilohith"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "2a8c53e990014589f57319be4396a9d4d10b33c0"  # Token

# Set the tracking URI to your DagsHub repository
mlflow.set_tracking_uri("https://dagshub.com/vattikutilohith/wine_quality_project.mlflow")

# Define the experiment name
experiment_name = "Custom Experiment - Hyperparameter Tuning #7"

# Create or set the experiment
mlflow.set_experiment(experiment_name)

# Load the dataset
features_path = "./data/wine_features.csv"
labels_path = "./data/wine_labels.csv"

# Load features and labels
features = pd.read_csv(features_path)
labels = pd.read_csv(labels_path)

# Combine features and labels
data = pd.concat([features, labels], axis=1)

# Define features and target
X = data.drop("quality", axis=1)
y = data["quality"]

# Normalize target variable
y = y - y.min()  # Ensure classes start at 0

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the XGBoost Classifier and parameter grid
xgb = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
param_grid = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "n_estimators": [50, 100, 200],
}

# Perform GridSearchCV
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring="f1_macro", cv=5, verbose=1, n_jobs=-1)

# Start MLFlow run for the experiment
with mlflow.start_run(run_name="XGBoost Hyperparameter Tuning - Experiment #7"):
    # Fit GridSearchCV
    grid_search.fit(X_train_scaled, y_train)
    
    # Get the best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    mean_cv_score = grid_search.best_score_
    
    # Evaluate on the test set
    y_pred = best_model.predict(X_test_scaled)
    test_f1 = f1_score(y_test, y_pred, average="macro")
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Log parameters and metrics
    mlflow.log_params(best_params)
    mlflow.log_metric("Mean CV F1 Score", mean_cv_score)
    mlflow.log_metric("Test F1 Score", test_f1)
    
    # Log confusion matrix
    mlflow.log_text(str(conf_matrix), "confusion_matrix.txt")
    
    # Save and log the best model
    model_path = "./outputs/xgboost_best_model.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="models")

    print(f"Best Parameters: {best_params}")
    print(f"Mean CV F1 Score: {mean_cv_score:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print("Confusion Matrix:\n", conf_matrix)

print(f"Experiment '{experiment_name}' completed.")