import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np
import pandas as pd
import sqlite3
import os

# Set credentials for DagsHub
os.environ["MLFLOW_TRACKING_USERNAME"] = "vattikutilohith"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "2a8c53e990014589f57319be4396a9d4d10b33c0"

# Set MLFlow tracking to DagsHub
DAGS_HUB_URL = "https://dagshub.com/vattikutilohith/wine_quality_project.mlflow"
mlflow.set_tracking_uri(DAGS_HUB_URL)
mlflow.sklearn.autolog()

# Database location and table
db_path = "data/wine_data.db"
table_name = "wine_features"

# Load data from the database
conn = sqlite3.connect(db_path)
dataset = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
conn.close()

# Verify the dataset schema
print("Schema of 'wine_features':", dataset.dtypes)
print("Dataset columns:", dataset.columns)

# Feature Engineering: Create new features by combining existing ones
dataset['density_pH_ratio'] = dataset['density'] / dataset['pH']
dataset['total_acidity'] = dataset['fixed acidity'] + dataset['volatile acidity']
dataset['sulfur_to_alcohol'] = dataset['total sulfur dioxide'] / dataset['alcohol']

# Define features and target
X = dataset.drop("quality", axis=1)
y = dataset["quality"]

# Normalize the target variable
y = y - y.min()  # Ensures classes start at 0

# Train/test split with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create a pipeline with Logistic Regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

try:
    # Start MLFlow run for the experiment
    with mlflow.start_run(run_name="Feature Engineering - Experiment #3"):
        # Cross-validation
        scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1_macro')
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"Cross-Validation F1 Scores: {scores}")
        print(f"Mean CV F1-Score: {mean_score:.4f}, Std: {std_score:.4f}")

        # Fit on the full training data
        pipeline.fit(X_train, y_train)

        # Predict on test data
        y_pred = pipeline.predict(X_test)

        # Log confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:\n", cm)

        # Log metrics in MLFlow
        mlflow.log_metric("Mean CV F1 Score", mean_score)
        mlflow.log_metric("CV Std F1 Score", std_score)
        mlflow.log_metric("Test F1 Score", f1_score(y_test, y_pred, average="macro"))
        mlflow.log_text(str(cm), "confusion_matrix.txt")

        # Log confusion matrix metrics
        mlflow.log_metric("TP", cm[1][1] if cm.shape[0] > 1 else 0)
        mlflow.log_metric("TN", cm[0][0])
        mlflow.log_metric("FP", cm[0][1] if cm.shape[1] > 1 else 0)
        mlflow.log_metric("FN", cm[1][0] if cm.shape[0] > 1 else 0)

        print("Experiment #3 - Feature Engineering completed successfully.")

except mlflow.exceptions.MlflowException as e:
    print("Error occurred during the MLflow run. Please check your credentials or server status.")
    print("Error details:", str(e))