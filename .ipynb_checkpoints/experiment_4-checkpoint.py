import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix
import mlflow
import joblib
import sqlite3

# Set credentials for DagsHub
os.environ["MLFLOW_TRACKING_USERNAME"] = "vattikutilohith"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "2a8c53e990014589f57319be4396a9d4d10b33c0"

# Set the tracking URI to your DagsHub repository
mlflow.set_tracking_uri("https://dagshub.com/vattikutilohith/wine_quality_project.mlflow")

# Define the experiment name
experiment_name = "Feature Selection - Experiment #4"

# Create or set the experiment
if not mlflow.get_experiment_by_name(experiment_name):
    mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)

# Load the dataset from SQLite database
db_path = "data/wine_data.db"
conn = sqlite3.connect(db_path)
data = pd.read_sql_query("SELECT * FROM wine_features", conn)
conn.close()

# Verify the final dataset
print("Final dataset columns:", data.columns)

if "quality" not in data.columns:
    raise ValueError("The target column 'quality' is missing in the dataset.")

X = data.drop("quality", axis=1)
y = data["quality"]

# Normalize the target variable
y = y - y.min()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform feature selection and train the model
with mlflow.start_run(run_name="Feature Selection Experiment"):
    # Variance Threshold
    print("Applying Variance Threshold...")
    var_thresh = VarianceThreshold(threshold=0.01)
    X_train_var = var_thresh.fit_transform(X_train)
    
    # Feature Importance using Random Forest
    print("Selecting features using Random Forest...")
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train_var, y_train)
    selector = SelectFromModel(rf, prefit=True)
    X_train_selected = selector.transform(X_train_var)
    
    # Pipeline with selected features
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(random_state=42))
    ])
    
    # Cross-validation
    print("Performing cross-validation...")
    scores = cross_val_score(pipeline, X_train_selected, y_train, cv=5, scoring='f1_macro')
    mean_score = scores.mean()
    std_score = scores.std()
    
    # Train and evaluate on test set
    pipeline.fit(X_train_selected, y_train)
    X_test_var = var_thresh.transform(X_test)
    X_test_selected = selector.transform(X_test_var)
    y_pred = pipeline.predict(X_test_selected)
    f1 = f1_score(y_test, y_pred, average="macro")
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Log parameters and metrics to MLFlow
    mlflow.log_param("Variance Threshold", 0.01)
    mlflow.log_metric("CV Mean F1 Score", mean_score)
    mlflow.log_metric("CV Std F1 Score", std_score)
    mlflow.log_metric("Test F1 Score", f1)
    
    # Log confusion matrix
    mlflow.log_text(str(conf_matrix), "confusion_matrix.txt")

    # Save the model
    output_model_path = "./outputs/feature_selected_model.joblib"
    os.makedirs("./outputs", exist_ok=True)
    joblib.dump(pipeline, output_model_path)
    mlflow.log_artifact(output_model_path, artifact_path="models")

    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Cross-Validation Mean F1 Score: {mean_score:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print(f"Experiment '{experiment_name}' completed successfully.")