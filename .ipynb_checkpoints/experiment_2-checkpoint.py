import sqlite3  # Ensure sqlite3 is imported
import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score

# Set the MLFlow tracking URI
mlflow.set_tracking_uri("https://dagshub.com/vattikutilohith/wine_quality_project.mlflow")

# Load the dataset
db_path = "/Users/lohithvattikuti/Desktop/Wine_Quality_project/data/wine_data.db"
table_name = "wine_features"

# Load data directly from the database
conn = sqlite3.connect(db_path)  # Ensure sqlite3 is used here
data = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
conn.close()

# Split features and target
X = data.drop("quality", axis=1)
y = data["quality"]

# Normalize the target variable
y = y - y.min()  # Ensures classes start at 0
print("Target classes after normalization:", sorted(y.unique()))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Experiment 2 - Logistic Regression, Ridge Classifier, Random Forest, XGBoost
experiment_name = "Experiment #2"
mlflow.set_experiment(experiment_name)

# Logistic Regression
print("Running Logistic Regression...")
try:
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(random_state=42, max_iter=1000))
    ])
    lr_scores = cross_val_score(lr_pipeline, X_train, y_train, cv=5, scoring="f1_macro")
    lr_pipeline.fit(X_train, y_train)
    y_pred_lr = lr_pipeline.predict(X_test)
    lr_f1 = f1_score(y_test, y_pred_lr, average="macro")
    
    with mlflow.start_run(run_name="Logistic Regression - Experiment #2"):
        mlflow.log_metric("Mean CV F1 Score", lr_scores.mean())
        mlflow.log_metric("Test F1 Score", lr_f1)
    print(f"Logistic Regression completed successfully. Mean CV F1 Score: {lr_scores.mean()}")
except Exception as e:
    print("Error during Logistic Regression:", str(e))

# Ridge Classifier
print("Running Ridge Classifier...")
try:
    ridge_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RidgeClassifier())
    ])
    ridge_scores = cross_val_score(ridge_pipeline, X_train, y_train, cv=5, scoring="f1_macro")
    ridge_pipeline.fit(X_train, y_train)
    y_pred_ridge = ridge_pipeline.predict(X_test)
    ridge_f1 = f1_score(y_test, y_pred_ridge, average="macro")
    
    with mlflow.start_run(run_name="Ridge Classifier - Experiment #2"):
        mlflow.log_metric("Mean CV F1 Score", ridge_scores.mean())
        mlflow.log_metric("Test F1 Score", ridge_f1)
    print(f"Ridge Classifier completed successfully. Mean CV F1 Score: {ridge_scores.mean()}")
except Exception as e:
    print("Error during Ridge Classifier:", str(e))

# Random Forest
print("Running Random Forest...")
try:
    rf_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(random_state=42))
    ])
    rf_scores = cross_val_score(rf_pipeline, X_train, y_train, cv=5, scoring="f1_macro")
    rf_pipeline.fit(X_train, y_train)
    y_pred_rf = rf_pipeline.predict(X_test)
    rf_f1 = f1_score(y_test, y_pred_rf, average="macro")
    
    with mlflow.start_run(run_name="Random Forest - Experiment #2"):
        mlflow.log_metric("Mean CV F1 Score", rf_scores.mean())
        mlflow.log_metric("Test F1 Score", rf_f1)
    print(f"Random Forest completed successfully. Mean CV F1 Score: {rf_scores.mean()}")
except Exception as e:
    print("Error during Random Forest:", str(e))

# XGBoost Classifier
print("Running XGBoost...")
try:
    xgb_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="mlogloss"))
    ])
    xgb_scores = cross_val_score(xgb_pipeline, X_train, y_train, cv=5, scoring="f1_macro")
    xgb_pipeline.fit(X_train, y_train)
    y_pred_xgb = xgb_pipeline.predict(X_test)
    xgb_f1 = f1_score(y_test, y_pred_xgb, average="macro")
    
    with mlflow.start_run(run_name="XGBoost - Experiment #2"):
        mlflow.log_metric("Mean CV F1 Score", xgb_scores.mean())
        mlflow.log_metric("Test F1 Score", xgb_f1)
    print(f"XGBoost completed successfully. Mean CV F1 Score: {xgb_scores.mean()}")
except Exception as e:
    print("Error during XGBoost:", str(e))