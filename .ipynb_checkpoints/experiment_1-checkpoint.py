import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix
import mlflow

# Set the tracking URI
mlflow.set_tracking_uri("https://dagshub.com/vattikutilohith/wine_quality_project.mlflow")

# Define the experiment name
experiment_name = "Experiment #1"
mlflow.set_experiment(experiment_name)

# Load dataset
dataset_path = "./data/wine_features.csv"
labels_path = "./data/wine_labels.csv"
features = pd.read_csv(dataset_path)
labels = pd.read_csv(labels_path)

# Combine features and labels
data = pd.concat([features, labels], axis=1)

X = data.drop("quality", axis=1)
y = data["quality"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline for Logistic Regression
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])

# Perform cross-validation
with mlflow.start_run(run_name="Logistic Regression - Experiment #1"):
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1_macro")
    mean_score = scores.mean()
    std_score = scores.std()

    # Train the pipeline
    pipeline.fit(X_train, y_train)

    # Evaluate on the test set
    y_pred = pipeline.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="macro")
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Log metrics
    mlflow.log_param("classifier", "Logistic Regression")
    mlflow.log_metric("CV Mean F1 Score", mean_score)
    mlflow.log_metric("CV Std F1 Score", std_score)
    mlflow.log_metric("Test F1 Score", f1)
    mlflow.log_text(str(conf_matrix), "confusion_matrix.txt")

    print(f"Cross-Validation F1 Scores: {scores}")
    print(f"Mean CV F1-Score: {mean_score:.4f}, Std: {std_score:.4f}")
    print("Confusion Matrix:\n", conf_matrix)

print(f"Experiment '{experiment_name}' completed.")