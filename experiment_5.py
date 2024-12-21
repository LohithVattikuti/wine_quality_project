import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, confusion_matrix
import mlflow
import matplotlib.pyplot as plt

# Set MLFlow credentials for DagsHub
os.environ["MLFLOW_TRACKING_USERNAME"] = "vattikutilohith"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "2a8c53e990014589f57319be4396a9d4d10b33c0"

# Set the tracking URI for DagsHub
mlflow.set_tracking_uri("https://dagshub.com/vattikutilohith/wine_quality_project.mlflow")

# Experiment name
experiment_name = "Experiment #5 - PCA Dimensionality Reduction"
mlflow.set_experiment(experiment_name)

# Load the dataset
features_path = "./data/wine_features.csv"
labels_path = "./data/wine_labels.csv"
features = pd.read_csv(features_path)
labels = pd.read_csv(labels_path)

# Combine features and labels
data = pd.concat([features, labels], axis=1)

# Separate features and target
X = data.drop("quality", axis=1)
y = data["quality"]

# Normalize the target variable
y = y - y.min()

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# PCA analysis
pca = PCA()
X_pca = pca.fit_transform(StandardScaler().fit_transform(X))

# Scree plot
explained_variance_ratio = pca.explained_variance_ratio_
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', linestyle='--')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.grid()
plt.savefig("./outputs/scree_plot.png")
print("Scree plot saved as 'scree_plot.png'.")

# Perform PCA with selected components
n_components = 8  # Adjust based on the scree plot
pca = PCA(n_components=n_components)

# Create a pipeline with PCA and Random Forest
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", pca),
    ("classifier", RandomForestClassifier(random_state=42))
])

# Run the experiment
with mlflow.start_run(run_name="PCA Dimensionality Reduction - Random Forest"):
    # Cross-validation
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1_macro")
    mean_cv_score = scores.mean()
    std_cv_score = scores.std()

    # Train the pipeline
    pipeline.fit(X_train, y_train)

    # Predict on the test set
    y_pred = pipeline.predict(X_test)
    test_f1_score = f1_score(y_test, y_pred, average="macro")
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Log parameters, metrics, and artifacts
    mlflow.log_param("n_components", n_components)
    mlflow.log_metric("CV Mean F1 Score", mean_cv_score)
    mlflow.log_metric("CV Std F1 Score", std_cv_score)
    mlflow.log_metric("Test F1 Score", test_f1_score)
    mlflow.log_artifact("./outputs/scree_plot.png", artifact_path="plots")
    mlflow.log_text(str(conf_matrix), "confusion_matrix.txt")

    print(f"Cross-Validation F1 Scores: {scores}")
    print(f"Mean CV F1-Score: {mean_cv_score:.4f}, Std: {std_cv_score:.4f}")
    print(f"Test F1-Score: {test_f1_score:.4f}")
    print("Confusion Matrix:\n", conf_matrix)

print(f"Experiment '{experiment_name}' completed.")