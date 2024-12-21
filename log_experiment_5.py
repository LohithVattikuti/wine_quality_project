import mlflow

# Set tracking URI
mlflow.set_tracking_uri("https://dagshub.com/vattikutilohith/wine_quality_project.mlflow")

# Log Experiment #5 manually
mlflow.set_experiment("PCA - Experiment #5")
with mlflow.start_run(run_name="PCA Dimensionality Reduction - Random Forest"):
    mlflow.log_metric("CV Mean F1 Score", 0.356)  # Replace with actual values
    mlflow.log_metric("CV Std F1 Score", 0.045)
    mlflow.log_metric("Test F1 Score", 0.324)
    print("Experiment #5 metrics logged successfully.")