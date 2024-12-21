import mlflow

# Set tracking URI
mlflow.set_tracking_uri("https://dagshub.com/vattikutilohith/wine_quality_project.mlflow")

# Log Experiment #3 manually
mlflow.set_experiment("Experiment #3")
with mlflow.start_run(run_name="Feature Engineering - Experiment #3"):
    mlflow.log_metric("CV Mean F1 Score", 0.317)  # Replace with actual values
    mlflow.log_metric("CV Std F1 Score", 0.056)
    mlflow.log_metric("Test F1 Score", 0.277)
    print("Experiment #3 metrics logged successfully.")

# Log Experiment #4 manually
mlflow.set_experiment("Experiment #4")
with mlflow.start_run(run_name="Feature Selection - Experiment #4"):
    mlflow.log_metric("CV Mean F1 Score", 0.368)  # Replace with actual values
    mlflow.log_metric("CV Std F1 Score", 0.032)
    mlflow.log_metric("Test F1 Score", 0.31)
    print("Experiment #4 metrics logged successfully.")

# Log Experiment #5 manually
mlflow.set_experiment("Experiment #5")
with mlflow.start_run(run_name="PCA - Experiment #5"):
    mlflow.log_metric("CV Mean F1 Score", 0.356)  # Replace with actual values
    mlflow.log_metric("CV Std F1 Score", 0.045)
    mlflow.log_metric("Test F1 Score", 0.324)
    print("Experiment #5 metrics logged successfully.")