import mlflow

# Set Tracking URI
mlflow.set_tracking_uri("https://dagshub.com/vattikutilohith/wine_quality_project.mlflow")

# Test connection
with mlflow.start_run(run_name="Test Connection"):
    mlflow.log_param("test_param", "test_value")
print("Connection successful!")