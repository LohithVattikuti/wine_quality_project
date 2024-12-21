import mlflow
import joblib
import os

# Set the tracking URI
mlflow.set_tracking_uri("https://dagshub.com/vattikutilohith/wine_quality_project.mlflow")

# Best run tracking variables
best_run = None
best_score = -1

# Fetch all experiments
experiments = mlflow.search_experiments()

for experiment in experiments:
    experiment_id = experiment.experiment_id
    print(f"Checking Experiment: {experiment.name}")
    try:
        # Fetch runs
        runs = mlflow.search_runs(experiment_ids=[experiment_id])
        
        if not runs.empty:
            print(f"Available runs for experiment {experiment.name}:")
            print(runs[["run_id", "metrics.Test F1 Score", "metrics.CV Mean F1 Score"]])
            
            # Find the best run based on Test F1 Score
            for _, run in runs.iterrows():
                score = run.get("metrics.Test F1 Score")
                if score is not None and score > best_score:
                    best_score = score
                    best_run = run
        else:
            print(f"No runs found for experiment {experiment.name}.")
    except Exception as e:
        print(f"Error processing experiment {experiment.name}: {e}")

# Load and save the best model
if best_run is not None:
    print(f"Best run found: {best_run['run_id']} with Test F1 Score: {best_score}")
    model_uri = f"runs:/{best_run['run_id']}/model"
    print(f"Best model URI: {model_uri}")

    # Download the model
    try:
        model = mlflow.sklearn.load_model(model_uri)
        joblib.dump(model, "best_model.joblib")
        print("Best model saved as 'best_model.joblib'")
    except Exception as e:
        print(f"Error loading or saving the best model: {e}")
        print("Attempting manual download...")
        
        # Attempt manual download
        artifact_uri = best_run.get("artifact_uri", "")
        print(f"Artifact URI: {artifact_uri}")
else:
    print("No valid runs found with Test F1 Score.")