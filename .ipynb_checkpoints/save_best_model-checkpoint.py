import mlflow
import joblib

# Set the tracking URI to your DagsHub repository
mlflow.set_tracking_uri("https://dagshub.com/vattikutilohith/wine_quality_project.mlflow")

# Initialize variables to track the best model
best_run = None
best_score = -1

# Fetch all experiments
experiments = mlflow.search_experiments()

for experiment in experiments:
    experiment_id = experiment.experiment_id
    print(f"Checking Experiment: {experiment.name}")
    try:
        # Search for runs in the experiment
        runs = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string="metrics.Test F1 Score IS NOT NULL",  # Ensure metric exists
            order_by=["metrics.`Test F1 Score` DESC"]  # Use correct syntax for ordering
        )

        if not runs.empty:
            # Get the top run
            top_run = runs.iloc[0]
            score = top_run.get("metrics.Test F1 Score", None)

            if score and score > best_score:
                best_score = score
                best_run = top_run
    except Exception as e:
        print(f"Error processing experiment {experiment.name}: {e}")

# Check if a best run was found
if best_run is None:
    print("No valid runs found with Test F1 Score.")
else:
    print(f"Best run found: {best_run['run_id']} with Test F1 Score: {best_score}")

    # Fetch the logged model
    model_uri = f"runs:/{best_run['run_id']}/model"
    print(f"Best model URI: {model_uri}")

    # Load and save the model locally
    try:
        model = mlflow.sklearn.load_model(model_uri)
        joblib.dump(model, "best_model.joblib")
        print("Best model saved as 'best_model.joblib'")
    except Exception as e:
        print(f"Error loading or saving the best model: {e}")