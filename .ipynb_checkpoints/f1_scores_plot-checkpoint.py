import mlflow
import pandas as pd
import matplotlib.pyplot as plt

# Set tracking URI
mlflow.set_tracking_uri("https://dagshub.com/vattikutilohith/wine_quality_project.mlflow")

# List of experiment names
experiment_names = [
    "Experiment #1",
    "Experiment #2",
    "Experiment #3",
    "Experiment #4",
    "PCA - Experiment #5",
    "Custom Experiment - Stacked Model #6",
    "Custom Experiment - Hyperparameter Tuning #7"
]

# Initialize lists to store results
experiments = []
cv_means = []
test_f1s = []

# Fetch data from MLFlow
for exp_name in experiment_names:
    experiment = mlflow.get_experiment_by_name(exp_name)
    if experiment:
        experiment_id = experiment.experiment_id
        runs = mlflow.search_runs(experiment_ids=[experiment_id])
        if not runs.empty:
            # Fetch the latest run
            latest_run = runs.iloc[0]
            print(f"Available keys in the run for {exp_name}: {latest_run.keys()}")  # Debugging
            try:
                # Dynamically retrieve available metric keys
                cv_mean = latest_run.get("metrics.CV Mean F1 Score") or latest_run.get("metrics.Mean CV F1 Score")
                test_f1 = latest_run.get("metrics.Test F1 Score")
                
                if cv_mean is not None and test_f1 is not None:
                    cv_means.append(cv_mean)
                    test_f1s.append(test_f1)
                    experiments.append(exp_name)
                else:
                    print(f"Metrics missing for {exp_name}: CV Mean F1 Score or Test F1 Score.")
            except KeyError as e:
                print(f"Metric missing for {exp_name}: {e}")
        else:
            print(f"No runs found for {exp_name}.")
    else:
        print(f"Experiment {exp_name} not found.")

# Ensure we have data to plot
if not experiments:
    print("No valid data available for plotting.")
else:
    # Create a DataFrame for visualization
    data = pd.DataFrame({
        "Experiments": experiments,
        "CV Mean F1 Score": cv_means,
        "Test F1 Score": test_f1s
    })

    # Drop rows with missing data
    data = data.dropna()

    # Plot the F1 score comparison
    plt.figure(figsize=(10, 6))
    data.plot(x="Experiments", kind="bar", figsize=(12, 6))
    plt.title("F1 Scores Comparison Across Experiments")
    plt.xlabel("Experiments")
    plt.ylabel("F1 Scores")
    plt.xticks(rotation=45)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()