import argparse
import mlflow
from mlflow.tracking import MlflowClient

def create_experiment_only(name="NewExperiment", mlflow_uri=None):
    # Set MLflow tracking URI if provided
    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)

    client = MlflowClient()

    print("\n--- MLflow: Create Experiment ---")
    print(f"Requested name: '{name}'")

    # Check if experiment already exists
    existing = client.get_experiment_by_name(name)
    if existing:
        print(f"✔ Experiment '{name}' already exists. ID = {existing.experiment_id}")
        return existing.experiment_id

    # Create new experiment
    try:
        exp_id = client.create_experiment(name)
        print(f"✔ Created new experiment '{name}' with ID = {exp_id}")
        return exp_id
    except Exception as e:
        print(f"❌ Failed to create experiment '{name}': {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Create a new MLflow experiment.")
    parser.add_argument("--mlflow-uri", default=None, help="MLflow tracking URI (e.g. http://127.0.0.1:5000)")
    parser.add_argument("--name", default="NewExperiment", help="Name for the experiment")
    args = parser.parse_args()

    exp_id = create_experiment_only(name=args.name, mlflow_uri=args.mlflow_uri)
    if exp_id:
        print(f"\nDone — experiment ID: {exp_id}\n")
    else:
        print("\nExperiment creation failed.\n")

if __name__ == "__main__":
    main()
