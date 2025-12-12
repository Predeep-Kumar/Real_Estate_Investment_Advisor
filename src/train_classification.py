import os
import json
import argparse
import tempfile
import shutil
from typing import Optional, Tuple, Dict, List, Any
import pandas as pd
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException


class JoblibModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import joblib as _joblib
        model_path = context.artifacts["model_file"]
        self.model = _joblib.load(model_path)

    def predict(self, context: mlflow.pyfunc.PythonModelContext, model_input: pd.DataFrame) -> Any:
        return self.model.predict(model_input)


def sanitize_name(s: str) -> str:
    return s.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").lower()


def start_run_log_model_with_folders(
    model_name: str,
    model_file: Optional[str],
    params: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, float]] = None,
    registered_model_name: Optional[str] = None,
) -> Tuple[str, str]:
    params = params or {}
    metrics = metrics or {}

    with mlflow.start_run(run_name=model_name) as run:
        run_id = run.info.run_id

        try:
            if params:
                mlflow.log_params(params)
        except Exception:
            pass
        try:
            if metrics:
                mlflow.log_metrics(metrics)
        except Exception:
            pass

        tmpdir = tempfile.mkdtemp(prefix=f"mlrun_{run_id}_")
        try:
            artifacts_dir = os.path.join(tmpdir, "artifacts")
            metrics_dir = os.path.join(tmpdir, "metrics")
            params_dir = os.path.join(tmpdir, "params")
            logs_dir = os.path.join(tmpdir, "logs")
            os.makedirs(artifacts_dir, exist_ok=True)
            os.makedirs(metrics_dir, exist_ok=True)
            os.makedirs(params_dir, exist_ok=True)
            os.makedirs(logs_dir, exist_ok=True)

            try:
                with open(os.path.join(params_dir, "params.json"), "w", encoding="utf-8") as f:
                    json.dump(params, f, indent=2)
            except Exception:
                pass
            try:
                with open(os.path.join(metrics_dir, "metrics.json"), "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2)
            except Exception:
                pass
            try:
                with open(os.path.join(logs_dir, "run.log"), "w", encoding="utf-8") as lf:
                    lf.write(f"Run id: {run_id}\nModel: {model_name}\n")
            except Exception:
                pass

            if not model_file or not os.path.exists(model_file):
                raise FileNotFoundError(f"Model file missing: {model_file}")
            try:
                local_joblib_path = os.path.join(artifacts_dir, os.path.basename(model_file))
                shutil.copy(model_file, local_joblib_path)
            except Exception:
                local_joblib_path = model_file

            try:
                mlflow.log_artifacts(tmpdir, artifact_path="run_contents")
            except Exception:
                pass

            try:
                mlflow.pyfunc.log_model(
                    name="model",
                    python_model=JoblibModelWrapper(),
                    artifacts={"model_file": local_joblib_path},
                    registered_model_name=registered_model_name,
                )
            except TypeError:
                try:
                    mlflow.pyfunc.log_model(
                        artifact_path="model",
                        python_model=JoblibModelWrapper(),
                        artifacts={"model_file": local_joblib_path},
                        registered_model_name=registered_model_name,
                    )
                except Exception:
                    pass
            except Exception:
                try:
                    mlflow.pyfunc.log_model(
                        artifact_path="model",
                        python_model=JoblibModelWrapper(),
                        artifacts={"model_file": local_joblib_path},
                        registered_model_name=registered_model_name,
                    )
                except Exception:
                    pass

            artifact_uri = mlflow.get_artifact_uri()
        finally:
            try:
                shutil.rmtree(tmpdir)
            except Exception:
                pass

    return run_id, artifact_uri


def choose_best_classification(clf_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not clf_list:
        raise RuntimeError("No classification candidates provided.")

    normalized = []
    for e in clf_list:
        entry = dict(e)
        metrics = entry.get("metrics")
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                entry.setdefault(k, v)
        normalized.append(entry)

    df = pd.DataFrame(normalized)

    for col in ["ROC_AUC", "F1_Score", "Accuracy"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df_sorted = df.sort_values(
        by=["ROC_AUC", "F1_Score", "Accuracy"],
        ascending=[False, False, False]
    )

    best = df_sorted.iloc[0].to_dict()

    run_id = best.get("run_id")
    if run_id:
        for orig in clf_list:
            if orig.get("run_id") == run_id:
                return orig
    return best


def create_and_transition_model_version(
    client: MlflowClient,
    registered_name: str,
    run_id: str,
    artifact_subpath="model",
    promote_to_production=False,
) -> Tuple[Optional[str], Optional[str]]:
    try:
        try:
            client.get_registered_model(registered_name)
        except Exception:
            client.create_registered_model(registered_name)
    except Exception as e:
        print(f"[WARN] Could not ensure registered model '{registered_name}': {e}")

    try:
        mv = client.create_model_version(
            name=registered_name,
            source=f"runs:/{run_id}/{artifact_subpath}",
            run_id=run_id,
        )
    except MlflowException as e:
        print(f"[ERROR] create_model_version failed for {registered_name} from run {run_id}: {e}")
        return None, None
    except Exception as e:
        print(f"[ERROR] Unexpected error creating model version for {registered_name}: {e}")
        return None, None

    try:
        if promote_to_production:
            client.transition_model_version_stage(
                name=registered_name,
                version=mv.version,
                stage="Production",
                archive_existing_versions=True,
            )
            stage = "Production"
        else:
            client.transition_model_version_stage(
                name=registered_name,
                version=mv.version,
                stage="Staging",
                archive_existing_versions=False,
            )
            stage = "Staging"
    except Exception as e:
        print(f"[WARN] Could not transition model version {registered_name} v{mv.version}: {e}")
        stage = getattr(mv, "current_stage", "None")

    return mv.version, stage


def main(base_path: str = ".", mlflow_uri: str = "http://127.0.0.1:5000"):
    models_path = os.path.join(base_path, "models")
    metrics_path = os.path.join(base_path, "reports", "metrics")
    config_path = os.path.join(base_path, "config")
    os.makedirs(config_path, exist_ok=True)

    mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient(tracking_uri=mlflow_uri)

    clf_csv = os.path.join(metrics_path, "classification_all_models_metrics.csv")
    if not os.path.exists(clf_csv):
        raise FileNotFoundError(f"{clf_csv} not found.")

    clf_df = pd.read_csv(clf_csv)

    clf_file_map = {
        "Logistic Regression (Tuned)": "clf_logistic_regression_tuned.joblib",
        "Random Forest (Tuned)": "clf_random_forest_tuned.joblib",
        "XGBoost Classifier (Tuned)": "clf_xgboost_tuned.joblib",
    }

    clf_summary: List[Dict[str, Any]] = []
    for _, row in clf_df.iterrows():
        model_name = row.get("Model") or row.get("model") or "<unknown>"
        joblib_name = clf_file_map.get(model_name, sanitize_name(model_name) + ".joblib")
        joblib_path = os.path.normpath(os.path.join(models_path, joblib_name))

        params = {"model_name": model_name}
        metrics = {
            "Accuracy": float(row.get("Accuracy", 0) or 0),
            "Precision": float(row.get("Precision", 0) or 0),
            "Recall": float(row.get("Recall", 0) or 0),
            "F1_Score": float(row.get("F1_Score", 0) or 0),
            "ROC_AUC": float(row.get("ROC_AUC", 0) or 0),
            "CV_ROC_AUC_Mean": float(row.get("Base_CV_ROC_AUC_Mean", 0) or 0),
            "CV_ROC_AUC_Std": float(row.get("Base_CV_ROC_AUC_Std", 0) or 0),
        }

        try:
            run_id, art_uri = start_run_log_model_with_folders(
                model_name=model_name,
                model_file=joblib_path,
                params=params,
                metrics=metrics,
                registered_model_name=None,
            )
        except FileNotFoundError as fnf:
            print(f"[WARN] Skipping {model_name}: {fnf}")
            continue
        except Exception as ex:
            print(f"[WARN] Failed to log {model_name}: {ex}")
            continue

        entry: Dict[str, Any] = {
            "model": model_name,
            "joblib": joblib_path,
            "run_id": run_id,
            "artifact_uri": art_uri,
            **metrics,
            "params": params,
            "metrics": metrics,
        }
        clf_summary.append(entry)
        print(f"Logged {model_name} -> run {run_id} (artifacts: {art_uri})")

    summary_path = os.path.join(config_path, "mlflow_logged_classification_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"classification": clf_summary}, f, indent=2)
    print("Saved classification logging summary:", summary_path)

    if len(clf_summary) == 0:
        raise RuntimeError("No classification models were logged. Aborting registration.")

    best_clf = choose_best_classification(clf_summary)
    best_model_name = best_clf.get("model")
    print("Best classification candidate:", best_model_name)

    registered_all: List[Dict[str, Any]] = []
    prefix = "Classification"

    for entry in clf_summary:
        model_name = entry["model"]
        run_id = entry["run_id"]
        reg_name = f"{prefix}_{sanitize_name(model_name)}"
        promote = (model_name == best_model_name)

        version, stage = create_and_transition_model_version(
            client=client,
            registered_name=reg_name,
            run_id=run_id,
            artifact_subpath="model",
            promote_to_production=promote
        )
        if version is None:
            print(f"[WARN] Registration failed for {model_name} (run {run_id})")
            continue

        try:
            if promote:
                desc = f"Registered from run {run_id}. MARKED AS (best). Promoted to Production."
            else:
                desc = f"Registered from run {run_id}."
            try:
                client.update_registered_model(name=reg_name, description=desc)
            except Exception:
                try:
                    client.set_registered_model_tag(reg_name, "description", desc)
                except Exception:
                    pass
        except Exception:
            pass

        display_name = model_name + (" (best)" if promote else "")
        registered_all.append({
            "model": model_name,
            "display_name": display_name,
            "params": entry.get("params", {}),
            "metrics": entry.get("metrics", {}),
            "joblib": entry.get("joblib"),
            "run_id": run_id,
            "artifact_uri": entry.get("artifact_uri"),
            "registered_name": reg_name,
            "registered_version": version,
            "stage": stage,
            "is_best": promote,
        })

        print(f"Registered {model_name} as {reg_name} v{version} (stage={stage}, best={promote})")

    all_config = {
        "mlflow_uri": mlflow_uri,
        "registered_models": registered_all,
        "best_model": next((m for m in registered_all if m["is_best"]), None)
    }
    config_file = os.path.join(config_path, "mlflow_all_models_config.json")
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(all_config, f, indent=2)
    print("Saved comprehensive models config:", config_file)

    best_info = all_config["best_model"]
    if best_info:
        final_path = os.path.join(config_path, "best_classification_registered.json")
        with open(final_path, "w", encoding="utf-8") as f:
            json.dump({
                "model": best_info["model"],
                "joblib": best_info["joblib"],
                "run_id": best_info["run_id"],
                "artifact_uri": best_info["artifact_uri"],
                "registered_name": best_info["registered_name"],
                "registered_version": best_info["registered_version"],
                "production_uri": f"models:/{best_info['registered_name']}/Production"
            }, f, indent=2)
        print("Saved best classification registered info:", final_path)
    else:
        print("[WARN] No best classification model recorded.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", default=".", help="project root path")
    parser.add_argument("--mlflow_uri", default="http://127.0.0.1:5000", help="mlflow tracking uri")
    args = parser.parse_args()
    main(base_path=args.base_path, mlflow_uri=args.mlflow_uri)
