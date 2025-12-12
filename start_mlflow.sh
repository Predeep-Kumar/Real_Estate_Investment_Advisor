#!/usr/bin/env bash
# Start MLflow server (background)
# Make executable: chmod +x start_mlflow.sh
set -e

DB_PATH="./mlflow_registry.db"
ARTIFACT_ROOT="./mlruns"
PORT=5000

mkdir -p "$ARTIFACT_ROOT"

echo "Starting MLflow server..."
mlflow server \
  --backend-store-uri "sqlite:///${DB_PATH}" \
  --default-artifact-root "${ARTIFACT_ROOT}" \
  --host 0.0.0.0 --port ${PORT} \
  > mlflow_server.log 2>&1 &

echo "MLflow server started. UI: http://localhost:${PORT}"
echo "Logs: mlflow_server.log"
