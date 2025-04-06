#!/bin/bash
set -e

echo "Starting MLflow server..."

# Create directories
mkdir -p /mlflow/artifacts /mlflow/logs

# Default backend store is SQLite
BACKEND_STORE_URI=${BACKEND_STORE_URI:-"sqlite:///mlflow/mlflow.db"}
echo "Using backend store: $BACKEND_STORE_URI"

# Use local file system for artifact storage
echo "Using local file system for artifact storage"
DEFAULT_ARTIFACT_ROOT="file:///mlflow/artifacts"

# Log configuration for debugging
echo "Starting MLflow server with config:"
echo "- Backend store URI: $BACKEND_STORE_URI"
echo "- Default artifact root: $DEFAULT_ARTIFACT_ROOT"
echo "- Host: 0.0.0.0"
echo "- Port: 5000"

# Launch MLflow with appropriate parameters
mlflow server \
    --backend-store-uri="$BACKEND_STORE_URI" \
    --default-artifact-root="$DEFAULT_ARTIFACT_ROOT" \
    --host=0.0.0.0 \
    --port=5000 \
    --workers=4