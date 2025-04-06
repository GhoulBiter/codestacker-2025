#!/bin/bash

# List of directories to create
directories=(
    "src/codestacker/app"
    "src/codestacker/training"
    "src/codestacker/common"
    "docker"
    "mlflow"
    ".github/workflows"
)

for dir in "${directories[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo "Created directory: $dir"
    fi
done

# Create root-level files if they do not exist
rootFiles=("pyproject.toml" "poetry.toml" "README.md" "LICENSE")
for file in "${rootFiles[@]}"; do
    if [ ! -f "$file" ]; then
        touch "$file"
        echo "Created file: $file"
    fi
done

# Create placeholder files in the src/codestacker/app directory
appFiles=("src/codestacker/app/__init__.py" "src/codestacker/app/main.py" "src/codestacker/app/utils.py")
for file in "${appFiles[@]}"; do
    if [ ! -f "$file" ]; then
        touch "$file"
        echo "Created file: $file"
    fi
done

# Create placeholder files in the src/codestacker/training directory
trainingFiles=("src/codestacker/training/__init__.py" "src/codestacker/training/train.py" "src/codestacker/training/preprocessing.py")
for file in "${trainingFiles[@]}"; do
    if [ ! -f "$file" ]; then
        touch "$file"
        echo "Created file: $file"
    fi
done

# Create placeholder files in the src/codestacker/common directory
commonFiles=("src/codestacker/common/__init__.py" "src/codestacker/common/logger.py" "src/codestacker/common/utils.py")
for file in "${commonFiles[@]}"; do
    if [ ! -f "$file" ]; then
        touch "$file"
        echo "Created file: $file"
    fi
done

# Create placeholder files in the docker directory
dockerFiles=("docker/Dockerfile.app" "docker/Dockerfile.train" "docker/docker-compose.yml")
for file in "${dockerFiles[@]}"; do
    if [ ! -f "$file" ]; then
        touch "$file"
        echo "Created file: $file"
    fi
done

# Create placeholder file in the mlflow directory
if [ ! -f "mlflow/mlflow.yml" ]; then
    touch "mlflow/mlflow.yml"
    echo "Created file: mlflow/mlflow.yml"
fi

# Create GitHub Actions workflow file
if [ ! -f ".github/workflows/ci.yml" ]; then
    touch ".github/workflows/ci.yml"
    echo "Created file: .github/workflows/ci.yml"
fi

echo "Project structure created successfully."
