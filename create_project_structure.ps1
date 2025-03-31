# Define the directories to be created
$directories = @(
    "src\codestacker\app",
    "src\codestacker\training",
    "src\codestacker\common",
    "docker",
    "mlflow",
    ".github\workflows"
)

foreach ($dir in $directories) {
    if (!(Test-Path -Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
        Write-Host "Created directory: $dir"
    }
}

# Create root-level files if they don't exist
$rootFiles = @("pyproject.toml", "poetry.toml", "README.md", "LICENSE")
foreach ($file in $rootFiles) {
    if (!(Test-Path -Path $file)) {
        New-Item -ItemType File -Path $file -Force | Out-Null
        Write-Host "Created file: $file"
    }
}

# Create placeholder files in the src/codestacker/app directory
$appFiles = @("src\codestacker\app\__init__.py", "src\codestacker\app\main.py", "src\codestacker\app\utils.py")
foreach ($file in $appFiles) {
    if (!(Test-Path -Path $file)) {
        New-Item -ItemType File -Path $file -Force | Out-Null
        Write-Host "Created file: $file"
    }
}

# Create placeholder files in the src/codestacker/training directory
$trainingFiles = @("src\codestacker\training\__init__.py", "src\codestacker\training\train.py", "src\codestacker\training\preprocessing.py")
foreach ($file in $trainingFiles) {
    if (!(Test-Path -Path $file)) {
        New-Item -ItemType File -Path $file -Force | Out-Null
        Write-Host "Created file: $file"
    }
}

# Create placeholder files in the src/codestacker/common directory
$commonFiles = @("src\codestacker\common\__init__.py", "src\codestacker\common\logger.py", "src\codestacker\common\utils.py")
foreach ($file in $commonFiles) {
    if (!(Test-Path -Path $file)) {
        New-Item -ItemType File -Path $file -Force | Out-Null
        Write-Host "Created file: $file"
    }
}

# Create placeholder files in the docker directory
$dockerFiles = @("docker\Dockerfile.app", "docker\Dockerfile.train", "docker\docker-compose.yml")
foreach ($file in $dockerFiles) {
    if (!(Test-Path -Path $file)) {
        New-Item -ItemType File -Path $file -Force | Out-Null
        Write-Host "Created file: $file"
    }
}

# Create placeholder file in the mlflow directory
if (!(Test-Path -Path "mlflow\mlflow.yml")) {
    New-Item -ItemType File -Path "mlflow\mlflow.yml" -Force | Out-Null
    Write-Host "Created file: mlflow\mlflow.yml"
}

# Create GitHub Actions workflow file
if (!(Test-Path -Path ".github\workflows\ci.yml")) {
    New-Item -ItemType File -Path ".github\workflows\ci.yml" -Force | Out-Null
    Write-Host "Created file: .github\workflows\ci.yml"
}

Write-Host "Project structure created successfully."
