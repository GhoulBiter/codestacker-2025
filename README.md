# Crime Classification System

A comprehensive system for classifying crime reports using NLP and machine learning with MLOps integration.

## Features

- **Machine Learning Model**: Uses CatBoost classifier with NLP features to predict crime categories
- **Interactive Dashboard**: Built with Streamlit for data visualization and prediction
- **MLOps Integration**: MLflow tracking and model registry
- **Docker Support**: Containerized deployment for both training and serving
- **CI/CD Pipeline**: GitHub Actions for automated testing and deployment
- **PDF Processing**: Extract information from PDF police reports
- **Gemini AI Integration**: Optional Gemini API integration for standardized description formats

## Project Structure

```text
crime-classification-system/
├── .github/
│   └── workflows/            # GitHub Actions CI/CD
├── docker/                   # Docker configuration
│   ├── Dockerfile.app        # Streamlit app Dockerfile
│   ├── Dockerfile.train      # Training Dockerfile
│   ├── Dockerfile.mlflow     # MLflow server Dockerfile
│   └── docker-compose.yml    # Docker Compose setup
├── mlflow/                   # MLflow configuration
├── src/
│   └── codestacker/
│       ├── app/              # Streamlit application
│       ├── training/         # Model training code
│       └── common/           # Shared utilities
├── data/                     # Data files
├── model_artifacts/          # Saved models and artifacts
├── logs/                     # Application logs
├── output/                   # Training outputs and visualizations
├── tests/                    # Test suite
├── .env                      # Environment variables (not in git)
├── poetry.toml               # Poetry configuration
├── pyproject.toml            # Project dependencies
└── README.md                 # Project documentation
```

## Setup and Installation

### Prerequisites

- Python 3.10+
- Poetry for dependency management
- Docker and Docker Compose for containerized deployment

### Local Development Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/crime-classification-system.git
   cd crime-classification-system
   ```

2. Install dependencies with Poetry:

   ```bash
   poetry install
   ```

3. Set up environment variables:

   ```bash
   cp .env.example .env
   # Edit .env file with your configuration
   ```

4. Run the training process:

   ```bash
   poetry run python -m src.codestacker.training.train --data_path data/Competition_Dataset.csv --visualize
   ```

5. Start the Streamlit app:

   ```bash
   poetry run streamlit run src/codestacker/app/main.py
   ```

### Docker Setup

1. Build and start the services:

   ```bash
   docker-compose -f docker/docker-compose.yml up -d
   ```

2. Access the services:
   - Streamlit app: <http://localhost:8501>
   - MLflow UI: <http://localhost:5000>

## Model Training

The training pipeline includes:

1. Text preprocessing and NLP feature extraction
2. Spatial feature engineering
3. Temporal feature engineering
4. Model training with CatBoost classifier
5. Model evaluation and visualization
6. Model artifacts saved for inference

```bash
# Basic training
python -m src.codestacker.training.train --data_path data/Competition_Dataset.csv

# Training with MLflow tracking and visualization
python -m src.codestacker.training.train --data_path data/Competition_Dataset.csv --use_mlflow --visualize

# Using GPU acceleration
python -m src.codestacker.training.train --data_path data/Competition_Dataset.csv --use_gpu
```

## Deployment

### VPS Deployment

1. Provision a VPS with Docker and Docker Compose installed
2. Set up required GitHub Secrets for CI/CD
3. Push to main branch to trigger deployment
4. Configure appropriate security (firewall, SSL, etc.)

### CI/CD Pipeline

The GitHub Actions workflow includes:

1. Testing: Run linting and tests on each push/PR
2. Building: Build Docker images on pushes to main/develop
3. Deployment: Deploy to VPS on pushes to main branch

## Optional Integrations

### Gemini AI

For enhanced text processing, set up a Gemini API key:

1. Obtain an API key from Google AI Studio
2. Add to `.env` file: `GEMINI_API_KEY=your_api_key_here`
3. Enable "Use Gemini for Description Processing" in the app

### MongoDB

For advanced logging and data storage:

1. Set up MongoDB credentials in the `.env` file
2. The app will automatically use MongoDB for logging and storing results

## License

This project is licensed under the MIT License - see the LICENSE file for details.
