import json
from matplotlib.patches import Patch
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import re
import string
from collections import Counter
import nltk
import pickle
import os
import numpy as np
from datetime import datetime

# MLflow and logging imports
import mlflow
import mlflow.sklearn
from src.codestacker.train.logger import get_logger

# Get a logger for this module
logger = get_logger("train")

# Set MLflow tracking URI from environment variable
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(mlflow_tracking_uri)
logger.info(f"MLflow tracking URI: {mlflow_tracking_uri}")

# Download NLTK resources if not already downloaded
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    logger.info("Downloading NLTK punkt_tab...")
    nltk.download("punkt_tab")
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    logger.info("Downloading NLTK stopwords...")
    nltk.download("stopwords")
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    logger.info("Downloading NLTK wordnet...")
    nltk.download("wordnet")

# 1. Load Data
logger.info("Loading data...")
data = pd.read_csv("Competition_Dataset.csv")

# Log column names
logger.debug(f"Available columns in dataset: {data.columns.tolist()}")

# Split data into train and test sets
train, test = train_test_split(data, test_size=0.2, stratify=data["Category"])
logger.info(f"Data split: {len(train)} training samples, {len(test)} test samples")


# 2. Text preprocessing functions
def preprocess_text(text):
    """Cleans and preprocesses text for NLP features"""
    if not isinstance(text, str):
        return ""

    # Convert to lowercase and remove punctuation
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", " ", text)

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_nlp_features(df, text_column, max_features=200, n_components=50):
    """Extracts NLP features from a text column"""
    # Ensure text column exists
    if text_column not in df.columns:
        logger.warning(f"{text_column} column not found in DataFrame")
        return df, None, None, None

    # Fill NaN values
    df[text_column] = df[text_column].fillna("")

    # Preprocess text
    logger.info(f"Preprocessing {text_column} text...")
    df[f"{text_column}_processed"] = df[text_column].apply(preprocess_text)

    # Create TF-IDF vectorizer
    logger.info("Creating TF-IDF features...")
    tfidf = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),  # Unigrams and bigrams
    )

    # Transform text to TF-IDF features
    tfidf_matrix = tfidf.fit_transform(df[f"{text_column}_processed"])

    # Use TruncatedSVD to reduce dimensionality (similar to PCA for sparse matrices)
    logger.info(f"Reducing dimensionality to {n_components} components...")
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    tfidf_svd = svd.fit_transform(tfidf_matrix)

    # Create DataFrame with reduced features
    tfidf_df = pd.DataFrame(
        tfidf_svd, columns=[f"tfidf_{text_column}_{i}" for i in range(n_components)]
    )

    # Add TF-IDF features to original DataFrame
    for col in tfidf_df.columns:
        df[col] = tfidf_df[col]

    # Get the most important terms for each component
    feature_names = tfidf.get_feature_names_out()
    logger.info("Top terms in each component:")
    for i in range(
        min(5, n_components)
    ):  # Log only first 5 components to avoid log spam
        top_terms_idx = svd.components_[i].argsort()[-10:][::-1]
        top_terms = [feature_names[idx] for idx in top_terms_idx]
        logger.info(f"Component {i}: {', '.join(top_terms)}")

    # Count word frequencies for analysis
    logger.info("Analyzing word frequencies...")
    all_text = " ".join(df[f"{text_column}_processed"])
    words = all_text.split()
    word_freq = Counter(words).most_common(20)  # Reduced from 50 to 20 for logs
    logger.info("Top 20 words in descriptions:")
    for word, count in word_freq:
        logger.info(f"{word}: {count}")

    return df, tfidf, svd, feature_names


# 3. Feature engineering with spatial features


def engineer_advanced_spatial_features(df, training=True, model_dir=None):
    """
    Engineer advanced spatial features from the original dataset
    Args:
        df: DataFrame with at least Dates, Longitude (X), Latitude (Y), and PdDistrict columns
        training: Boolean indicating if this is for training (True) or inference (False)
        model_dir: Directory to save/load mappings (if None, uses model_artifacts/latest)

    Returns:
        DataFrame with additional spatial features
    """
    # Create a copy to avoid modifying the original
    data = df.copy()

    # Set default model directory if none provided
    if model_dir is None:
        model_dir = os.path.join("model_artifacts", "latest")

    # Ensure datetime format
    if "Dates" in data.columns:
        data["Dates"] = pd.to_datetime(data["Dates"])

    # 1. Create spatial grid
    # Divide CityX into a grid for better spatial granularity
    min_lon, max_lon = -122.51, -122.36
    min_lat, max_lat = 37.71, 37.83

    # Create grid cells (5x5 grid = 25 cells)
    lon_step = (max_lon - min_lon) / 5
    lat_step = (max_lat - min_lat) / 5

    # Assign grid cell ID based on location
    def assign_grid_cell(row):
        if "Longitude (X)" in data.columns and "Latitude (Y)" in data.columns:
            lon = row["Longitude (X)"]
            lat = row["Latitude (Y)"]
        elif "lon" in data.columns and "lat" in data.columns:
            lon = row["lon"]
            lat = row["lat"]
        else:
            return -1  # No coordinates

        # Calculate grid indices
        lon_idx = min(int((lon - min_lon) / lon_step), 4)
        lat_idx = min(int((lat - min_lat) / lat_step), 4)

        # Convert to 1D cell ID
        cell_id = lat_idx * 5 + lon_idx
        return cell_id

    data["GridCell"] = data.apply(assign_grid_cell, axis=1)

    # 2. Crime density features
    if training:
        # For training: calculate crime density from the entire dataset
        # Group by grid cell and count crimes
        grid_counts = data.groupby("GridCell").size().reset_index(name="TotalCrimes")

        # Create a mapping from grid cell to crime count
        grid_to_crimes = dict(zip(grid_counts["GridCell"], grid_counts["TotalCrimes"]))

        # Calculate crime density for each grid cell
        # (normalize by dividing by the maximum count)
        max_count = max(grid_to_crimes.values()) if grid_to_crimes else 1
        grid_to_density = {k: v / max_count for k, v in grid_to_crimes.items()}

        # Add crime density feature
        data["CrimeDensity"] = data["GridCell"].map(grid_to_density).fillna(0)

        # Store these mappings for inference
        density_mappings = {"grid_to_density": grid_to_density}

        # We'll also create category-specific density maps
        if "Category" in data.columns:
            category_density = {}
            for category in data["Category"].unique():
                category_data = data[data["Category"] == category]
                grid_counts = (
                    category_data.groupby("GridCell")
                    .size()
                    .reset_index(name="CategoryCrimes")
                )
                grid_to_cat_crimes = dict(
                    zip(grid_counts["GridCell"], grid_counts["CategoryCrimes"])
                )
                # Normalize
                max_cat_count = (
                    max(grid_to_cat_crimes.values()) if grid_to_cat_crimes else 1
                )
                grid_to_cat_density = {
                    k: v / max_cat_count for k, v in grid_to_cat_crimes.items()
                }
                category_density[category] = grid_to_cat_density

                # Add as a feature
                col_name = f'Density_{category.replace("/", "_").replace(" ", "_")}'
                data[col_name] = data["GridCell"].map(grid_to_cat_density).fillna(0)

            density_mappings["category_density"] = category_density

        # Save mappings for inference
        density_path = os.path.join(model_dir, "density_mappings.pkl")
        os.makedirs(os.path.dirname(density_path), exist_ok=True)
        with open(density_path, "wb") as f:
            pickle.dump(density_mappings, f)
        logger.info(f"Density mappings saved to {density_path}")
    else:
        # For inference: load pre-calculated density mappings
        try:
            density_path = os.path.join(model_dir, "density_mappings.pkl")
            with open(density_path, "rb") as f:
                density_mappings = pickle.load(f)

            # Apply crime density
            data["CrimeDensity"] = (
                data["GridCell"].map(density_mappings["grid_to_density"]).fillna(0)
            )

            # Apply category-specific densities
            if "category_density" in density_mappings:
                for category, grid_to_cat_density in density_mappings[
                    "category_density"
                ].items():
                    col_name = f'Density_{category.replace("/", "_").replace(" ", "_")}'
                    data[col_name] = data["GridCell"].map(grid_to_cat_density).fillna(0)
        except FileNotFoundError:
            logger.warning(f"Could not load density mappings from {density_path}")
            # If file doesn't exist, use placeholder values
            data["CrimeDensity"] = 0.1

    # 3. Time-space interaction features
    # Define time periods
    def get_time_period(hour):
        if 5 <= hour < 12:
            return "Morning"  # Morning
        elif 12 <= hour < 17:
            return "Afternoon"  # Afternoon
        elif 17 <= hour < 21:
            return "Evening"  # Evening
        else:
            return "Night"  # Night

    if "Dates" in data.columns:
        # Extract hour
        data["Hour"] = data["Dates"].dt.hour
        data["TimePeriod"] = data["Hour"].apply(get_time_period)
        data["IsWeekend"] = data["Dates"].dt.weekday >= 5

        # Create time-space interaction features
        if training:
            # Calculate risk scores for different time periods and locations
            time_space_risk = {}
            weekend_loc_risk = {}

            # Group by district and time period
            for district in data["PdDistrict"].unique():
                district_data = data[data["PdDistrict"] == district]

                # Time period risk by district
                for period in ["Morning", "Afternoon", "Evening", "Night"]:
                    period_count = len(
                        district_data[district_data["TimePeriod"] == period]
                    )
                    total_count = len(district_data)
                    risk_score = period_count / total_count if total_count > 0 else 0
                    time_space_risk[(district, period)] = risk_score

                # Weekend vs weekday risk by district
                for is_weekend in [True, False]:
                    weekend_count = len(
                        district_data[district_data["IsWeekend"] == is_weekend]
                    )
                    total_count = len(district_data)
                    risk_score = weekend_count / total_count if total_count > 0 else 0
                    weekend_loc_risk[(district, is_weekend)] = risk_score

            # Save mappings for inference
            time_space_path = os.path.join(model_dir, "time_space_mappings.pkl")
            os.makedirs(os.path.dirname(time_space_path), exist_ok=True)
            with open(time_space_path, "wb") as f:
                pickle.dump(
                    {
                        "time_space_risk": time_space_risk,
                        "weekend_loc_risk": weekend_loc_risk,
                    },
                    f,
                )
            logger.info(f"Time-space mappings saved to {time_space_path}")
        else:
            # For inference: load pre-calculated risk scores
            try:
                time_space_path = os.path.join(model_dir, "time_space_mappings.pkl")
                with open(time_space_path, "rb") as f:
                    time_space_mappings = pickle.load(f)
                    time_space_risk = time_space_mappings["time_space_risk"]
                    weekend_loc_risk = time_space_mappings["weekend_loc_risk"]
            except FileNotFoundError:
                logger.warning(
                    f"Could not load time-space mappings from {time_space_path}"
                )
                # If file doesn't exist, use placeholder values
                time_space_risk = {}
                weekend_loc_risk = {}

        # Apply time-space risk scores
        def get_time_district_risk(row):
            key = (row["PdDistrict"], row["TimePeriod"])
            return time_space_risk.get(key, 0.1)

        def get_weekend_district_risk(row):
            key = (row["PdDistrict"], row["IsWeekend"])
            return weekend_loc_risk.get(key, 0.1)

        data["TimeLocationRisk"] = data.apply(get_time_district_risk, axis=1)
        data["WeekendLocationRisk"] = data.apply(get_weekend_district_risk, axis=1)

        # Create interaction features for grid cells too
        if training:
            # More granular time-space risk using grid cells
            grid_time_risk = {}
            grid_weekend_risk = {}

            for grid_cell in data["GridCell"].unique():
                if grid_cell != -1:  # Skip rows without coordinates
                    grid_data = data[data["GridCell"] == grid_cell]

                    # Time period risk by grid cell
                    for period in ["Morning", "Afternoon", "Evening", "Night"]:
                        period_count = len(grid_data[grid_data["TimePeriod"] == period])
                        total_count = len(grid_data)
                        risk_score = (
                            period_count / total_count if total_count > 0 else 0
                        )
                        grid_time_risk[(grid_cell, period)] = risk_score

                    # Weekend vs weekday risk by grid cell
                    for is_weekend in [True, False]:
                        weekend_count = len(
                            grid_data[grid_data["IsWeekend"] == is_weekend]
                        )
                        total_count = len(grid_data)
                        risk_score = (
                            weekend_count / total_count if total_count > 0 else 0
                        )
                        grid_weekend_risk[(grid_cell, is_weekend)] = risk_score

            # Save mappings for inference
            grid_time_path = os.path.join(model_dir, "grid_time_mappings.pkl")
            os.makedirs(os.path.dirname(grid_time_path), exist_ok=True)
            with open(grid_time_path, "wb") as f:
                pickle.dump(
                    {
                        "grid_time_risk": grid_time_risk,
                        "grid_weekend_risk": grid_weekend_risk,
                    },
                    f,
                )
            logger.info(f"Grid-time mappings saved to {grid_time_path}")
        else:
            # For inference: load pre-calculated grid risk scores
            try:
                grid_time_path = os.path.join(model_dir, "grid_time_mappings.pkl")
                with open(grid_time_path, "rb") as f:
                    grid_time_mappings = pickle.load(f)
                    grid_time_risk = grid_time_mappings["grid_time_risk"]
                    grid_weekend_risk = grid_time_mappings["grid_weekend_risk"]
            except FileNotFoundError:
                logger.warning(
                    f"Could not load grid-time mappings from {grid_time_path}"
                )
                # If file doesn't exist, use placeholder values
                grid_time_risk = {}
                grid_weekend_risk = {}

        # Apply grid time-space risk scores
        def get_grid_time_risk(row):
            key = (row["GridCell"], row["TimePeriod"])
            return grid_time_risk.get(key, 0.1)

        def get_grid_weekend_risk(row):
            key = (row["GridCell"], row["IsWeekend"])
            return grid_weekend_risk.get(key, 0.1)

        data["GridTimeRisk"] = data.apply(get_grid_time_risk, axis=1)
        data["GridWeekendRisk"] = data.apply(get_grid_weekend_risk, axis=1)

    return data


def process_data(df, tfidf=None, svd=None, include_nlp=True):
    """Process data with both standard and NLP features, plus enhanced spatial features"""
    # Create copy to avoid modifying original
    df_processed = df.copy()

    # Basic datetime conversion - check if Dates column exists
    if "Dates" in df_processed.columns:
        df_processed["Dates"] = pd.to_datetime(df_processed["Dates"])

        # Temporal features
        df_processed["Hour"] = df_processed["Dates"].dt.hour
        df_processed["Day"] = df_processed["Dates"].dt.day
        df_processed["Month"] = df_processed["Dates"].dt.month
        df_processed["DayOfWeek_Num"] = df_processed["Dates"].dt.weekday
        df_processed["WeekendFlag"] = df_processed["DayOfWeek_Num"].apply(
            lambda x: 1 if x >= 5 else 0
        )

        # Time periods
        def assign_time_period(hour):
            if 5 <= hour < 12:
                return 0  # Morning
            elif 12 <= hour < 17:
                return 1  # Afternoon
            elif 17 <= hour < 21:
                return 2  # Evening
            else:
                return 3  # Night

        df_processed["TimePeriod"] = df_processed["Hour"].apply(assign_time_period)
    else:
        logger.warning("'Dates' column not found, skipping temporal features")
        # Add placeholder columns with default values
        for col in [
            "Hour",
            "Day",
            "Month",
            "DayOfWeek_Num",
            "WeekendFlag",
            "TimePeriod",
        ]:
            df_processed[col] = 0

    # Spatial features - check if X and Y columns exist
    if (
        "Longitude (X)" in df_processed.columns
        and "Latitude (Y)" in df_processed.columns
    ):
        city_center_x, city_center_y = -122.4194, 37.7749  # CityX coordinates
        df_processed["DistanceFromCenter"] = np.sqrt(
            (df_processed["Longitude (X)"] - city_center_x) ** 2
            + (df_processed["Latitude (Y)"] - city_center_y) ** 2
        )

        # Instead of high-cardinality GeoCluster, create quadrant features
        df_processed["QuadrantNS"] = (
            df_processed["Latitude (Y)"] > city_center_y
        ).astype(int)
        df_processed["QuadrantEW"] = (
            df_processed["Longitude (X)"] > city_center_x
        ).astype(int)

        # Add enhanced spatial features
        logger.info("Adding enhanced spatial features...")
        df_processed = engineer_advanced_spatial_features(df_processed, training=True)

    else:
        logger.warning(
            "'Longitude (X)' and/or 'Latitude (Y)' columns not found, skipping spatial features"
        )
        # Add placeholder columns with default values
        df_processed["DistanceFromCenter"] = 0
        df_processed["QuadrantNS"] = 0
        df_processed["QuadrantEW"] = 0
        df_processed["CrimeDensity"] = 0.1
        df_processed["TimeLocationRisk"] = 0.1
        df_processed["WeekendLocationRisk"] = 0.1
        df_processed["GridTimeRisk"] = 0.1
        df_processed["GridWeekendRisk"] = 0.1

    # Categorical encoding - check if columns exist
    if "PdDistrict" in df_processed.columns:
        df_processed["PdDistrict"] = pd.Categorical(df_processed["PdDistrict"]).codes
    else:
        logger.warning("'PdDistrict' column not found, adding placeholder")
        df_processed["PdDistrict"] = 0

    if "DayOfWeek" in df_processed.columns:
        df_processed["DayOfWeek"] = pd.Categorical(df_processed["DayOfWeek"]).codes
    else:
        logger.warning("'DayOfWeek' column not found, adding placeholder")
        df_processed["DayOfWeek"] = 0

    # Add NLP features if we have the transformers
    if include_nlp and "Descript" in df_processed.columns:
        # Process any new text data with the fitted transformers
        if tfidf is not None and svd is not None:
            df_processed["Descript_processed"] = (
                df_processed["Descript"].fillna("").apply(preprocess_text)
            )
            tfidf_matrix = tfidf.transform(df_processed["Descript_processed"])
            tfidf_svd = svd.transform(tfidf_matrix)

            # Add TF-IDF SVD features
            for i in range(tfidf_svd.shape[1]):
                df_processed[f"tfidf_Descript_{i}"] = tfidf_svd[:, i]

    # Select features - now including our new spatial features
    features = [
        "Hour",
        "Day",
        "Month",
        "DayOfWeek_Num",
        "WeekendFlag",
        "TimePeriod",
        "DayOfWeek",
        "PdDistrict",
        "DistanceFromCenter",
        "QuadrantNS",
        "QuadrantEW",
        "CrimeDensity",
        "TimeLocationRisk",
        "WeekendLocationRisk",
        "GridTimeRisk",
        "GridWeekendRisk",
    ]

    # Add category-specific density columns if they exist
    category_density_cols = [
        col for col in df_processed.columns if col.startswith("Density_")
    ]
    features.extend(category_density_cols)

    # Add NLP features if present
    nlp_features = [col for col in df_processed.columns if col.startswith("tfidf_")]
    features.extend(nlp_features)

    # Filter to only columns that exist
    valid_features = [col for col in features if col in df_processed.columns]

    return df_processed[valid_features]


# 4. Main execution
def plot_feature_importance(model, feature_names):
    """Plot feature importance and return the figure"""
    importance = model.get_feature_importance()
    features = pd.DataFrame(
        {"Feature": feature_names, "Importance": importance}
    ).sort_values("Importance", ascending=False)

    # Highlight NLP features
    features["is_nlp"] = features["Feature"].str.startswith("tfidf_")

    fig = plt.figure(figsize=(12, 8))

    # Plot with color differentiation for NLP features
    colors = ["#1f77b4" if not is_nlp else "#ff7f0e" for is_nlp in features["is_nlp"]]

    plt.barh(range(len(features)), features["Importance"], align="center", color=colors)
    plt.yticks(range(len(features)), features["Feature"])
    plt.xlabel("Feature Importance")
    plt.title("Feature Importance (NLP features in orange)")

    # Add legend
    legend_elements = [
        Patch(facecolor="#1f77b4", label="Standard Features"),
        Patch(facecolor="#ff7f0e", label="NLP Features"),
    ]
    plt.legend(handles=legend_elements)

    plt.tight_layout()

    # Also log top NLP features separately
    nlp_features = features[features["is_nlp"]].head(10)
    if not nlp_features.empty:
        logger.info("Top 10 NLP Features:")
        for i, row in nlp_features.iterrows():
            logger.info(f"{row['Feature']}: {row['Importance']:.4f}")

    return fig


def main():
    # Start MLflow experiment
    experiment_name = "crime-classification"
    mlflow.set_experiment(experiment_name)

    # Start the MLflow run
    with mlflow.start_run() as run:
        try:
            run_id = run.info.run_id
            logger.info(f"Starting MLflow run: {run_id}")

            # Extract NLP features from training data
            logger.info("Starting NLP feature extraction...")
            train_nlp, tfidf_vectorizer, svd_transformer, feature_names = (
                extract_nlp_features(
                    train, "Descript", max_features=200, n_components=50
                )
            )

            # Log NLP parameters
            mlflow.log_params(
                {
                    "nlp_max_features": 200,
                    "nlp_n_components": 50,
                }
            )

            # Process datasets with NLP features
            logger.info("Processing training data...")
            X_full = process_data(train_nlp, tfidf_vectorizer, svd_transformer)
            y_full = train_nlp["Category"]

            logger.info("Processing test data...")
            X_test = process_data(test, tfidf_vectorizer, svd_transformer)

            # Split training data
            X_train, X_val, y_train, y_val = train_test_split(
                X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
            )
            logger.info(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")

            # Define categorical features
            categorical_features = [
                "DayOfWeek",
                "PdDistrict",
                "TimePeriod",
                "QuadrantNS",
                "QuadrantEW",
            ]

            # Train model
            logger.info("Training model with NLP features...")
            train_pool = Pool(X_train, y_train, cat_features=categorical_features)
            val_pool = Pool(X_val, y_val, cat_features=categorical_features)

            # Log model parameters
            model_params = {
                "iterations": 1000,
                "learning_rate": 0.05,
                "depth": 8,
                "l2_leaf_reg": 3,
                "loss_function": "MultiClass",
                "eval_metric": "TotalF1",
                "random_seed": 42,
            }

            mlflow.log_params(model_params)

            # Check if GPU is available, otherwise use CPU
            try:
                model = CatBoostClassifier(
                    iterations=model_params["iterations"],
                    learning_rate=model_params["learning_rate"],
                    depth=model_params["depth"],
                    l2_leaf_reg=model_params["l2_leaf_reg"],
                    loss_function=model_params["loss_function"],
                    eval_metric=model_params["eval_metric"],
                    random_seed=model_params["random_seed"],
                    od_type="Iter",
                    od_wait=50,
                    verbose=100,
                    task_type="GPU",
                )
                model.fit(train_pool, eval_set=val_pool, use_best_model=True)
                logger.info("Model training completed using GPU")
            except Exception as e:
                logger.warning(f"GPU training failed with error: {e}")
                logger.info("Falling back to CPU training...")
                model = CatBoostClassifier(
                    iterations=500,  # Reduced for CPU
                    learning_rate=model_params["learning_rate"],
                    depth=model_params["depth"],
                    l2_leaf_reg=model_params["l2_leaf_reg"],
                    loss_function=model_params["loss_function"],
                    eval_metric=model_params["eval_metric"],
                    random_seed=model_params["random_seed"],
                    od_type="Iter",
                    od_wait=50,
                    verbose=100,
                    task_type="CPU",
                )
                model.fit(train_pool, eval_set=val_pool, use_best_model=True)
                mlflow.log_param("iterations", 500)  # Log the adjusted iterations
                logger.info("Model training completed using CPU")

            # Log training curves
            for metric in ["learn", "validation"]:
                if metric in model.evals_result_:
                    for metric_name, values in model.evals_result_[metric].items():
                        for i, value in enumerate(values):
                            mlflow.log_metric(f"{metric}_{metric_name}", value, step=i)

            # Evaluate model
            logger.info("Evaluating model...")
            val_predictions = model.predict(X_val)

            # Calculate metrics
            accuracy = accuracy_score(y_val, val_predictions)
            logger.info(f"Validation Accuracy: {accuracy:.4f}")

            macro_f1 = f1_score(y_val, val_predictions, average="macro")
            logger.info(f"Validation Macro F1 Score: {macro_f1:.4f}")

            weighted_f1 = f1_score(y_val, val_predictions, average="weighted")
            logger.info(f"Validation Weighted F1 Score: {weighted_f1:.4f}")

            # Log metrics to MLflow
            mlflow.log_metrics(
                {
                    "accuracy": float(accuracy),
                    "macro_f1": float(macro_f1),
                    "weighted_f1": float(weighted_f1),
                }
            )

            # Generate and log feature importance
            logger.info("Analyzing feature importance...")
            fig = plot_feature_importance(model, X_train.columns)
            fig_path = "plots/feature_importance.png"
            os.makedirs(os.path.dirname(fig_path), exist_ok=True)
            plt.savefig(fig_path)
            mlflow.log_artifact(fig_path, artifact_path="plots")

            # Infer the model signature using a sample from X_train and the model's predictions
            input_example = X_train.head(5)
            signature = mlflow.models.infer_signature(X_train, model.predict(X_train))

            # Set model tags for additional metadata
            tags = {
                "model_type": "CatBoostClassifier",
                "framework": "catboost",
                "dataset": "Competition_Dataset.csv",
                "experiment": experiment_name,
                "author": "Mike Hanna",
            }
            mlflow.set_tags(tags)

            # Log the model with signature and input example
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name="crime_classification_model",
                input_example=input_example,
                signature=signature,
            )

            # Save and log preprocessing objects (TF-IDF and SVD)
            preprocessors_dir = "preprocessors"
            os.makedirs(preprocessors_dir, exist_ok=True)
            tfidf_path = os.path.join(preprocessors_dir, "tfidf_vectorizer.pkl")
            svd_path = os.path.join(preprocessors_dir, "svd_transformer.pkl")
            with open(tfidf_path, "wb") as f:
                pickle.dump(tfidf_vectorizer, f)
            with open(svd_path, "wb") as f:
                pickle.dump(svd_transformer, f)
            mlflow.log_artifact(tfidf_path, artifact_path="preprocessors")
            mlflow.log_artifact(svd_path, artifact_path="preprocessors")

            # Save dataset snapshots for reproducibility
            datasets_dir = "datasets"
            os.makedirs(datasets_dir, exist_ok=True)
            train_dataset_path = os.path.join(datasets_dir, "dataset_train.csv")
            test_dataset_path = os.path.join(datasets_dir, "dataset_test.csv")
            train.to_csv(train_dataset_path, index=False)
            test.to_csv(test_dataset_path, index=False)
            mlflow.log_artifact(train_dataset_path, artifact_path="datasets")
            mlflow.log_artifact(test_dataset_path, artifact_path="datasets")

            # Prepare metadata for the run
            feature_names_list = (
                (
                    feature_names.tolist()
                    if isinstance(feature_names, np.ndarray)
                    else list(feature_names)
                )
                if feature_names is not None
                else None
            )

            svd_components = (
                svd_transformer.components_.tolist()
                if hasattr(svd_transformer, "components_")
                else None
            )

            metadata = {
                "feature_names": X_train.columns.tolist(),
                "categorical_features": categorical_features,
                "svd_components": svd_components,
                "tfidf_feature_names": feature_names_list,
                "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "metrics": {
                    "accuracy": float(accuracy),
                    "macro_f1": float(macro_f1),
                    "weighted_f1": float(weighted_f1),
                },
                "mlflow_run_id": run_id,
            }

            metadata_path = "metadata/model_metadata.json"
            os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4, default=str)
            mlflow.log_artifact(metadata_path, artifact_path="metadata")

            logger.info(f"MLflow run completed: {run_id}")
            logger.info("NLP feature analysis complete!")

        except Exception as e:
            logger.error(f"Error in training pipeline: {e}", exc_info=True)
            mlflow.log_param("error", str(e))
            raise


if __name__ == "__main__":
    main()
