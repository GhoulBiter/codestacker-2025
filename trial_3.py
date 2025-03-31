import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
import os
import numpy as np
from datetime import datetime

# Download NLTK resources if not already downloaded
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

# 1. Load Data
print("Loading data...")
data = pd.read_csv("Competition_Dataset.csv")

# Print column names to debug
print("Available columns in dataset:", data.columns.tolist())

# Split data into train and test sets
train, test = train_test_split(data, test_size=0.2, stratify=data["Category"])


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
        print(f"Warning: {text_column} column not found in DataFrame")
        return df, None, None, None

    # Fill NaN values
    df[text_column] = df[text_column].fillna("")

    # Preprocess text
    print(f"Preprocessing {text_column} text...")
    df[f"{text_column}_processed"] = df[text_column].apply(preprocess_text)

    # Create TF-IDF vectorizer
    print("Creating TF-IDF features...")
    tfidf = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),  # Unigrams and bigrams
    )

    # Transform text to TF-IDF features
    tfidf_matrix = tfidf.fit_transform(df[f"{text_column}_processed"])

    # Use TruncatedSVD to reduce dimensionality (similar to PCA for sparse matrices)
    print(f"Reducing dimensionality to {n_components} components...")
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
    print("\nTop terms in each component:")
    for i in range(n_components):
        top_terms_idx = svd.components_[i].argsort()[-10:][::-1]
        top_terms = [feature_names[idx] for idx in top_terms_idx]
        print(f"Component {i}: {', '.join(top_terms)}")

    # Count word frequencies for analysis
    print("\nAnalyzing word frequencies...")
    all_text = " ".join(df[f"{text_column}_processed"])
    words = all_text.split()
    word_freq = Counter(words).most_common(50)
    print("Top 50 words in descriptions:")
    for word, count in word_freq:
        print(f"{word}: {count}")

    return df, tfidf, svd, feature_names


# 3. Feature engineering with text features


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
    # Divide SF into a grid for better spatial granularity
    # San Francisco boundaries
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
        print(f"Density mappings saved to {density_path}")
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
            print(f"Warning: Could not load density mappings from {density_path}")
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
            print(f"Time-space mappings saved to {time_space_path}")
        else:
            # For inference: load pre-calculated risk scores
            try:
                time_space_path = os.path.join(model_dir, "time_space_mappings.pkl")
                with open(time_space_path, "rb") as f:
                    time_space_mappings = pickle.load(f)
                    time_space_risk = time_space_mappings["time_space_risk"]
                    weekend_loc_risk = time_space_mappings["weekend_loc_risk"]
            except FileNotFoundError:
                print(
                    f"Warning: Could not load time-space mappings from {time_space_path}"
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
            print(f"Grid-time mappings saved to {grid_time_path}")
        else:
            # For inference: load pre-calculated grid risk scores
            try:
                grid_time_path = os.path.join(model_dir, "grid_time_mappings.pkl")
                with open(grid_time_path, "rb") as f:
                    grid_time_mappings = pickle.load(f)
                    grid_time_risk = grid_time_mappings["grid_time_risk"]
                    grid_weekend_risk = grid_time_mappings["grid_weekend_risk"]
            except FileNotFoundError:
                print(
                    f"Warning: Could not load grid-time mappings from {grid_time_path}"
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
        print("Warning: 'Dates' column not found, skipping temporal features")
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
        city_center_x, city_center_y = -122.4194, 37.7749  # SF coordinates
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
        print("Adding enhanced spatial features...")
        df_processed = engineer_advanced_spatial_features(df_processed, training=True)

    else:
        print(
            "Warning: 'Longitude (X)' and/or 'Latitude (Y)' columns not found, skipping spatial features"
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
        print("Warning: 'PdDistrict' column not found, adding placeholder")
        df_processed["PdDistrict"] = 0

    if "DayOfWeek" in df_processed.columns:
        df_processed["DayOfWeek"] = pd.Categorical(df_processed["DayOfWeek"]).codes
    else:
        print("Warning: 'DayOfWeek' column not found, adding placeholder")
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
print("Starting NLP feature extraction...")
# Extract NLP features from training data
train, tfidf_vectorizer, svd_transformer, feature_names = extract_nlp_features(
    train, "Descript", max_features=200, n_components=50
)

# Process datasets with NLP features
print("Processing training data...")
X_full = process_data(train, tfidf_vectorizer, svd_transformer)
y_full = train["Category"]

print("Processing test data...")
X_test = process_data(test, tfidf_vectorizer, svd_transformer)

# 5. Split training data
X_train, X_val, y_train, y_val = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
)


# 6. Visualize NLP features
def plot_nlp_components(X, y, n_components=6):
    """Visualize how NLP components separate classes"""
    # Get the most frequent classes for clearer visualization
    top_classes = y.value_counts().head(5).index
    mask = y.isin(top_classes)
    X_subset = X[mask]
    y_subset = y[mask]

    nlp_cols = [col for col in X.columns if col.startswith("tfidf_")][:n_components]

    if len(nlp_cols) < 2:
        print("Not enough NLP components to create visualization")
        return

    # Create pairwise scatter plots of top components
    plt.figure(figsize=(15, 10))
    for i in range(min(3, len(nlp_cols))):
        for j in range(i + 1, min(i + 2, len(nlp_cols))):
            if i * 2 + j - i <= 5:  # Limit to 6 subplots
                plt.subplot(2, 3, i * 2 + j - i)
                for crime_type in top_classes:
                    mask_class = y_subset == crime_type
                    plt.scatter(
                        X_subset.loc[mask_class, nlp_cols[i]],
                        X_subset.loc[mask_class, nlp_cols[j]],
                        alpha=0.5,
                        label=crime_type,
                    )
                plt.xlabel(nlp_cols[i])
                plt.ylabel(nlp_cols[j])
                plt.legend(loc="best")
                plt.title(f"NLP Components {i} vs {j}")

    plt.tight_layout()
    plt.show()


# Visualize NLP feature distributions
print("Visualizing NLP components...")
try:
    plot_nlp_components(X_train, y_train)
except Exception as e:
    print(f"Error in visualization: {e}")

# 7. Define categorical features
categorical_features = [
    "DayOfWeek",
    "PdDistrict",
    "TimePeriod",
    "QuadrantNS",
    "QuadrantEW",
]

# 8. Train model
print("Training model with NLP features...")
train_pool = Pool(X_train, y_train, cat_features=categorical_features)
val_pool = Pool(X_val, y_val, cat_features=categorical_features)

# Check if GPU is available, otherwise use CPU
try:
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3,
        loss_function="MultiClass",
        eval_metric="TotalF1",
        random_seed=42,
        od_type="Iter",
        od_wait=50,
        verbose=100,
        task_type="GPU",
    )
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
except Exception as e:
    print(f"GPU training failed with error: {e}")
    print("Falling back to CPU training...")
    model = CatBoostClassifier(
        iterations=500,  # Reduced for CPU
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3,
        loss_function="MultiClass",
        eval_metric="TotalF1",
        random_seed=42,
        od_type="Iter",
        od_wait=50,
        verbose=100,
        task_type="CPU",
    )
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

# 9. Evaluate model
print("Evaluating model...")
val_predictions = model.predict(X_val)

# Calculate metrics
accuracy = accuracy_score(y_val, val_predictions)
print(f"Validation Accuracy: {accuracy:.4f}")

macro_f1 = f1_score(y_val, val_predictions, average="macro")
print(f"Validation Macro F1 Score: {macro_f1:.4f}")

weighted_f1 = f1_score(y_val, val_predictions, average="weighted")
print(f"Validation Weighted F1 Score: {weighted_f1:.4f}")


# 10. Feature importance
def plot_feature_importance(model, feature_names):
    importance = model.get_feature_importance()
    features = pd.DataFrame(
        {"Feature": feature_names, "Importance": importance}
    ).sort_values("Importance", ascending=False)

    # Highlight NLP features
    features["is_nlp"] = features["Feature"].str.startswith("tfidf_")

    plt.figure(figsize=(12, 8))

    # Plot with color differentiation for NLP features
    colors = ["#1f77b4" if not is_nlp else "#ff7f0e" for is_nlp in features["is_nlp"]]

    plt.barh(range(len(features)), features["Importance"], align="center", color=colors)
    plt.yticks(range(len(features)), features["Feature"])
    plt.xlabel("Feature Importance")
    plt.title("Feature Importance (NLP features in orange)")

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#1f77b4", label="Standard Features"),
        Patch(facecolor="#ff7f0e", label="NLP Features"),
    ]
    plt.legend(handles=legend_elements)

    plt.tight_layout()
    plt.show()

    # Also print top NLP features separately
    nlp_features = features[features["is_nlp"]].head(10)
    if not nlp_features.empty:
        print("\nTop 10 NLP Features:")
        for i, row in nlp_features.iterrows():
            print(f"{row['Feature']}: {row['Importance']:.4f}")


print("\nAnalyzing feature importance...")
plot_feature_importance(model, X_train.columns)

print("\nNLP feature analysis complete!")

# 11. Save the engineered datasets to CSV
# print("\nSaving engineered datasets to CSV files...")

# Save training data (with features and target)
# train_output = X_full.copy()
# train_output["Category"] = y_full
# train_output.to_csv("engineered_train_data.csv", index=False)
# print(
#     f"Training data with {train_output.shape[1]} features saved to engineered_train_data.csv"
# )

# # Save test data
# test_output = X_test.copy()
# if "Category" in test.columns:  # Only add Category column if it exists in the test data
#     test_output["Category"] = test["Category"]
#     print(
#         f"Test data with {test_output.shape[1]} features and labels saved to engineered_test_data.csv"
#     )
# else:
#     print(
#         f"Test data with {test_output.shape[1]} features saved to engineered_test_data.csv"
#     )
# test_output.to_csv("engineered_test_data.csv", index=False)

# print("Dataset saving complete!")

# 12. Save model and preprocessing objects
# Create a directory for saving model artifacts if it doesn't exist
os.makedirs("model_artifacts", exist_ok=True)


def save_model_artifacts(
    model, tfidf_vectorizer, svd_transformer, metadata, feature_names
):
    """Save model artifacts in a timestamped directory"""
    # Create a timestamp for the run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a base directory for model artifacts if it doesn't exist
    base_dir = "model_artifacts"
    os.makedirs(base_dir, exist_ok=True)

    # Create a subdirectory with the timestamp
    model_dir = os.path.join(base_dir, timestamp)
    os.makedirs(model_dir, exist_ok=True)

    print(f"\nSaving model and preprocessing objects to {model_dir}...")

    # Save the CatBoost model
    model_path = os.path.join(model_dir, "catboost_model.cbm")
    model.save_model(model_path)
    print(f"Model saved to {model_path}")

    # Save the TF-IDF vectorizer
    tfidf_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
    with open(tfidf_path, "wb") as f:
        pickle.dump(tfidf_vectorizer, f)
    print(f"TF-IDF vectorizer saved to {tfidf_path}")

    # Save the SVD transformer
    svd_path = os.path.join(model_dir, "svd_transformer.pkl")
    with open(svd_path, "wb") as f:
        pickle.dump(svd_transformer, f)
    print(f"SVD transformer saved to {svd_path}")

    # Convert NumPy arrays to lists for safe serialization
    feature_names_list = None
    if feature_names is not None:
        feature_names_list = (
            feature_names.tolist()
            if isinstance(feature_names, np.ndarray)
            else list(feature_names)
        )

    svd_components = None
    if svd_transformer is not None and hasattr(svd_transformer, "components_"):
        svd_components = svd_transformer.components_.tolist()

    # Save metadata
    metadata_path = os.path.join(model_dir, "metadata.pkl")
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    print(f"Model metadata saved to {metadata_path}")

    # Save density mappings if they exist
    density_mappings_path = os.path.join(base_dir, "density_mappings.pkl")
    if os.path.exists(density_mappings_path):
        # Copy to the timestamped directory
        with open(density_mappings_path, "rb") as f:
            density_mappings = pickle.load(f)

        new_path = os.path.join(model_dir, "density_mappings.pkl")
        with open(new_path, "wb") as f:
            pickle.dump(density_mappings, f)
        print(f"Density mappings copied to {new_path}")

    # Save time-space mappings if they exist
    time_space_path = os.path.join(base_dir, "time_space_mappings.pkl")
    if os.path.exists(time_space_path):
        # Copy to the timestamped directory
        with open(time_space_path, "rb") as f:
            time_space_mappings = pickle.load(f)

        new_path = os.path.join(model_dir, "time_space_mappings.pkl")
        with open(new_path, "wb") as f:
            pickle.dump(time_space_mappings, f)
        print(f"Time-space mappings copied to {new_path}")

    # Save grid-time mappings if they exist
    grid_time_path = os.path.join(base_dir, "grid_time_mappings.pkl")
    if os.path.exists(grid_time_path):
        # Copy to the timestamped directory
        with open(grid_time_path, "rb") as f:
            grid_time_mappings = pickle.load(f)

        new_path = os.path.join(model_dir, "grid_time_mappings.pkl")
        with open(new_path, "wb") as f:
            pickle.dump(grid_time_mappings, f)
        print(f"Grid-time mappings copied to {new_path}")

    # Create a simple readme file with model information
    readme_path = os.path.join(model_dir, "README.txt")
    with open(readme_path, "w") as f:
        f.write(f"Model trained on: {timestamp}\n")
        f.write(f"Number of features: {len(metadata['feature_names'])}\n")
        f.write(
            f"Categorical features: {', '.join(metadata['categorical_features'])}\n"
        )

        # Add enhanced spatial features if present
        spatial_features = [
            feat
            for feat in metadata["feature_names"]
            if any(x in feat for x in ["Density", "Risk", "Grid"])
        ]
        if spatial_features:
            f.write(f"\nEnhanced spatial features: {len(spatial_features)}\n")
            f.write("- " + "\n- ".join(spatial_features))

        # Add performance metrics if present
        if "metrics" in metadata:
            metrics = metadata["metrics"]
            f.write(f"\n\nPerformance Metrics:\n")
            for metric_name, metric_value in metrics.items():
                f.write(f"- {metric_name}: {metric_value:.4f}\n")

    print(f"Model information saved to {readme_path}")
    print(f"Model artifacts successfully saved to {model_dir}")

    # Create a "latest" directory by copying (Windows-friendly approach)
    latest_link = os.path.join(base_dir, "latest")

    # Import shutil for directory operations
    import shutil

    # If "latest" already exists, try to remove it
    if os.path.exists(latest_link):
        try:
            # Remove the directory and all its contents
            shutil.rmtree(latest_link)
            print(f"Removed existing 'latest' directory")
        except Exception as e:
            print(f"Warning: Could not remove existing 'latest' directory: {e}")
            # Use an alternative name if we can't remove the existing directory
            latest_link = os.path.join(base_dir, "latest_model")
            if os.path.exists(latest_link):
                try:
                    shutil.rmtree(latest_link)
                except:
                    print(
                        f"Also could not remove '{latest_link}', using timestamp as name"
                    )
                    latest_link = os.path.join(base_dir, f"latest_{timestamp}")

    # Copy the entire model directory to "latest"
    try:
        shutil.copytree(model_dir, latest_link)
        print(
            f"Created '{os.path.basename(latest_link)}' directory pointing to {timestamp}"
        )
    except Exception as e:
        print(
            f"Warning: Could not create '{os.path.basename(latest_link)}' directory: {e}"
        )
        print("Continuing without creating a 'latest' reference")

    return model_dir


# Prepare metadata
# Prepare metadata with properly defined svd_components
# Convert NumPy arrays to lists for safe serialization
feature_names_list = None
if feature_names is not None:
    feature_names_list = (
        feature_names.tolist()
        if isinstance(feature_names, np.ndarray)
        else list(feature_names)
    )

# Define svd_components here in the main code before using it
svd_components = None
if svd_transformer is not None and hasattr(svd_transformer, "components_"):
    svd_components = svd_transformer.components_.tolist()

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
}

# Save model artifacts using the function
print("\nSaving model artifacts with timestamped directory...")
model_dir = save_model_artifacts(
    model, tfidf_vectorizer, svd_transformer, metadata, feature_names
)
print(f"Model artifacts saved to {model_dir}")

print("Model saving complete!")
