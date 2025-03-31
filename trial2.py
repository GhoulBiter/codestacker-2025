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
import optuna
import pickle
import os
import json
import matplotlib
import time
import sys
import datetime
import lightgbm as lgb
import xgboost as xgb

# Create a timestamp for directory naming
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_dir = os.path.join("model_artifacts", timestamp)
os.makedirs(model_dir, exist_ok=True)
print(f"Model artifacts will be saved in: {model_dir}")

matplotlib.use("Agg")  # Set non-interactive backend

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


def extract_nlp_features(df, text_column, max_features=100, n_components=50):
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
def process_data(df, tfidf=None, svd=None, include_nlp=True):
    """Process data with both standard and NLP features"""
    # Create copy to avoid modifying original
    df_processed = df.copy()

    # Print sample of first few rows to debug
    print("First few rows of data:")
    print(df_processed.head(2))

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
    else:
        print(
            "Warning: 'Longitude (X)' and/or 'Latitude (Y)' columns not found, skipping spatial features"
        )
        # Add placeholder columns with default values
        df_processed["DistanceFromCenter"] = 0
        df_processed["QuadrantNS"] = 0
        df_processed["QuadrantEW"] = 0

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

    # Select features
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
    ]

    # Add X and Y if they exist
    if "X" in df_processed.columns and "Y" in df_processed.columns:
        features.extend(["X", "Y"])

    # Add NLP features if present
    nlp_features = [col for col in df_processed.columns if col.startswith("tfidf_")]
    features.extend(nlp_features)

    return df_processed[features]


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
    vis_path = os.path.join(model_dir, "nlp_components_visualization.png")
    plt.savefig(vis_path)
    print(f"NLP component visualization saved to '{vis_path}'")


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

# 8. Hyperparameter optimization with Optuna and model training
# Hybrid optimization approach: CatBoost's built-in tuner + Optuna for other models

# 8a. CatBoost optimization using its native grid search
print("\n" + "=" * 50)
print("Starting CatBoost hyperparameter optimization with native grid search...")
print("=" * 50)

# Create the pools for CatBoost
train_pool = Pool(X_train, y_train, cat_features=categorical_features)
val_pool = Pool(X_val, y_val, cat_features=categorical_features)

# Define parameter grid for CatBoost
# This is more efficient than having Optuna generate these combinations
catboost_param_grid = {
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "depth": [4, 6, 8, 10],
    "l2_leaf_reg": [1, 3, 5, 7],
    "bootstrap_type": ["Bayesian", "Bernoulli"],
    "random_strength": [1, 3, 5],
    "one_hot_max_size": [10, 50, 100],
    "bagging_temperature": [0, 1, 10],  # Only used for 'Bayesian' bootstrap_type
}

# Create a base CatBoost model
catboost_base = CatBoostClassifier(
    loss_function="MultiClass",
    eval_metric="TotalF1",
    iterations=500,  # We'll optimize early stopping within grid search
    random_seed=42,
    od_type="Iter",
    od_wait=50,
    thread_count=-1,  # Use all available CPU cores
)

# Check GPU availability
try_gpu = True
try:
    # Quick test if GPU is available
    test_model = CatBoostClassifier(task_type="GPU", iterations=1, verbose=0)
    test_pool = Pool(X_train.iloc[:10], y_train.iloc[:10])
    test_model.fit(test_pool)
    has_gpu = True
    print("GPU is available and will be used for compatible configurations")
except Exception as e:
    has_gpu = False
    print(f"GPU is not available or encountered an error: {e}")
    print("Using CPU for CatBoost")

# Calculate total number of grid search combinations
total_combinations = (
    len(catboost_param_grid["learning_rate"])
    * len(catboost_param_grid["depth"])
    * len(catboost_param_grid["l2_leaf_reg"])
    * len(catboost_param_grid["bootstrap_type"])
    * len(catboost_param_grid["random_strength"])
    * len(catboost_param_grid["one_hot_max_size"])
    * len(catboost_param_grid["bagging_temperature"])
)

# Run the grid search
print(
    f"Running CatBoost grid search with {total_combinations} possible combinations..."
)
start_time = time.time()

try:
    grid_search_results = catboost_base.grid_search(
        param_grid=catboost_param_grid,
        X=train_pool,
        y=None,  # Not needed since we pass Pool
        cv=3,
        partition_random_seed=42,
        calc_cv_statistics=True,
        search_by_train_test_split=True,
        refit=False,
        shuffle=True,
        stratified=True,
        train_size=0.8,
        verbose=True,
        plot=False,
    )

    # Extract the best parameters
    catboost_best_params = grid_search_results["params"]
    catboost_best_score = grid_search_results["cv_results"]["test-TotalF1-mean"][-1]

    print(f"\nCatBoost Grid Search completed in {time.time() - start_time:.2f} seconds")
    print("\nBest CatBoost parameters:")
    for param, value in catboost_best_params.items():
        print(f"{param}: {value}")
    print(f"Best cv TotalF1 score: {catboost_best_score:.4f}")

    # Save grid search results
    grid_results_path = os.path.join(model_dir, "catboost_grid_search_results.pkl")
    with open(grid_results_path, "wb") as f:
        pickle.dump(grid_search_results, f)
    print(f"Grid search results saved to '{grid_results_path}'")

    # Train the final CatBoost model with the best parameters
    if has_gpu and try_gpu and catboost_best_params.get("bootstrap_type") != "MVS":
        catboost_best_params["task_type"] = "GPU"
    else:
        catboost_best_params["task_type"] = "CPU"

    print(
        f"\nTraining final CatBoost model with best parameters using {catboost_best_params['task_type']}..."
    )

    catboost_model = CatBoostClassifier(
        **catboost_best_params,
        loss_function="MultiClass",
        eval_metric="TotalF1",
        random_seed=42,
        iterations=1000,  # We'll use early stopping
        od_type="Iter",
        od_wait=50,
        verbose=100,
    )

    catboost_model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    # Evaluate CatBoost
    catboost_preds = catboost_model.predict(X_val)
    catboost_f1 = f1_score(y_val, catboost_preds, average="macro")
    print(f"CatBoost Validation Macro F1: {catboost_f1:.4f}")

except Exception as e:
    print(f"Error in CatBoost grid search: {e}")
    # Fallback to default CatBoost
    print("Falling back to default CatBoost model...")
    catboost_model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3,
        loss_function="MultiClass",
        eval_metric="TotalF1",
        random_seed=42,
        verbose=100,
        task_type="CPU",
    )
    catboost_model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    catboost_preds = catboost_model.predict(X_val)
    catboost_f1 = f1_score(y_val, catboost_preds, average="macro")
    catboost_best_params = {
        "iterations": 500,
        "learning_rate": 0.05,
        "depth": 8,
        "l2_leaf_reg": 3,
        "task_type": "CPU",
    }
    catboost_best_score = catboost_f1
    print(f"Default CatBoost Validation Macro F1: {catboost_f1:.4f}")

# 8b. Optuna optimization for other models (LightGBM, XGBoost, etc.)
print("\n" + "=" * 50)
print("Starting Optuna hyperparameter optimization for other models...")
print("=" * 50)


# Define LightGBM objective function for Optuna
def lightgbm_objective(trial):
    # Define hyperparameters to search
    params = {
        "objective": "multiclass",
        "metric": "multi_logloss",
        "num_class": len(np.unique(y_train)),
        "verbosity": -1,
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
    }

    # Handle categorical features
    cat_features_indices = [
        X_train.columns.get_loc(col)
        for col in categorical_features
        if col in X_train.columns
    ]

    # Create datasets for LightGBM
    lgb_train = lgb.Dataset(
        X_train,
        y_train,
        categorical_feature=cat_features_indices if cat_features_indices else "auto",
    )
    lgb_val = lgb.Dataset(
        X_val,
        y_val,
        reference=lgb_train,
        categorical_feature=cat_features_indices if cat_features_indices else "auto",
    )

    try:
        # Train LightGBM model with early stopping
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=1000,
            valid_sets=[lgb_val],
            early_stopping_rounds=50,
            verbose_eval=False,
        )

        # Make predictions
        preds = model.predict(X_val)
        pred_labels = np.argmax(preds, axis=1)
        f1 = f1_score(y_val, pred_labels, average="macro")

        return f1

    except Exception as e:
        print(f"Error in LightGBM trial: {e}")
        raise optuna.exceptions.TrialPruned()


# Define XGBoost objective function for Optuna
def xgboost_objective(trial):
    # Define hyperparameters to search
    params = {
        "objective": "multi:softmax",
        "num_class": len(np.unique(y_train)),
        "eval_metric": "mlogloss",
        "verbosity": 0,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
    }

    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    try:
        # Train XGBoost model with early stopping
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )

        # Make predictions
        preds = model.predict(dval)
        f1 = f1_score(y_val, preds, average="macro")

        return f1

    except Exception as e:
        print(f"Error in XGBoost trial: {e}")
        raise optuna.exceptions.TrialPruned()


# Run Optuna studies for LightGBM and XGBoost
models = {}
best_scores = {}
lgb_best_params = None
xgb_best_params = None
lgb_study = None
xgb_study = None

# Optuna for LightGBM
print("Running Optuna optimization for LightGBM...")
lgb_study = optuna.create_study(direction="maximize")
try:
    lgb_study.optimize(
        lightgbm_objective, n_trials=20
    )  # Fewer trials than CatBoost grid search

    # Get best parameters
    lgb_best_params = lgb_study.best_params
    lgb_best_params.update(
        {
            "objective": "multiclass",
            "metric": "multi_logloss",
            "num_class": len(np.unique(y_train)),
            "verbosity": -1,
            "boosting_type": "gbdt",
        }
    )

    print("\nBest LightGBM parameters:")
    for param, value in lgb_best_params.items():
        print(f"{param}: {value}")

    # Train final LightGBM model
    cat_features_indices = [
        X_train.columns.get_loc(col)
        for col in categorical_features
        if col in X_train.columns
    ]
    lgb_train = lgb.Dataset(
        X_train,
        y_train,
        categorical_feature=cat_features_indices if cat_features_indices else "auto",
    )
    lgb_val = lgb.Dataset(
        X_val,
        y_val,
        reference=lgb_train,
        categorical_feature=cat_features_indices if cat_features_indices else "auto",
    )

    lgb_model = lgb.train(
        lgb_best_params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_val],
        early_stopping_rounds=50,
        verbose_eval=100,
    )

    models["lightgbm"] = lgb_model

    # Evaluate LightGBM
    lgb_preds = lgb_model.predict(X_val)
    lgb_pred_labels = np.argmax(lgb_preds, axis=1)
    lgb_f1 = f1_score(y_val, lgb_pred_labels, average="macro")
    print(f"LightGBM Validation Macro F1: {lgb_f1:.4f}")
    best_scores["lightgbm"] = lgb_f1

    # Save LightGBM study
    lgb_study_path = os.path.join(model_dir, "lightgbm_study.pkl")
    with open(lgb_study_path, "wb") as f:
        pickle.dump(lgb_study, f)
    print(f"LightGBM study saved to '{lgb_study_path}'")

except Exception as e:
    print(f"Error in LightGBM optimization: {e}")

# Optuna for XGBoost
print("\nRunning Optuna optimization for XGBoost...")
xgb_study = optuna.create_study(direction="maximize")
try:
    xgb_study.optimize(
        xgboost_objective, n_trials=20
    )  # Fewer trials than CatBoost grid search

    # Get best parameters
    xgb_best_params = xgb_study.best_params
    xgb_best_params.update(
        {
            "objective": "multi:softmax",
            "num_class": len(np.unique(y_train)),
            "eval_metric": "mlogloss",
            "verbosity": 0,
        }
    )

    print("\nBest XGBoost parameters:")
    for param, value in xgb_best_params.items():
        print(f"{param}: {value}")

    # Train final XGBoost model
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    xgb_model = xgb.train(
        xgb_best_params,
        dtrain,
        num_boost_round=1000,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=100,
    )

    models["xgboost"] = xgb_model

    # Evaluate XGBoost
    xgb_preds = xgb_model.predict(dval)
    xgb_f1 = f1_score(y_val, xgb_preds, average="macro")
    print(f"XGBoost Validation Macro F1: {xgb_f1:.4f}")
    best_scores["xgboost"] = xgb_f1

    # Save XGBoost study
    xgb_study_path = os.path.join(model_dir, "xgboost_study.pkl")
    with open(xgb_study_path, "wb") as f:
        pickle.dump(xgb_study, f)
    print(f"XGBoost study saved to '{xgb_study_path}'")

except Exception as e:
    print(f"Error in XGBoost optimization: {e}")

# 9. Compare all optimized models
print("\n" + "=" * 50)
print("Model Performance Comparison")
print("=" * 50)

all_models = {
    "CatBoost": catboost_model,
}

if "lightgbm" in models:
    all_models["LightGBM"] = models["lightgbm"]
if "xgboost" in models:
    all_models["XGBoost"] = models["xgboost"]

comparison_results = {
    "CatBoost": {"F1 Score": catboost_f1, "Optimization Method": "Native Grid Search"}
}

if "lightgbm" in best_scores:
    comparison_results["LightGBM"] = {
        "F1 Score": best_scores["lightgbm"],
        "Optimization Method": "Optuna",
    }

if "xgboost" in best_scores:
    comparison_results["XGBoost"] = {
        "F1 Score": best_scores["xgboost"],
        "Optimization Method": "Optuna",
    }

# Print results table
comparison_df = pd.DataFrame(comparison_results).T
print("\nModel Comparison:")
print(comparison_df.sort_values("F1 Score", ascending=False))

# Save comparison results
comparison_path = os.path.join(model_dir, "model_comparison.csv")
comparison_df.to_csv(comparison_path)
print(f"Model comparison saved to '{comparison_path}'")

# Select the best overall model
best_model_name = comparison_df.sort_values("F1 Score", ascending=False).index[0]
print(f"\nBest performing model: {best_model_name}")

# Set the best model as the final model
if best_model_name == "CatBoost":
    model = catboost_model
elif best_model_name == "LightGBM":
    model = models["lightgbm"]
elif best_model_name == "XGBoost":
    model = models["xgboost"]

# 10. Evaluate model
print("Evaluating best model...")
if best_model_name == "CatBoost":
    val_predictions = model.predict(X_val)
elif best_model_name == "LightGBM":
    val_predictions = np.argmax(model.predict(X_val), axis=1)
elif best_model_name == "XGBoost":
    val_predictions = model.predict(xgb.DMatrix(X_val))

# Calculate metrics
accuracy = accuracy_score(y_val, val_predictions)
print(f"Validation Accuracy: {accuracy:.4f}")

macro_f1 = f1_score(y_val, val_predictions, average="macro")
print(f"Validation Macro F1 Score: {macro_f1:.4f}")

weighted_f1 = f1_score(y_val, val_predictions, average="weighted")
print(f"Validation Weighted F1 Score: {weighted_f1:.4f}")

# Print detailed classification report
print("\nDetailed Classification Report:")
report = classification_report(y_val, val_predictions)
print(report)


# 11. Feature importance
def plot_feature_importance(model, feature_names, model_name):
    plt.figure(figsize=(12, 8))

    if model_name == "CatBoost":
        importance = model.get_feature_importance()
        features = pd.DataFrame(
            {"Feature": feature_names, "Importance": importance}
        ).sort_values("Importance", ascending=False)

        # Highlight NLP features
        features["is_nlp"] = features["Feature"].str.startswith("tfidf_")

        # Plot with color differentiation for NLP features
        colors = [
            "#1f77b4" if not is_nlp else "#ff7f0e" for is_nlp in features["is_nlp"]
        ]

        plt.barh(
            range(len(features)), features["Importance"], align="center", color=colors
        )
        plt.yticks(range(len(features)), features["Feature"])

        # Add legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="#1f77b4", label="Standard Features"),
            Patch(facecolor="#ff7f0e", label="NLP Features"),
        ]
        plt.legend(handles=legend_elements)

        # Print top NLP features
        nlp_features = features[features["is_nlp"]].head(10)
        if not nlp_features.empty:
            print("\nTop 10 NLP Features:")
            for i, row in nlp_features.iterrows():
                print(f"{row['Feature']}: {row['Importance']:.4f}")

    elif model_name == "LightGBM":
        importance = model.feature_importance()
        features = pd.DataFrame(
            {"Feature": feature_names, "Importance": importance}
        ).sort_values("Importance", ascending=False)

        features["is_nlp"] = features["Feature"].str.startswith("tfidf_")
        colors = [
            "#1f77b4" if not is_nlp else "#ff7f0e" for is_nlp in features["is_nlp"]
        ]

        plt.barh(
            range(len(features)), features["Importance"], align="center", color=colors
        )
        plt.yticks(range(len(features)), features["Feature"])

        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="#1f77b4", label="Standard Features"),
            Patch(facecolor="#ff7f0e", label="NLP Features"),
        ]
        plt.legend(handles=legend_elements)

    elif model_name == "XGBoost":
        importance = model.get_score(importance_type="weight")
        features = pd.DataFrame(
            {
                "Feature": list(importance.keys()),
                "Importance": list(importance.values()),
            }
        ).sort_values("Importance", ascending=False)

        if len(features) > 0:
            features["is_nlp"] = features["Feature"].str.startswith("tfidf_")
            colors = [
                "#1f77b4" if not is_nlp else "#ff7f0e" for is_nlp in features["is_nlp"]
            ]

            plt.barh(
                range(len(features)),
                features["Importance"],
                align="center",
                color=colors,
            )
            plt.yticks(range(len(features)), features["Feature"])

            from matplotlib.patches import Patch

            legend_elements = [
                Patch(facecolor="#1f77b4", label="Standard Features"),
                Patch(facecolor="#ff7f0e", label="NLP Features"),
            ]
            plt.legend(handles=legend_elements)

    plt.xlabel("Feature Importance")
    plt.title(f"Feature Importance ({model_name})")
    plt.tight_layout()

    importance_path = os.path.join(
        model_dir, f"feature_importance_{model_name.lower()}.png"
    )
    plt.savefig(importance_path)
    print(f"Feature importance plot saved to '{importance_path}'")


print("\nAnalyzing feature importance...")
plot_feature_importance(model, X_train.columns, best_model_name)

# 12. Save model, preprocessing objects, and optimization results
print("\n" + "=" * 50)
print("Saving model, preprocessing objects, and optimization results...")
print("=" * 50)

# Save the best model
if best_model_name == "CatBoost":
    model_path = os.path.join(model_dir, "catboost_model.cbm")
    model.save_model(model_path)
elif best_model_name == "LightGBM":
    model_path = os.path.join(model_dir, "lightgbm_model.txt")
    model.save_model(model_path)
elif best_model_name == "XGBoost":
    model_path = os.path.join(model_dir, "xgboost_model.json")
    model.save_model(model_path)

print(f"Best model ({best_model_name}) saved to '{model_path}'")

# Save models for all algorithms
if "CatBoost" in all_models:
    model_path = os.path.join(model_dir, "catboost_model.cbm")
    all_models["CatBoost"].save_model(model_path)
    print(f"CatBoost model saved to '{model_path}'")

if "LightGBM" in all_models:
    model_path = os.path.join(model_dir, "lightgbm_model.txt")
    all_models["LightGBM"].save_model(model_path)
    print(f"LightGBM model saved to '{model_path}'")

if "XGBoost" in all_models:
    model_path = os.path.join(model_dir, "xgboost_model.json")
    all_models["XGBoost"].save_model(model_path)
    print(f"XGBoost model saved to '{model_path}'")

# Save the TF-IDF vectorizer
tfidf_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
with open(tfidf_path, "wb") as f:
    pickle.dump(tfidf_vectorizer, f)
print(f"TF-IDF vectorizer saved to '{tfidf_path}'")

# Save the SVD transformer
svd_path = os.path.join(model_dir, "svd_transformer.pkl")
with open(svd_path, "wb") as f:
    pickle.dump(svd_transformer, f)
print(f"SVD transformer saved to '{svd_path}'")

# Save study objects if available
if lgb_study is not None:
    study_path = os.path.join(model_dir, "lightgbm_study.pkl")
    with open(study_path, "wb") as f:
        pickle.dump(lgb_study, f)
    print(f"LightGBM study saved to '{study_path}'")

if xgb_study is not None:
    study_path = os.path.join(model_dir, "xgboost_study.pkl")
    with open(study_path, "wb") as f:
        pickle.dump(xgb_study, f)
    print(f"XGBoost study saved to '{study_path}'")

# Save best hyperparameters as JSON
params_info = {
    "catboost": catboost_best_params if "catboost_best_params" in locals() else None,
    "lightgbm": lgb_best_params,
    "xgboost": xgb_best_params,
    "best_model": best_model_name,
}

best_params_path = os.path.join(model_dir, "best_hyperparameters.json")
with open(best_params_path, "w") as f:
    json.dump(params_info, f, indent=4)
print(f"Best hyperparameters saved to '{best_params_path}'")

# Save classification report as text
report_path = os.path.join(model_dir, "classification_report.txt")
with open(report_path, "w") as f:
    f.write(report)
print(f"Classification report saved to '{report_path}'")

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

# Save feature names and other metadata
metadata = {
    "feature_names": X_train.columns.tolist(),
    "categorical_features": categorical_features,
    "svd_components": svd_components,
    "tfidf_feature_names": feature_names_list,
    "optimization_results": {
        "catboost": {
            "best_params": (
                catboost_best_params if "catboost_best_params" in locals() else None
            ),
            "best_score": (
                catboost_best_score if "catboost_best_score" in locals() else None
            ),
        },
        "lightgbm": {
            "best_params": lgb_best_params,
            "best_score": best_scores.get("lightgbm", None),
        },
        "xgboost": {
            "best_params": xgb_best_params,
            "best_score": best_scores.get("xgboost", None),
        },
        "best_model": best_model_name,
    },
    "timestamp": timestamp,
}

metadata_path = os.path.join(model_dir, "metadata.pkl")
with open(metadata_path, "wb") as f:
    pickle.dump(metadata, f)
print(f"Model metadata saved to '{metadata_path}'")

# Save performance metrics
metrics = {
    "accuracy": float(accuracy),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),
    "best_model": best_model_name,
}

metrics_path = os.path.join(model_dir, "performance_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)
print(f"Performance metrics saved to '{metrics_path}'")

# Create a simplified README file to explain the contents of this directory
readme_content = f"""# Crime Classification Model - Generated on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Model Summary
- Best Model: {best_model_name}
- Validation Accuracy: {accuracy:.4f}
- Validation Macro F1 Score: {macro_f1:.4f}
- Validation Weighted F1 Score: {weighted_f1:.4f}

## Directory Contents
- Best model: {best_model_name.lower()}_model.*
- TF-IDF Vectorizer: tfidf_vectorizer.pkl
- SVD Transformer: svd_transformer.pkl
- Feature Importance: feature_importance_{best_model_name.lower()}.png
- Performance Metrics: performance_metrics.json
- Metadata: metadata.pkl
- NLP Component Visualization: nlp_components_visualization.png
- Optimization Results: best_hyperparameters.json

## How to Use
To load and use this model for predictions, see the example in inference.py.
This model was trained on crime descriptions to classify crime types.
"""

readme_path = os.path.join(model_dir, "README.md")
with open(readme_path, "w") as f:
    f.write(readme_content)
print(f"README file generated at '{readme_path}'")

print("\nAll model artifacts and optimization results saved successfully!")
print(f"Model artifacts directory: {model_dir}")
print("=" * 50)
