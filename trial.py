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
import pickle
import os
import numpy as np

# Create a directory for saving model artifacts if it doesn't exist
os.makedirs("model_artifacts", exist_ok=True)

print("\nSaving model and preprocessing objects...")

# Save the CatBoost model
model_path = "model_artifacts/catboost_model.cbm"
model.save_model(model_path)
print(f"Model saved to {model_path}")

# Save the TF-IDF vectorizer
tfidf_path = "model_artifacts/tfidf_vectorizer.pkl"
with open(tfidf_path, "wb") as f:
    pickle.dump(tfidf_vectorizer, f)
print(f"TF-IDF vectorizer saved to {tfidf_path}")

# Save the SVD transformer
svd_path = "model_artifacts/svd_transformer.pkl"
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

# Save feature names and other metadata
metadata = {
    "feature_names": X_train.columns.tolist(),
    "categorical_features": categorical_features,
    "svd_components": svd_components,
    "tfidf_feature_names": feature_names_list,
}
metadata_path = "model_artifacts/metadata.pkl"
with open(metadata_path, "wb") as f:
    pickle.dump(metadata, f)
print(f"Model metadata saved to {metadata_path}")

print("Model saving complete!")
