import os
import string
import pandas as pd
import pdfplumber
import re
import pickle
from catboost import CatBoostClassifier, Pool
import numpy as np
from datetime import datetime


# Function to extract data from a police report PDF
def extract_data_from_pdf(pdf_path):
    data = {}

    with pdfplumber.open(pdf_path) as pdf:
        text = pdf.pages[0].extract_text()

        # Extract the key fields using regex patterns
        date_match = re.search(r"Date & Time:\s+([\d-]+\s+[\d:]+)", text)
        if date_match:
            data["Dates"] = date_match.group(1)

        description_match = re.search(
            r"Detailed Description:\s+(.+?)(?:Police District:|$)", text, re.DOTALL
        )
        if description_match:
            data["Descript"] = description_match.group(1).strip()

        district_match = re.search(r"Police District:\s+(.+?)(?:Resolution:|$)", text)
        if district_match:
            data["PdDistrict"] = district_match.group(1).strip()

        coordinates_match = re.search(
            r"Coordinates:\s+\(([-\d.]+),\s+([-\d.]+)\)", text
        )
        if coordinates_match:
            data["Latitude (Y)"] = float(coordinates_match.group(1))
            data["Longitude (X)"] = float(coordinates_match.group(2))

        dayofweek = datetime.strptime(data["Dates"], "%Y-%m-%d %H:%M:%S").strftime("%A")
        data["DayOfWeek"] = dayofweek

    return data


# Load all PDF data and create a DataFrame
def load_pdf_data(pdf_dir):
    all_data = []

    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_dir, filename)
            try:
                report_data = extract_data_from_pdf(file_path)
                all_data.append(report_data)
                print(f"Processed {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    return pd.DataFrame(all_data)


# For this example, let's assume we've already loaded the PDFs into a DataFrame
# This would normally be: df = load_pdf_data('/path/to/pdf/directory')

# Create a sample DataFrame based on the PDFs you shared
pdf_data = [
    {
        "Dates": "2013-08-11 18:00:00",
        "Descript": "Petty theft from locked auto. Personal belongings were stolen from a parked vehicle.",
        "DayOfWeek": "Sunday",
        "PdDistrict": "Southern",
        "Longitude (X)": -122.404100362918,
        "Latitude (Y)": 37.78091651016261,
    },
    {
        "Dates": "2010-10-04 12:15:00",
        "Descript": "Petty theft of property. Wallet and phone reported stolen from a public location.",
        "DayOfWeek": "Monday",
        "PdDistrict": "Central",
        "Longitude (X)": -122.406429382098,
        "Latitude (Y)": 37.8071368068488,
    },
    {
        "Dates": "2010-09-10 17:17:00",
        "Descript": "Investigative detention. Officers detained a person matching the description of a suspect in another crime.",
        "DayOfWeek": "Friday",
        "PdDistrict": "Tenderloin",
        "Longitude (X)": -122.407854234538,
        "Latitude (Y)": 37.7854922044602,
    },
    {
        "Dates": "2008-09-20 09:00:00",
        "Descript": "Burglary of apartment house, unlawful entry. Suspect forced entry and stole electronics.",
        "DayOfWeek": "Saturday",
        "PdDistrict": "Tenderloin",
        "Longitude (X)": -122.412573643201,
        "Latitude (Y)": 37.7834687204586,
    },
    {
        "Dates": "2012-08-07 18:00:00",
        "Descript": "Missing juvenile reported. Individual was later located safely.",
        "DayOfWeek": "Tuesday",
        "PdDistrict": "Park",
        "Longitude (X)": -122.444994681535,
        "Latitude (Y)": 37.77743975916521,
    },
    {
        "Dates": "2014-09-06 19:00:00",
        "Descript": "Aided case, mentally disturbed individual. Subject taken to psychiatric care.",
        "DayOfWeek": "Saturday",
        "PdDistrict": "Ingleside",
        "Longitude (X)": -122.415587200313,
        "Latitude (Y)": 37.7126758141231,
    },
    {
        "Dates": "2010-12-29 08:39:00",
        "Descript": "Evading a police officer recklessly. High-speed pursuit ended in suspect escaping.",
        "DayOfWeek": "Wednesday",
        "PdDistrict": "Bayview",
        "Longitude (X)": -122.386420682522,
        "Latitude (Y)": 37.7295447729645,
    },
    {
        "Dates": "2015-04-28 14:35:00",
        "Descript": "Petty theft of property. Personal items taken from a commercial store.",
        "DayOfWeek": "Tuesday",
        "PdDistrict": "Taraval",
        "Longitude (X)": -122.474102710416,
        "Latitude (Y)": 37.743174496996296,
    },
    {
        "Dates": "2004-03-06 17:45:00",
        "Descript": "Grand theft from locked auto. Expensive belongings stolen from a parked car.",
        "DayOfWeek": "Saturday",
        "PdDistrict": "Southern",
        "Longitude (X)": -122.389008295709,
        "Latitude (Y)": 37.7892474519723,
    },
]

df = pd.DataFrame(pdf_data)

# Load model and preprocessing components
print("Loading model and preprocessing objects...")
model = CatBoostClassifier()
model.load_model("model_artifacts/catboost_model.cbm")

with open("model_artifacts/tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

with open("model_artifacts/svd_transformer.pkl", "rb") as f:
    svd_transformer = pickle.load(f)

with open("model_artifacts/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

categorical_features = metadata["categorical_features"]


# Text preprocessing function (same as in training)
def preprocess_text(text):
    """Cleans and preprocesses text for NLP features"""
    if not isinstance(text, str):
        return ""

    # Convert to lowercase and remove punctuation
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Process data for prediction
def process_inference_data(df, tfidf, svd):
    """Process data for inference using the same transformations as training"""
    import string

    # Create copy to avoid modifying original
    df_processed = df.copy()

    # Basic datetime conversion
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

    # Spatial features
    city_center_x, city_center_y = -122.4194, 37.7749  # SF coordinates
    if (
        "Longitude (X)" in df_processed.columns
        and "Latitude (Y)" in df_processed.columns
    ):
        df_processed["DistanceFromCenter"] = np.sqrt(
            (df_processed["Longitude (X)"] - city_center_x) ** 2
            + (df_processed["Latitude (Y)"] - city_center_y) ** 2
        )

        df_processed["QuadrantNS"] = (
            df_processed["Latitude (Y)"] > city_center_y
        ).astype(int)
        df_processed["QuadrantEW"] = (
            df_processed["Longitude (X)"] > city_center_x
        ).astype(int)

    # Categorical encoding
    if "PdDistrict" in df_processed.columns:
        df_processed["PdDistrict"] = pd.Categorical(df_processed["PdDistrict"]).codes

    if "DayOfWeek" in df_processed.columns:
        df_processed["DayOfWeek"] = pd.Categorical(df_processed["DayOfWeek"]).codes

    # Add NLP features if we have the transformers
    if "Descript" in df_processed.columns and tfidf is not None and svd is not None:
        df_processed["Descript_processed"] = (
            df_processed["Descript"].fillna("").apply(preprocess_text)
        )
        tfidf_matrix = tfidf.transform(df_processed["Descript_processed"])
        tfidf_svd = svd.transform(tfidf_matrix)

        # Add TF-IDF SVD features
        for i in range(tfidf_svd.shape[1]):
            df_processed[f"tfidf_Descript_{i}"] = tfidf_svd[:, i]

    # Select features used by model
    features = metadata["feature_names"]

    # Check if all features exist in the dataframe
    missing_features = [f for f in features if f not in df_processed.columns]
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        # Add missing features with default value of 0
        for feat in missing_features:
            df_processed[feat] = 0

    return df_processed[features]


# Process data
print("Processing data...")
X_data = process_inference_data(df, tfidf_vectorizer, svd_transformer)

# Create pool with categorical features
data_pool = Pool(X_data, cat_features=categorical_features)

# Make predictions
print("Making predictions...")
predictions = model.predict(data_pool)
probabilities = model.predict_proba(data_pool)

# Get category mapping
if hasattr(model, "classes_"):
    class_names = model.classes_
else:
    # If the model doesn't store class names, we'll need to recreate the mapping
    # For example, if you know the original categories:
    class_names = [
        "LARCENY/THEFT",
        "OTHER OFFENSES",
        "NON-CRIMINAL",
        "ASSAULT",
        "DRUG/NARCOTIC",
        "VEHICLE THEFT",
        "BURGLARY",
        "VANDALISM",
        "WARRANTS",
        "SUSPICIOUS OCC",
        "ROBBERY",
        "MISSING PERSON",
    ]

# Display results
print("\nPrediction Results:")
for i, (desc, pred) in enumerate(zip(df["Descript"], predictions)):
    print(f"\nReport {i+1}: {desc[:80]}...")
    print(
        f"Predicted Category: {class_names[int(pred)] if isinstance(class_names, list) else pred}"
    )

    print("Top 3 Probabilities:")
    probs = probabilities[i]
    top3_indices = probs.argsort()[-3:][::-1]
    for idx in top3_indices:
        category = class_names[idx] if isinstance(class_names, list) else idx
        print(f"  {category}: {probs[idx]:.4f}")
