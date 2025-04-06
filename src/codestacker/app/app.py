# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import json
import matplotlib.pyplot as plt
import pydeck as pdk
import plotly.express as px
from catboost import CatBoostClassifier, Pool
import datetime
import re
import string
import pdfplumber
from google import genai
from google.genai import types
from dotenv import load_dotenv
import textwrap

load_dotenv()

# Set page configuration
st.set_page_config(page_title="CityX Crime Classifier", page_icon="🔍", layout="wide")


# Helper functions for data processing and prediction
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def preprocess_text(text):
    """Cleans and preprocesses text for NLP features"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", " ", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_model_artifacts(model_dir):
    """Load model and preprocessing components"""
    model_path = os.path.join(model_dir, "catboost_model.cbm")
    tfidf_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
    svd_path = os.path.join(model_dir, "svd_transformer.pkl")
    metadata_path = os.path.join(model_dir, "metadata.pkl")

    # Load catboost model
    model = CatBoostClassifier()
    model.load_model(model_path)

    # Load vectorizer and SVD transformer
    with open(tfidf_path, "rb") as f:
        tfidf_vectorizer = pickle.load(f)

    with open(svd_path, "rb") as f:
        svd_transformer = pickle.load(f)

    # Load metadata
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    return model, tfidf_vectorizer, svd_transformer, metadata


def extract_report_fields(pdf_text):
    """
    Extract all labeled fields from the PDF text by explicitly stopping when the next known title is encountered.
    Newlines are replaced with spaces and extra whitespace is stripped.
    """
    text = pdf_text.strip()
    patterns = {
        "report_number": r"Report Number:\s*(.*?)\s*Date & Time:",
        "date_time": r"Date & Time:\s*(.*?)\s*Reporting Officer:",
        "reporting_officer": r"Reporting Officer:\s*(.*?)\s*Incident Location:",
        "incident_location": r"Incident Location:\s*(.*?)\s*Coordinates:",
        # For coordinates, explicitly capture content inside parentheses.
        "coordinates": r"Coordinates:\s*(\([^)]*\))\s*Detailed Description:",
        "detailed_description": r"Detailed Description:\s*(.*?)\s*Police District:",
        "police_district": r"Police District:\s*(.*?)\s*Resolution:",
        "resolution": r"Resolution:\s*(.*?)\s*Suspect Description:",
        "suspect_description": r"Suspect Description:\s*(.*?)\s*Victim Information:",
        "victim_information": r"Victim Information:\s*(.*)",
    }
    flags = re.DOTALL | re.IGNORECASE
    results = {}
    for field, pattern in patterns.items():
        match = re.search(pattern, text, flags=flags)
        if match:
            # Replace newline characters with a space and strip extra whitespace
            value = match.group(1).replace("\n", " ").strip()
            results[field] = value
        else:
            results[field] = ""
    return results


def setup_gemini_api():
    """Set up Gemini API with API key from .env file"""
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        st.warning(
            "Gemini API key not found. Please set GEMINI_API_KEY in your .env file."
        )
        return None

    try:
        # Create a client
        client = genai.Client(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"Failed to initialize Gemini client: {e}")
        return None


@st.cache_resource
def get_gemini_client():
    """Get the Gemini client with caching"""
    return setup_gemini_api()


SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are an expert in police report classification.
    Your sole task is to convert a narrative crime description into one of the standardized police report formats.
    THINK STEP BY STEP internally, but DO NOT output any chain-of-thought, reasoning, or intermediate text.
    
    YOU MUST:
      - Analyze the crime description thoroughly.
      - Compare it with the provided list of standardized formats.
      - Select EXACTLY ONE format from the list that best matches the key elements of the description.
      - Output ONLY that format, in EXACTLY the same wording, punctuation, and spacing as provided, in ALL CAPITAL LETTERS.
    
    DO NOT:
      - Output any additional text, punctuation, or explanation.
      - Modify or paraphrase the provided formats.
      - Include any chain-of-thought or reasoning in your final output.
    """
).strip()


def rephrase_description_with_gemini(description: str, crime_formats: list) -> str:
    """
    Use the Gemini API to rephrase a narrative crime description into a standardized police report format.
    This version provides extensive, step-by-step guidance and implements validation with retries.

    Args:
        description: The original narrative crime description.
        crime_formats: List of standardized police report format examples.

    Returns:
        A rephrased description that exactly matches one of the standardized formats.
    """
    client = get_gemini_client()
    if not client:
        # Fallback: return the description in uppercase if the API client isn't available.
        return description.upper().strip()

    # Create a numbered list of the first 30 formats (to control prompt size).
    format_examples = "\n".join(
        [f"{i+1}. {fmt}" for i, fmt in enumerate(crime_formats[:30])]
    )

    # Build the user prompt (dynamic part) with strict instructions.
    user_prompt = textwrap.dedent(
        f"""\
        # Provided Standardized Formats:
        {format_examples}

        # Narrative Crime Description:
        "{description}"

        # STRICT INSTRUCTIONS:
        Analyze the description carefully.
        Compare it with the above list of standardized formats.
        Select ONLY the SINGLE best matching format.
        Your output MUST be the EXACT standardized format, exactly as provided, in ALL CAPITAL LETTERS.
        Do not include any extra words, punctuation, or explanations.
        """
    ).strip()

    max_retries = 3
    attempt = 0

    while attempt < max_retries:
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[user_prompt],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.1,
                    top_p=0.95,
                    top_k=40,
                    max_output_tokens=64,
                ),
            )
            rephrased = response.text.strip()

            # Enforce uppercase.
            if not rephrased.isupper():
                rephrased = rephrased.upper()

            # If response is multi-line or too long, extract the first valid short line.
            if len(rephrased) > 100 or "\n" in rephrased:
                lines = [line.strip() for line in rephrased.split("\n") if line.strip()]
                for line in lines:
                    if line.isupper() and len(line) < 100:
                        rephrased = line
                        break
                else:
                    rephrased = lines[0] if lines else rephrased

            # Validation: if the output exactly matches one of the provided formats, return it.
            if rephrased in crime_formats:
                return rephrased

            # Alternatively, check if any provided format is contained within the output.
            for fmt in crime_formats:
                if fmt in rephrased:
                    rephrased = fmt
                    break

            if rephrased in crime_formats:
                return rephrased

            attempt += 1

        except Exception as e:
            st.error(f"Error using Gemini API on attempt {attempt + 1}: {e}")
            attempt += 1

    st.warning(
        "Failed to generate an exact standardized format after multiple attempts. Using fallback."
    )
    return description.upper().strip()


# Modified prediction function to use Gemini for rephrasing
def make_prediction_with_gemini(
    description, date, district, coordinates_string, model, tfidf, svd, metadata
):
    """Make a prediction using the model with Gemini-powered rephrasing"""
    try:
        # Get list of unique crime formats
        crime_formats = get_crime_formats()

        # Rephrase the description using Gemini
        formatted_description = rephrase_description_with_gemini(
            description, crime_formats
        )

        # For debugging, show the formatted description if enabled
        if st.session_state.get("show_formatted", True):
            st.info(f"Formatted description for prediction: {formatted_description}")

        # Extract coordinates
        coordinates = extract_coordinates(coordinates_string)

        # Process input with the rephrased description
        X = process_input_for_prediction(
            formatted_description,
            date,
            district,
            coordinates,
            model,
            tfidf,
            svd,
            metadata,
        )

        # Create Pool for CatBoost
        cat_features = metadata["categorical_features"]
        pool = Pool(X, cat_features=cat_features)

        # Get prediction and probabilities
        prediction = model.predict(pool)[0]
        probabilities = model.predict_proba(pool)[0]

        # Get class names
        class_names = model.classes_

        # Create dictionary of class probabilities
        proba_dict = {
            str(class_names[i]): float(probabilities[i])
            for i in range(len(class_names))
        }

        return str(prediction), proba_dict

    except Exception as e:
        import traceback

        st.error(f"Error making prediction: {e}")
        st.code(traceback.format_exc())
        return "Error", {}


# Helper function to get crime formats from file or hardcoded list
@st.cache_data
def get_crime_formats():
    """Get standardized crime formats from file or default to hardcoded list"""
    try:
        # If you have a file with unique descriptions
        with open("unique_descriptions.txt", "r") as f:
            formats = [line.strip() for line in f.readlines()]
        return formats
    except:
        # Return a subset of your crime categories
        return [
            "GRAND THEFT FROM LOCKED AUTO",
            "PETTY THEFT FROM LOCKED AUTO",
            "STOLEN AUTOMOBILE",
            "MALICIOUS MISCHIEF, VANDALISM OF VEHICLES",
            "GRAND THEFT FROM UNLOCKED AUTO",
            "MALICIOUS MISCHIEF, VANDALISM",
            "FOUND PROPERTY",
            "ROBBERY, ARMED WITH A KNIFE",
            "TRAFFIC VIOLATION",
            "ROBBERY, BODILY FORCE",
            "SUSPICIOUS OCCURRENCE",
            "BURGLARY, UNLAWFUL ENTRY",
            "BURGLARY OF RESIDENCE, FORCIBLE ENTRY",
            "GRAND THEFT FROM PERSON",
            "PETTY THEFT SHOPLIFTING",
            "GRAND THEFT FROM A BUILDING",
            "PETTY THEFT FROM A BUILDING",
            "PETTY THEFT OF PROPERTY",
            "STOLEN MOTORCYCLE",
            "GRAND THEFT BICYCLE",
            "PETTY THEFT BICYCLE",
            "AIDED CASE, MENTAL DISTURBED",
            "ILLEGAL SUBSTANCES",
            "LOST PROPERTY",
            "WARRANT ARREST",
            "TRAFFIC VIOLATION ARREST",
            "PROBATION VIOLATION",
            "DRIVERS LICENSE, SUSPENDED OR REVOKED",
            "TRESPASSING",
            "FORGERY & COUNTERFEITING (GENERAL)",
        ]


@st.cache_resource
def get_model_artifacts():
    """Cached function to load model artifacts from the latest model directory"""
    # Find most recent model directory
    base_dir = "model_artifacts"
    if not os.path.exists(base_dir):
        st.error("Model artifacts directory not found!")
        return None, None, None, None

    # Check if 'latest' directory exists
    latest_dir = os.path.join(base_dir, "latest")
    if os.path.exists(latest_dir) and os.path.isdir(latest_dir):
        st.info(f"Loading model from latest directory")
        model_dir = latest_dir
    else:
        # If 'latest' doesn't exist or isn't a directory, find the most recent one
        subdirs = [
            os.path.join(base_dir, d)
            for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d)) and d != "latest"
        ]

        if not subdirs:
            st.error("No model directories found!")
            return None, None, None, None

        model_dir = max(subdirs, key=os.path.getmtime)
        st.info(
            f"Loading model from most recent directory: {os.path.basename(model_dir)}"
        )

    try:
        return load_model_artifacts(model_dir)
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        import traceback

        st.code(traceback.format_exc())
        return None, None, None, None


@st.cache_data
def load_crime_data():
    """Load and preprocess the crime dataset with row limit from environment variable"""
    try:
        # Load limit from environment variable
        max_rows = int(os.getenv("STREAMLIT_MAX_DATA_ROWS", "5000"))

        file_path = "Competition_Dataset.csv"
        if not os.path.exists(file_path):
            st.error("Crime data file not found!")
            return create_demo_data()

        # Use chunked loading to avoid memory overload
        chunk_size = 1000
        chunks = []
        rows_loaded = 0

        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            if rows_loaded + len(chunk) >= max_rows:
                remaining = max_rows - rows_loaded
                chunks.append(chunk.iloc[:remaining])
                break
            else:
                chunks.append(chunk)
                rows_loaded += len(chunk)

        df = pd.concat(chunks, ignore_index=True)

        # Convert dates to datetime
        df["Dates"] = pd.to_datetime(df["Dates"])

        # Handle coordinates
        if "Longitude (X)" in df.columns and "Latitude (Y)" in df.columns:
            df["lon"] = df["Longitude (X)"]
            df["lat"] = df["Latitude (Y)"]
        elif "X" in df.columns and "Y" in df.columns:
            df["lon"] = df["X"]
            df["lat"] = df["Y"]
        else:
            st.warning("No coordinates found. Using random values.")
            df["lon"] = np.random.uniform(-122.51, -122.36, len(df))
            df["lat"] = np.random.uniform(37.71, 37.83, len(df))

        return df

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return create_demo_data()


def create_demo_data():
    """Create demo data if real data can't be loaded"""
    # Create sample data
    dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="D")
    categories = [
        "LARCENY/THEFT",
        "OTHER OFFENSES",
        "NON-CRIMINAL",
        "ASSAULT",
        "DRUG/NARCOTIC",
        "VEHICLE THEFT",
        "BURGLARY",
        "VANDALISM",
    ]

    n_samples = 1000

    df = pd.DataFrame(
        {
            "Dates": np.random.choice(dates, n_samples),
            "Category": np.random.choice(categories, n_samples),
            "Descript": ["Sample crime description"] * n_samples,
            "DayOfWeek": [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ],
            "PdDistrict": [
                "CENTRAL",
                "SOUTHERN",
                "BAYVIEW",
                "MISSION",
                "PARK",
                "RICHMOND",
                "INGLESIDE",
                "TARAVAL",
                "NORTHERN",
            ],
            "lon": np.random.uniform(-122.51, -122.36, n_samples),
            "lat": np.random.uniform(37.71, 37.83, n_samples),
        }
    )

    return df


def process_input_for_prediction(
    description, date, district, coordinates, model, tfidf, svd, metadata
):
    """Process input data for prediction with enhanced spatial features"""
    # Get feature names from metadata
    feature_names = metadata["feature_names"]
    categorical_features = metadata["categorical_features"]

    # Preprocess description
    processed_text = preprocess_text(description)

    # Create TF-IDF features
    tfidf_matrix = tfidf.transform([processed_text])

    # Apply SVD transformation
    tfidf_svd = svd.transform(tfidf_matrix)

    # Create DataFrame with features
    data = {}

    # Add temporal features
    date_dt = pd.to_datetime(date)
    data["Hour"] = [date_dt.hour]
    data["Day"] = [date_dt.day]
    data["Month"] = [date_dt.month]
    data["DayOfWeek_Num"] = [date_dt.weekday()]
    data["WeekendFlag"] = [1 if date_dt.weekday() >= 5 else 0]

    # Time period
    if 5 <= date_dt.hour < 12:
        time_period = 0  # Morning
        time_period_str = "Morning"
    elif 12 <= date_dt.hour < 17:
        time_period = 1  # Afternoon
        time_period_str = "Afternoon"
    elif 17 <= date_dt.hour < 21:
        time_period = 2  # Evening
        time_period_str = "Evening"
    else:
        time_period = 3  # Night
        time_period_str = "Night"
    data["TimePeriod"] = [time_period]

    # Weekend flag
    is_weekend = date_dt.weekday() >= 5

    # Categorical features
    data["DayOfWeek"] = [date_dt.weekday()]
    data["PdDistrict"] = [district]

    # Calculate basic spatial features if coordinates are provided
    city_center_x, city_center_y = (
        -122.4194,
        37.7749,
    )  # CityX Averaged Center coordinates

    # CityX boundaries for grid calculation
    min_lon, max_lon = -122.51, -122.36
    min_lat, max_lat = 37.71, 37.83

    # Grid steps
    lon_step = (max_lon - min_lon) / 5
    lat_step = (max_lat - min_lat) / 5

    if coordinates and isinstance(coordinates, tuple) and len(coordinates) == 2:
        longitude, latitude = coordinates
        # Calculate distance from city center
        data["DistanceFromCenter"] = [
            np.sqrt((longitude - city_center_x) ** 2 + (latitude - city_center_y) ** 2)
        ]
        # Calculate quadrants
        data["QuadrantNS"] = [1 if latitude > city_center_y else 0]
        data["QuadrantEW"] = [1 if longitude > city_center_x else 0]

        # Calculate grid cell
        lon_idx = min(int((longitude - min_lon) / lon_step), 4)
        lat_idx = min(int((latitude - min_lat) / lat_step), 4)
        grid_cell = lat_idx * 5 + lon_idx
    else:
        # If no coordinates provided, use reasonable defaults
        data["DistanceFromCenter"] = [0.05]  # Small non-zero value
        data["QuadrantNS"] = [0]
        data["QuadrantEW"] = [0]
        grid_cell = 12  # Center grid cell

    # Add enhanced spatial features from pre-computed mappings
    try:
        # Crime density features
        with open("model_artifacts/density_mappings.pkl", "rb") as f:
            density_mappings = pickle.load(f)

        # Apply general crime density
        data["CrimeDensity"] = [density_mappings["grid_to_density"].get(grid_cell, 0.1)]

        # Apply category-specific densities if available
        if "category_density" in density_mappings:
            for category, grid_to_cat_density in density_mappings[
                "category_density"
            ].items():
                col_name = f'Density_{category.replace("/", "_").replace(" ", "_")}'
                if col_name in feature_names:
                    data[col_name] = [grid_to_cat_density.get(grid_cell, 0.1)]

        # Time-space risk scores
        with open("model_artifacts/time_space_mappings.pkl", "rb") as f:
            time_space_mappings = pickle.load(f)

        # District-based risk scores
        data["TimeLocationRisk"] = [
            time_space_mappings["time_space_risk"].get((district, time_period_str), 0.1)
        ]
        data["WeekendLocationRisk"] = [
            time_space_mappings["weekend_loc_risk"].get((district, is_weekend), 0.1)
        ]

        # Grid-based risk scores
        with open("model_artifacts/grid_time_mappings.pkl", "rb") as f:
            grid_time_mappings = pickle.load(f)

        data["GridTimeRisk"] = [
            grid_time_mappings["grid_time_risk"].get((grid_cell, time_period_str), 0.1)
        ]
        data["GridWeekendRisk"] = [
            grid_time_mappings["grid_weekend_risk"].get((grid_cell, is_weekend), 0.1)
        ]

    except FileNotFoundError:
        # If mappings don't exist, use reasonable defaults
        data["CrimeDensity"] = [0.1]
        data["TimeLocationRisk"] = [0.1]
        data["WeekendLocationRisk"] = [0.1]
        data["GridTimeRisk"] = [0.1]
        data["GridWeekendRisk"] = [0.1]

        # Add category density columns if in feature names
        for col in feature_names:
            if col.startswith("Density_") and col not in data:
                data[col] = [0.1]

    # Add TF-IDF SVD features
    expected_tfidf_features = sum(
        1 for col in feature_names if col.startswith("tfidf_Descript_")
    )
    for i in range(min(tfidf_svd.shape[1], expected_tfidf_features)):
        data[f"tfidf_Descript_{i}"] = [tfidf_svd[0, i]]

    # Fill in any remaining tfidf features with zeros
    for i in range(tfidf_svd.shape[1], expected_tfidf_features):
        data[f"tfidf_Descript_{i}"] = [0.0]

    # Create DataFrame
    X = pd.DataFrame(data)

    # Add any missing columns from feature_names
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0

    # Ensure we only have the required columns in the right order
    X = X[feature_names]

    return X


def extract_coordinates(coordinates_string):
    """
    Extract longitude and latitude from a coordinates string
    Expected format: "(latitude, longitude)" or similar
    Returns a tuple of (longitude, latitude) or None if parsing fails
    """
    if not coordinates_string or not isinstance(coordinates_string, str):
        return None

    # Try to extract numbers from the string
    try:
        # Remove parentheses and split by comma
        cleaned = coordinates_string.strip().strip("()").split(",")
        if len(cleaned) >= 2:
            # Convert to float and return as (longitude, latitude)
            # Note: Our convention is (longitude, latitude) for calculations
            latitude = float(cleaned[0].strip())
            longitude = float(cleaned[1].strip())
            return (longitude, latitude)
    except (ValueError, IndexError):
        st.warning(f"Could not parse coordinates: {coordinates_string}")
        return None

    return None


def make_prediction(
    description, date, district, coordinates_string, model, tfidf, svd, metadata
):
    """Make a prediction using the model with enhanced spatial features"""
    try:
        # Extract coordinates
        coordinates = extract_coordinates(coordinates_string)

        # Process input
        X = process_input_for_prediction(
            description, date, district, coordinates, model, tfidf, svd, metadata
        )

        # Explicitly check the shape of X
        if X.shape[1] != len(metadata["feature_names"]):
            st.error(
                f"Feature mismatch: Got {X.shape[1]} features, expected {len(metadata['feature_names'])}"
            )
            return "Error", {}

        # Convert categorical features to proper type
        cat_features = metadata["categorical_features"]

        # Create Pool for CatBoost
        pool = Pool(X, cat_features=cat_features)

        # Get prediction and probabilities
        prediction = model.predict(pool)[0]
        probabilities = model.predict_proba(pool)[0]

        # Get class names
        class_names = model.classes_

        # Create dictionary of class probabilities
        proba_dict = {
            str(class_names[i]): float(probabilities[i])
            for i in range(len(class_names))
        }

        return str(prediction), proba_dict

    except Exception as e:
        import traceback

        st.error(f"Error making prediction: {e}")
        st.code(traceback.format_exc())
        return "Error", {}


def display_map(data, category_filter=None, time_filter=None, district_filter=None):
    filtered_data = data.copy()

    if category_filter:
        filtered_data = filtered_data[filtered_data["Category"].isin(category_filter)]
    if time_filter:
        start_date, end_date = time_filter
        filtered_data = filtered_data[
            (filtered_data["Dates"] >= start_date)
            & (filtered_data["Dates"] <= end_date)
        ]
    if district_filter:
        filtered_data = filtered_data[filtered_data["PdDistrict"].isin(district_filter)]

    filtered_data = filtered_data.dropna(subset=["Category", "lon", "lat"])
    if filtered_data.empty:
        st.warning("No data available for selected filters.")
        return

    categories = filtered_data["Category"].unique()
    colors = px.colors.qualitative.Plotly[: len(categories)]
    color_map = {cat: color for cat, color in zip(categories, colors)}

    # Ensure a default color for unmapped categories
    default_hex_color = "#CCCCCC"  # grey

    filtered_data["color"] = filtered_data["Category"].map(
        lambda x: color_map.get(x, default_hex_color)
    )

    filtered_data["color"] = filtered_data["color"].apply(
        lambda x: (
            list(hex_to_rgb(x)) + [200] if isinstance(x, str) else [200, 200, 200, 200]
        )
    )

    view_state = pdk.ViewState(latitude=37.7749, longitude=-122.4194, zoom=11)

    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=filtered_data,
        get_position=["lon", "lat"],
        get_fill_color="color",
        get_radius=50,
        pickable=True,
        opacity=0.8,
    )

    heatmap_layer = pdk.Layer(
        "HeatmapLayer",
        data=filtered_data,
        get_position=["lon", "lat"],
        get_weight=1,
        opacity=0.6,
    )

    layers = []
    if show_points:
        layers.append(scatter_layer)
    if show_heatmap:
        layers.append(heatmap_layer)

    deck = pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=view_state,
        layers=layers,
        tooltip={
            "html": "<b>Category:</b> {Category}<br><b>Description:</b> {Descript}<br><b>Date:</b> {Dates}",
            "style": {"color": "white"},
        },
    )

    st.pydeck_chart(deck)

    # Legend
    st.subheader("Categories Legend")
    legend_cols = st.columns(len(color_map))
    for i, (cat, color) in enumerate(color_map.items()):
        with legend_cols[i]:
            rgb_color = hex_to_rgb(color)
            st.markdown(
                f"<div style='background-color: rgb{rgb_color}; height:20px; width:20px; display:inline-block; margin-right:5px;'></div> {cat}",
                unsafe_allow_html=True,
            )


def display_crime_statistics(
    data, category_filter=None, time_filter=None, district_filter=None
):
    """Display crime statistics"""
    # Filter data
    filtered_data = data.copy()
    # st.write("Debug - Crime Statistics: Original shape:", filtered_data.shape)

    if category_filter:
        filtered_data = filtered_data[filtered_data["Category"].isin(category_filter)]
        # st.write(
        #     "Debug - Crime Statistics: After Category filter, shape:",
        #     filtered_data.shape,
        # )

    if time_filter:
        start_date, end_date = time_filter
        filtered_data = filtered_data[
            (filtered_data["Dates"] >= start_date)
            & (filtered_data["Dates"] <= end_date)
        ]
        # st.write(
        #     "Debug - Crime Statistics: After Time filter, shape:", filtered_data.shape
        # )

    if district_filter:
        filtered_data = filtered_data[filtered_data["PdDistrict"].isin(district_filter)]
        # st.write(
        #     "Debug - Crime Statistics: After District filter, shape:",
        #     filtered_data.shape,
        # )

    # Display basic statistics
    st.subheader("Crime Statistics")

    col1, col2 = st.columns(2)

    with col1:
        # Crime by category
        st.write("Crimes by Category")
        category_counts = filtered_data["Category"].value_counts().reset_index()
        category_counts.columns = ["Category", "Count"]
        fig = px.bar(
            category_counts,
            x="Count",
            y="Category",
            orientation="h",
            title="Number of Crimes by Category",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Crimes by district
        st.write("Crimes by District")
        district_counts = filtered_data["PdDistrict"].value_counts().reset_index()
        district_counts.columns = ["District", "Count"]
        fig = px.bar(
            district_counts,
            x="Count",
            y="District",
            orientation="h",
            title="Number of Crimes by District",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Crimes over time
    st.write("Crimes Over Time")
    time_options = ["Day", "Week", "Month", "Year"]
    selected_time = st.selectbox("Time Granularity", time_options)

    if selected_time == "Day":
        filtered_data["TimeGroup"] = filtered_data["Dates"].dt.date
    elif selected_time == "Week":
        filtered_data["TimeGroup"] = (
            filtered_data["Dates"].dt.to_period("W").apply(lambda x: x.start_time)
        )
    elif selected_time == "Month":
        filtered_data["TimeGroup"] = (
            filtered_data["Dates"].dt.to_period("M").apply(lambda x: x.start_time)
        )
    else:  # Year
        filtered_data["TimeGroup"] = filtered_data["Dates"].dt.year

    time_counts = (
        filtered_data.groupby(["TimeGroup", "Category"])
        .size()
        .reset_index(name="Count")
    )

    fig = px.line(
        time_counts,
        x="TimeGroup",
        y="Count",
        color="Category",
        title=f"Crimes by {selected_time}",
    )
    st.plotly_chart(fig, use_container_width=True)


# Main application
st.title("🔍 CityX Crime Classification Dashboard")

# Load model artifacts
model, tfidf, svd, metadata = get_model_artifacts()

if model is None:
    st.error("Failed to load model. Please check if model artifacts exist.")
    st.stop()

# Load crime data
data = load_crime_data()

# Log the data columns to the console
print("Data Columns:", data.columns.tolist())

# Add this to initialize session state variables if needed
if "use_gemini" not in st.session_state:
    st.session_state["use_gemini"] = True
if "show_formatted" not in st.session_state:
    st.session_state["show_formatted"] = True

# Sidebar for filters and navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Map Visualization", "Crime Prediction", "Crime Statistics", "PDF Upload"],
)

# Filters in sidebar
st.sidebar.title("Filters")

# Date filter
min_date = data["Dates"].min()
max_date = data["Dates"].max()
time_filter = st.sidebar.date_input(
    "Date Range", value=[min_date, max_date], min_value=min_date, max_value=max_date
)

# Convert to datetime for filtering
if len(time_filter) == 2:
    time_filter = [pd.to_datetime(date) for date in time_filter]
else:
    time_filter = None

# Category filter
categories = sorted(data["Category"].unique())
category_filter = st.sidebar.multiselect(
    "Crime Categories", options=categories, default=[]
)

# District filter
districts = sorted(data["PdDistrict"].unique())
district_filter = st.sidebar.multiselect(
    "Police Districts", options=districts, default=[]
)

# Map display options
st.sidebar.title("Map Options")
show_points = st.sidebar.checkbox("Show Points", value=True)
show_heatmap = st.sidebar.checkbox("Show Heatmap", value=False)

# Add this to your sidebar
st.sidebar.title("Advanced Options")

# Add toggles for Gemini API and format display
use_gemini = st.sidebar.checkbox(
    "Use Gemini for Description Processing",
    value=st.session_state["use_gemini"],
    help="Use Google's Gemini API to convert descriptions to standardized police report format",
)
st.session_state["use_gemini"] = use_gemini

show_formatted = st.sidebar.checkbox(
    "Show Formatted Description",
    value=st.session_state["show_formatted"],
    help="Display the formatted description used for prediction",
)
st.session_state["show_formatted"] = show_formatted

# If Gemini is enabled, check if API key is available
if use_gemini:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.sidebar.warning(
            "⚠️ Gemini API key not found. Add GEMINI_API_KEY to your .env file."
        )
    else:
        st.sidebar.success("✅ Gemini API key found.")

# Display the appropriate page
if page == "Map Visualization":
    st.header("Crime Map")
    st.markdown(
        """
    This map shows the locations of crimes in CityX. 
    Use the filters in the sidebar to narrow down the data.
    """
    )

    display_map(data, category_filter, time_filter, district_filter)

    # Show statistics below the map
    display_crime_statistics(data, category_filter, time_filter, district_filter)

elif page == "Crime Prediction":
    st.header("Crime Category Prediction")
    st.markdown(
        """
    Enter a crime description and other details to predict the crime category. 
    The model will classify the crime based on the text description, temporal, and spatial features.
    """
    )

    # Input form
    with st.form("prediction_form"):
        description = st.text_area(
            "Crime Description",
            "Suspect broke car window and stole laptop from the back seat.",
        )
        date = st.date_input("Date", value=datetime.datetime.now())
        time = st.time_input("Time", value=datetime.datetime.now().time())
        district = st.selectbox("Police District", options=districts)

        # Add coordinates input
        col1, col2 = st.columns(2)
        with col1:
            latitude = st.number_input(
                "Latitude",
                value=37.7749,
                min_value=37.70,
                max_value=37.85,
                format="%.6f",
            )
        with col2:
            longitude = st.number_input(
                "Longitude",
                value=-122.4194,
                min_value=-122.55,
                max_value=-122.35,
                format="%.6f",
            )

        coordinates_string = f"({latitude}, {longitude})"

        # Submit button
        submitted = st.form_submit_button("Predict Crime Category")

    if submitted:
        with st.spinner("Making prediction..."):
            # Combine date and time
            datetime_input = pd.to_datetime(f"{date} {time}")

            # Call prediction function with coordinates
            # Combine date and time
            if st.session_state.get("use_gemini", False):
                # Use Gemini-powered prediction
                prediction, probabilities = make_prediction_with_gemini(
                    description,
                    datetime_input,
                    district,
                    coordinates_string,
                    model,
                    tfidf,
                    svd,
                    metadata,
                )
            else:
                # Use regular prediction
                prediction, probabilities = make_prediction(
                    description,
                    datetime_input,
                    district,
                    coordinates_string,
                    model,
                    tfidf,
                    svd,
                    metadata,
                )

            # Get the confidence (probability of the top prediction)
            confidence = max(probabilities.values()) if probabilities else 0
            confidence_pct = confidence * 100

            # Create columns for prediction and confidence
            col1, col2 = st.columns(2)

            with col1:
                # Display prediction with appropriate styling based on confidence
                st.success(f"Predicted Crime Category: **{prediction}**")

            with col2:
                # Display confidence with color coding
                if confidence_pct >= 80:
                    st.success(f"Confidence: **{confidence_pct:.1f}%**")
                elif confidence_pct >= 50:
                    st.warning(f"Confidence: **{confidence_pct:.1f}%**")
                else:
                    st.error(f"Confidence: **{confidence_pct:.1f}%**")

            # Add confidence meter
            st.progress(confidence)

            if confidence_pct < 50:
                st.warning(
                    "Low confidence prediction. Consider adding more details to the description or checking location information."
                )

            # Display probabilities
            st.subheader("Probability by Category")

            # Sort probabilities
            sorted_probs = sorted(
                probabilities.items(), key=lambda x: x[1], reverse=True
            )

            # Create DataFrame for plotting
            prob_df = pd.DataFrame(
                sorted_probs[:5], columns=["Category", "Probability"]
            )

            # Add percentage column for display
            prob_df["Percentage"] = prob_df["Probability"].apply(
                lambda x: f"{x*100:.1f}%"
            )

            # Create bar chart
            fig = px.bar(
                prob_df,
                x="Probability",
                y="Category",
                orientation="h",
                labels={"x": "Probability", "y": "Category"},
                title="Top 5 Category Probabilities",
                text="Percentage",
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

            # Display feature importance if available
            st.subheader("What influenced this prediction?")

            # Show info about the enhanced features
            st.markdown(
                """
                **Key factors influencing this prediction:**
                
                1. **Text Description:** The words and phrases in your description matched patterns in our training data
                2. **Location Factors:**
                   - Police district crime patterns
                   - Geographic location within CityX
                   - Crime density in the specific area
                3. **Time Factors:**
                   - Time of day (morning/afternoon/evening/night)
                   - Day of week (weekday vs weekend)
                   - Seasonal patterns
                
                The model combines these factors using both the text description and contextual features (time, location).
                """
            )

            # Show a map with the predicted location
            st.subheader("Crime Location")
            map_df = pd.DataFrame(
                {"lat": [latitude], "lon": [longitude], "category": [prediction]}
            )

            st.map(map_df)

elif page == "Crime Statistics":
    st.header("Crime Statistics")
    st.markdown(
        """
    This page shows statistics and trends about crimes in CityX.
    Use the filters in the sidebar to narrow down the data.
    """
    )

    display_crime_statistics(data, category_filter, time_filter, district_filter)

    # Additional statistics
    st.subheader("Time-based Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Crimes by day of week
        filtered_data = data.copy()
        if category_filter:
            filtered_data = filtered_data[
                filtered_data["Category"].isin(category_filter)
            ]
        if time_filter:
            filtered_data = filtered_data[
                (filtered_data["Dates"] >= time_filter[0])
                & (filtered_data["Dates"] <= time_filter[1])
            ]
        if district_filter:
            filtered_data = filtered_data[
                filtered_data["PdDistrict"].isin(district_filter)
            ]

        # Day of week analysis
        day_order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        filtered_data["DayOfWeek"] = pd.Categorical(
            filtered_data["DayOfWeek"], categories=day_order, ordered=True
        )
        day_counts = (
            filtered_data["DayOfWeek"].value_counts().reindex(day_order).reset_index()
        )
        day_counts.columns = ["Day", "Count"]

        fig = px.bar(day_counts, x="Day", y="Count", title="Crimes by Day of Week")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Crimes by hour of day
        filtered_data["Hour"] = filtered_data["Dates"].dt.hour
        hour_counts = filtered_data.groupby("Hour").size().reset_index(name="Count")

        fig = px.bar(hour_counts, x="Hour", y="Count", title="Crimes by Hour of Day")
        st.plotly_chart(fig, use_container_width=True)

elif page == "PDF Upload":
    st.header("Upload PDF Crime Report")
    st.markdown(
        """
        Upload one or more police crime report PDFs. The app will extract all labeled fields using pdfplumber.
        These fields include Report Number, Date & Time, Reporting Officer, Incident Location, Coordinates, 
        Detailed Description, Police District, Resolution, Suspect Description, and Victim Information.
        The extracted values will prefill the form below so you can review/edit them before triggering prediction.
        """
    )
    uploaded_file = st.file_uploader(
        "Upload PDF files", type=["pdf"], accept_multiple_files=False
    )

    # Initialize a dictionary to hold the extracted data (from the last processed PDF)
    extracted_data = {}

    if uploaded_file:
        # Process each uploaded PDF. (For simplicity, we use the last file's data to prefill the form.)
        st.subheader(f"Processing file: {uploaded_file.name}")
        try:
            with pdfplumber.open(uploaded_file) as pdf:
                full_text = ""
                for page_obj in pdf.pages:
                    full_text += page_obj.extract_text() + "\n"
            # Extract all labeled fields from the full text
            fields = extract_report_fields(full_text)

            # Create a combined description from Detailed, Suspect, and Victim sections
            combined_description = "\n".join(
                [
                    fields.get("detailed_description", ""),
                    fields.get("suspect_description", ""),
                    fields.get("victim_information", ""),
                ]
            ).strip()

            # Save all extracted fields into our dictionary (this example uses the last file's data)
            extracted_data = {
                "report_number": fields.get("report_number", ""),
                "date_time": fields.get("date_time", ""),
                "reporting_officer": fields.get("reporting_officer", ""),
                "incident_location": fields.get("incident_location", ""),
                "coordinates": fields.get("coordinates", ""),
                "detailed_description": fields.get("detailed_description", ""),
                "police_district": fields.get("police_district", ""),
                "resolution": fields.get("resolution", ""),
                "suspect_description": fields.get("suspect_description", ""),
                "victim_information": fields.get("victim_information", ""),
                "combined_description": combined_description,
            }

            st.markdown("**Extracted Fields:**")
            st.json(extracted_data)

            with st.expander("Show Full Extracted Text"):
                st.text(full_text)

        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {e}")

    if extracted_data:
        st.markdown("### Prediction from Extracted PDF Data")
        with st.form("pdf_prediction_form"):
            report_number = st.text_input(
                "Report Number", value=extracted_data.get("report_number", "")
            )
            date_time_str = st.text_input(
                "Date & Time",
                value=extracted_data.get("date_time", str(datetime.datetime.now())),
            )
            reporting_officer = st.text_input(
                "Reporting Officer", value=extracted_data.get("reporting_officer", "")
            )
            incident_location = st.text_input(
                "Incident Location", value=extracted_data.get("incident_location", "")
            )
            coordinates = st.text_input(
                "Coordinates", value=extracted_data.get("coordinates", "")
            )

            # Add coordinate adjustment fields if coordinates exist
            coordinates_parsed = extract_coordinates(coordinates)
            if coordinates_parsed:
                st.success("Coordinates successfully parsed!")
                longitude, latitude = coordinates_parsed
                col1, col2 = st.columns(2)
                with col1:
                    latitude = st.number_input(
                        "Latitude", value=latitude, format="%.6f"
                    )
                with col2:
                    longitude = st.number_input(
                        "Longitude", value=longitude, format="%.6f"
                    )

                # Update coordinates string with the adjusted values
                coordinates = f"({latitude}, {longitude})"
            else:
                st.warning(
                    "Coordinates could not be parsed or were not provided. Using default location (CityX center)."
                )
                col1, col2 = st.columns(2)
                with col1:
                    latitude = st.number_input("Latitude", value=37.7749, format="%.6f")
                with col2:
                    longitude = st.number_input(
                        "Longitude", value=-122.4194, format="%.6f"
                    )

                # Create coordinates string
                coordinates = f"({latitude}, {longitude})"

            detailed_description = st.text_area(
                "Detailed Description",
                value=extracted_data.get("detailed_description", ""),
                height=150,
            )
            suspect_description = st.text_area(
                "Suspect Description",
                value=extracted_data.get("suspect_description", ""),
                height=100,
            )
            victim_information = st.text_area(
                "Victim Information",
                value=extracted_data.get("victim_information", ""),
                height=100,
            )
            combined_description = st.text_area(
                "Combined Description (for Prediction)",
                value=extracted_data.get("combined_description", ""),
                height=200,
            )

            # Standardize district selection
            district_value = extracted_data.get("police_district", "").upper().strip()
            district_index = 0
            for i, d in enumerate(districts):
                if d.upper() == district_value:
                    district_index = i
                    break

            police_district = st.selectbox(
                "Police District", options=districts, index=district_index
            )

            # If date_time_str can be parsed, use it; otherwise default to now.
            try:
                dt = pd.to_datetime(date_time_str)
            except Exception:
                dt = datetime.datetime.now()

            date = st.date_input("Date", value=dt.date())
            time = st.time_input("Time", value=dt.time())
            datetime_input = pd.to_datetime(f"{date} {time}")

            submitted = st.form_submit_button("Predict Crime Category")

        if submitted:
            with st.spinner("Making prediction..."):
                # Use the combined description and coordinates for prediction
                if st.session_state.get("use_gemini", False):
                    # Use Gemini-powered prediction
                    prediction, probabilities = make_prediction_with_gemini(
                        combined_description,
                        datetime_input,
                        police_district,
                        coordinates,
                        model,
                        tfidf,
                        svd,
                        metadata,
                    )
                else:
                    # Use regular prediction
                    prediction, probabilities = make_prediction(
                        combined_description,
                        datetime_input,
                        police_district,
                        coordinates,
                        model,
                        tfidf,
                        svd,
                        metadata,
                    )

                # Get the confidence (probability of the top prediction)
                confidence = max(probabilities.values()) if probabilities else 0
                confidence_pct = confidence * 100

                # Create columns for prediction and confidence
                col1, col2 = st.columns(2)

                with col1:
                    # Display prediction with appropriate styling
                    st.success(f"Predicted Crime Category: **{prediction}**")

                with col2:
                    # Display confidence with color coding
                    if confidence_pct >= 80:
                        st.success(f"Confidence: **{confidence_pct:.1f}%**")
                    elif confidence_pct >= 50:
                        st.warning(f"Confidence: **{confidence_pct:.1f}%**")
                    else:
                        st.error(f"Confidence: **{confidence_pct:.1f}%**")

                # Add confidence meter
                st.progress(confidence)

                if confidence_pct < 50:
                    st.warning(
                        "Low confidence prediction. Consider adding more details to the description or reviewing extracted data."
                    )

                # Display probabilities
                st.subheader("Probability by Category")
                sorted_probs = sorted(
                    probabilities.items(), key=lambda x: x[1], reverse=True
                )
                prob_df = pd.DataFrame(
                    sorted_probs[:5], columns=["Category", "Probability"]
                )

                # Add percentage column for display
                prob_df["Percentage"] = prob_df["Probability"].apply(
                    lambda x: f"{x*100:.1f}%"
                )

                fig = px.bar(
                    prob_df,
                    x="Probability",
                    y="Category",
                    orientation="h",
                    labels={"x": "Probability", "y": "Category"},
                    title="Top 5 Category Probabilities",
                    text="Percentage",
                )
                fig.update_traces(textposition="outside")
                st.plotly_chart(fig, use_container_width=True)

                # Show what influenced the prediction
                st.subheader("What influenced this prediction?")
                st.markdown(
                    """
                    **Key factors influencing this prediction:**
                    
                    1. **Text Description:** The words and phrases in your description matched patterns in our training data
                    2. **Location Factors:**
                       - Police district crime patterns
                       - Geographic location within CityX
                       - Crime density in the specific area
                    3. **Time Factors:**
                       - Time of day (morning/afternoon/evening/night)
                       - Day of week (weekday vs weekend)
                       - Seasonal patterns
                    
                    The model combines these factors using both the text description and contextual features (time, location).
                    """
                )

                # Show a map with the crime location
                st.subheader("Crime Location")

                # Get lat/lon from coordinates
                coords = extract_coordinates(coordinates)
                if coords:
                    longitude, latitude = coords
                    map_df = pd.DataFrame(
                        {
                            "lat": [latitude],
                            "lon": [longitude],
                            "category": [prediction],
                        }
                    )
                    st.map(map_df)
                else:
                    st.warning(
                        "Could not visualize location: coordinates not provided or could not be parsed."
                    )


# Footer
st.markdown("---")
st.markdown("© 2025 Crime Classification Dashboard | Powered by ML")
