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
import pdfplumber  # New import for PDF processing


# Set page configuration
st.set_page_config(page_title="SF Crime Classifier", page_icon="🔍", layout="wide")


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


@st.cache_resource
def get_model_artifacts():
    """Cached function to load model artifacts"""
    # Find most recent model directory
    base_dir = "model_artifacts"
    if not os.path.exists(base_dir):
        st.error("Model artifacts directory not found!")
        return None, None, None, None

    # Get the most recent model directory
    subdirs = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]

    if not subdirs:
        st.error("No model directories found!")
        return None, None, None, None

    latest_dir = max(subdirs, key=os.path.getmtime)

    return load_model_artifacts(base_dir)


@st.cache_data
def load_crime_data():
    """Load and preprocess the crime dataset"""
    # Replace with your actual data file path
    try:
        df = pd.read_csv("Competition_Dataset.csv")
        # Convert dates to datetime
        df["Dates"] = pd.to_datetime(df["Dates"])
        # Extract coordinates if available
        if "Longitude (X)" in df.columns and "Latitude (Y)" in df.columns:
            df["lon"] = df["Longitude (X)"]
            df["lat"] = df["Latitude (Y)"]
        elif "X" in df.columns and "Y" in df.columns:
            df["lon"] = df["X"]
            df["lat"] = df["Y"]
        else:
            # Generate random coordinates for demo purposes
            st.warning(
                "No coordinates found in dataset, using random coordinates for visualization"
            )
            # San Francisco bounds
            df["lon"] = np.random.uniform(-122.51, -122.36, len(df))
            df["lat"] = np.random.uniform(37.71, 37.83, len(df))

        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Create sample data for demo
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


def process_input_for_prediction(description, date, district, coordinates, model, tfidf, svd, metadata):
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
        time_period_str = 'Morning'
    elif 12 <= date_dt.hour < 17:
        time_period = 1  # Afternoon
        time_period_str = 'Afternoon'
    elif 17 <= date_dt.hour < 21:
        time_period = 2  # Evening
        time_period_str = 'Evening'
    else:
        time_period = 3  # Night
        time_period_str = 'Night'
    data["TimePeriod"] = [time_period]
    
    # Weekend flag
    is_weekend = date_dt.weekday() >= 5

    # Categorical features
    data["DayOfWeek"] = [date_dt.weekday()]
    data["PdDistrict"] = [district]

    # Calculate basic spatial features if coordinates are provided
    city_center_x, city_center_y = -122.4194, 37.7749  # SF coordinates
    
    # San Francisco boundaries for grid calculation
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
        with open('model_artifacts/density_mappings.pkl', 'rb') as f:
            density_mappings = pickle.load(f)
            
        # Apply general crime density
        data['CrimeDensity'] = [density_mappings['grid_to_density'].get(grid_cell, 0.1)]
        
        # Apply category-specific densities if available
        if 'category_density' in density_mappings:
            for category, grid_to_cat_density in density_mappings['category_density'].items():
                col_name = f'Density_{category.replace("/", "_").replace(" ", "_")}'
                if col_name in feature_names:
                    data[col_name] = [grid_to_cat_density.get(grid_cell, 0.1)]
        
        # Time-space risk scores
        with open('model_artifacts/time_space_mappings.pkl', 'rb') as f:
            time_space_mappings = pickle.load(f)
            
        # District-based risk scores
        data['TimeLocationRisk'] = [time_space_mappings['time_space_risk'].get((district, time_period_str), 0.1)]
        data['WeekendLocationRisk'] = [time_space_mappings['weekend_loc_risk'].get((district, is_weekend), 0.1)]
        
        # Grid-based risk scores
        with open('model_artifacts/grid_time_mappings.pkl', 'rb') as f:
            grid_time_mappings = pickle.load(f)
            
        data['GridTimeRisk'] = [grid_time_mappings['grid_time_risk'].get((grid_cell, time_period_str), 0.1)]
        data['GridWeekendRisk'] = [grid_time_mappings['grid_weekend_risk'].get((grid_cell, is_weekend), 0.1)]
        
    except FileNotFoundError:
        # If mappings don't exist, use reasonable defaults
        data['CrimeDensity'] = [0.1]
        data['TimeLocationRisk'] = [0.1]
        data['WeekendLocationRisk'] = [0.1]
        data['GridTimeRisk'] = [0.1]
        data['GridWeekendRisk'] = [0.1]
        
        # Add category density columns if in feature names
        for col in feature_names:
            if col.startswith('Density_') and col not in data:
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


def make_prediction(description, date, district, model, tfidf, svd, metadata):
    """Make a prediction using the model"""
    try:
        # Process input
        X = process_input_for_prediction(
            description, date, district, model, tfidf, svd, metadata
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
    """Display crime data on a map"""
    # Filter data
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

    # Limit data points for performance
    if len(filtered_data) > 5000:
        st.warning(
            f"Dataset is large ({len(filtered_data)} points). Showing a random sample of 5000 points."
        )
        filtered_data = filtered_data.sample(5000)

    # Create color mapping
    categories = filtered_data["Category"].unique()
    colors = px.colors.qualitative.Plotly[: len(categories)]
    color_map = {cat: color for cat, color in zip(categories, colors)}

    # Create column for colors
    filtered_data["color"] = filtered_data["Category"].map(color_map)

    # Basic map using PyDeck
    view_state = pdk.ViewState(latitude=37.7749, longitude=-122.4194, zoom=11, pitch=0)

    # Create scatter plot layer
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=filtered_data,
        get_position=["lon", "lat"],
        get_color="color",
        get_radius=50,
        pickable=True,
        opacity=0.8,
    )

    # Create heatmap layer
    heatmap_layer = pdk.Layer(
        "HeatmapLayer",
        data=filtered_data,
        get_position=["lon", "lat"],
        get_weight=1,
        pickable=False,
        opacity=0.6,
        aggregation="SUM",
    )

    # Combine layers based on checkbox selection
    layers = []
    if show_points:
        layers.append(scatter_layer)
    if show_heatmap:
        layers.append(heatmap_layer)

    # Create deck
    deck = pdk.Deck(
        api_keys={
            "mapbox": "pk.eyJ1IjoiZ2hvdWxiaXRlciIsImEiOiJjbTh3N243OTIwMGEyMnJyMWdrMDYzdHRnIn0.WcRP7hZGvmHimcB4XDb1IA"
        },
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=view_state,
        layers=layers,
        tooltip={
            "html": "<b>Category:</b> {Category}<br><b>Description:</b> {Descript}<br><b>Date:</b> {Dates}",
            "style": {"color": "white"},
        },
    )

    # Display the map
    st.pydeck_chart(deck)

    # Display legend
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

    # Create time series at different granularities
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
st.title("🔍 San Francisco Crime Classification Dashboard")

# Load model artifacts
model, tfidf, svd, metadata = get_model_artifacts()

if model is None:
    st.error("Failed to load model. Please check if model artifacts exist.")
    st.stop()

# Load crime data
data = load_crime_data()

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

# Display the appropriate page
if page == "Map Visualization":
    st.header("Crime Map")
    st.markdown(
        """
    This map shows the locations of crimes in San Francisco. 
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
    The model will classify the crime based on the text description and provide probability estimates.
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

        # Combine date and time
        datetime_input = pd.to_datetime(f"{date} {time}")

        # Submit button
        submitted = st.form_submit_button("Predict Crime Category")

    if submitted:
        with st.spinner("Making prediction..."):
            # Add at the beginning of your make_prediction function
            # Call prediction function
            prediction, probabilities = make_prediction(
                description, datetime_input, district, model, tfidf, svd, metadata
            )

            # Display prediction
            st.success(f"Predicted Crime Category: **{prediction}**")

            # Display probabilities
            st.subheader("Probability by Category")

            # Sort probabilities
            sorted_probs = sorted(
                probabilities.items(), key=lambda x: x[1], reverse=True
            )

            # Create DataFrame for plotting - THIS IS THE FIX
            prob_df = pd.DataFrame(
                sorted_probs[:5], columns=["Category", "Probability"]
            )

            # Show top 5 categories
            # top_categories = [cat for cat, _ in sorted_probs[:5]]
            # top_probs = [prob for _, prob in sorted_probs[:5]]

            # Create bar chart
            fig = px.bar(
                # x=top_probs,
                # y=top_categories,
                prob_df,
                x="Probability",
                y="Category",
                orientation="h",
                labels={"x": "Probability", "y": "Category"},
                title="Top 5 Category Probabilities",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Display feature importance if available
            st.subheader("What influenced this prediction?")
            st.markdown(
                """
            The model uses both the text description and contextual features (time, location) to make predictions.
            Text features are particularly important for classification.
            """
            )

elif page == "Crime Statistics":
    st.header("Crime Statistics")
    st.markdown(
        """
    This page shows statistics and trends about crimes in San Francisco.
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
    uploaded_files = st.file_uploader(
        "Upload PDF files", type=["pdf"], accept_multiple_files=True
    )

    # Initialize a dictionary to hold the extracted data (from the last processed PDF)
    extracted_data = {}

    if uploaded_files:
        # Process each uploaded PDF. (For simplicity, we use the last file's data to prefill the form.)
        for uploaded_file in uploaded_files:
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
            police_district = st.selectbox(
                "Police District",
                options=districts,
                index=(
                    districts.index(extracted_data.get("police_district"))
                    if extracted_data.get("police_district") in districts
                    else 0
                ),
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
                # Use the combined description for prediction
                prediction, probabilities = make_prediction(
                    combined_description,
                    datetime_input,
                    police_district,
                    model,
                    tfidf,
                    svd,
                    metadata,
                )
                st.success(f"Predicted Crime Category: **{prediction}**")
                st.subheader("Probability by Category")
                sorted_probs = sorted(
                    probabilities.items(), key=lambda x: x[1], reverse=True
                )
                prob_df = pd.DataFrame(
                    sorted_probs[:5], columns=["Category", "Probability"]
                )
                fig = px.bar(
                    prob_df,
                    x="Probability",
                    y="Category",
                    orientation="h",
                    labels={"x": "Probability", "y": "Category"},
                    title="Top 5 Category Probabilities",
                )
                st.plotly_chart(fig, use_container_width=True)
                st.subheader("What influenced this prediction?")
                st.markdown(
                    """
                    The model uses both the text description and contextual features (time, location) to make predictions.
                    Text features are particularly important for classification.
                    """
                )


# Footer
st.markdown("---")
st.markdown("© 2025 Crime Classification Dashboard | Powered by ML")
