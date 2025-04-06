# Crime Classification: End-to-End Documentation

Welcome to the **Crime Classification** project! This documentation walks you through the entire journey:

- **Data Exploration & Visualization**
- **Feature Engineering** (including NLP, spatial, and temporal features)
- **Model Training & Evaluation** using CatBoost
- **MLflow Experiment Tracking**
- **Interactive Streamlit Dashboard** for real-time crime classification and data insights

By the end of this guide, you should have a clear understanding of how the data was processed, how the final model was trained, and how the **Streamlit** app integrates with the model (including advanced features like PDF parsing and integration with Google's Gemini for text processing).

---

## Table of Contents

- [Crime Classification: End-to-End Documentation](#crime-classification-end-to-end-documentation)
  - [Table of Contents](#table-of-contents)
  - [1. Introduction \& Motivation](#1-introduction--motivation)
  - [2. Data Exploration](#2-data-exploration)
  - [3. Data Visualization \& Observation](#3-data-visualization--observation)
  - [4. Feature Engineering](#4-feature-engineering)
    - [4.1 Text Preprocessing (NLP)](#41-text-preprocessing-nlp)
    - [4.2 Spatial Features](#42-spatial-features)
    - [4.3 Temporal Features](#43-temporal-features)
    - [4.4 Combining Features](#44-combining-features)
  - [5. Model Training \& Evaluation](#5-model-training--evaluation)
    - [5.1 Training Pipeline Overview](#51-training-pipeline-overview)
    - [5.2 Performance Evaluation](#52-performance-evaluation)
    - [5.3 Feature Importance Analysis](#53-feature-importance-analysis)
  - [6. MLflow Integration \& Experiment Tracking](#6-mlflow-integration--experiment-tracking)
    - [6.1 Logging Artifacts](#61-logging-artifacts)
    - [6.2 Metadata \& Environment Tracking](#62-metadata--environment-tracking)
  - [7. Interactive Streamlit Application](#7-interactive-streamlit-application)
    - [7.1 Overview of the Streamlit App](#71-overview-of-the-streamlit-app)
    - [7.2 Key Features \& User Flow](#72-key-features--user-flow)
    - [7.3 Integration with Model \& Gemini API](#73-integration-with-model--gemini-api)
    - [7.4 PDF Upload \& Crime Report Extraction](#74-pdf-upload--crime-report-extraction)
    - [7.5 Running the Application](#75-running-the-application)
  - [8. Conclusion \& Future Work](#8-conclusion--future-work)

---

## 1. Introduction & Motivation

**Project Overview**  
This project aims to classify crime occurrences in San Francisco (or "CityX") based on:

- **Textual Descriptions** (`Descript` or PDF-based text)
- **Spatial Information** (longitude, latitude, police district)
- **Temporal Aspects** (date, time-of-day, day-of-week patterns)

By effectively **engineering these features**, I built a robust classification model that distinguishes between multiple crime categories. Finally, I made an **interactive dashboard** that allows both data exploration and real-time predictions, including advanced functionalities like **PDF parsing** of crime reports and **integration with Google's Gemini** for text rephrasing.

**Objectives**:

1. **Explore and visualize** the crime dataset to uncover relevant patterns.
2. **Engineer features** (NLP, spatial, temporal) to enhance classification performance.
3. **Train and evaluate** a multi-class CatBoost model.
4. **Track experiments** with MLflow for reproducibility and transparency.
5. **Deploy a Streamlit dashboard** to demonstrate real-time crime prediction and data insights.

---

## 2. Data Exploration

I begin by loading the **`Competition_Dataset.csv`** into a pandas `DataFrame`:

```python
import pandas as pd
data = pd.read_csv("Competition_Dataset.csv")
```

**Initial Checks**:

- **Shape**: `data.shape` to see the number of rows and columns.
- **Columns**: `data.columns` to list available features.
- **Quick Inspect**: `data.head()` and `data.info()` to observe data types, non-null counts, etc.
- **Missing Values**: Identify columns that might contain NaNs and assess if/where imputation is needed.

Example output might confirm columns such as:

```json
[
    "Dates",
    "Category",
    "Descript",
    "DayOfWeek",
    "PdDistrict",
    "Resolution",
    "Longitude (X)",
    "Latitude (Y)"
]
```

---

## 3. Data Visualization & Observation

I visualized the data to gain crucial insights:

1. **Category Distribution**  
   - A bar chart of `Category` frequencies to see which crimes are most common.
2. **Temporal Patterns**  
   - Charts of crime counts by day-of-week, hour-of-day, or month to spot periodic trends.
3. **Geospatial Plots**  
   - Scatter maps or heatmaps to see how crimes cluster in specific neighborhoods or near city landmarks.

**Takeaways**:

- Certain categories dominate the dataset, like **LARCENY/THEFT**.
- Crime rates vary by time of day, pointing to strong diurnal patterns.
- Spatial clustering around major districts suggests location-based features could be valuable.

---

## 4. Feature Engineering

### 4.1 Text Preprocessing (NLP)

Since crime descriptions in `Descript` provide valuable context, I applied a standard NLP pipeline:

1. **Text Cleaning**  
   - Lowercasing, punctuation removal, removing extra spaces, etc.

   ```python
   def preprocess_text(text):
       text = text.lower()
       text = re.sub(f"[{string.punctuation}]", " ", text)
       text = re.sub(r"\d+", "", text)
       text = re.sub(r"\s+", " ", text).strip()
       return text
   ```

2. **TF-IDF Transformation**  
   - Convert the processed text into numerical vectors.
   - Use `TfidfVectorizer` with `max_features=200`, unigrams & bigrams, and English stop words removed.

   ```python
   tfidf = TfidfVectorizer(max_features=200, stop_words="english", ngram_range=(1, 2))
   tfidf_matrix = tfidf.fit_transform(df["Descript_processed"])
   ```

3. **Dimensionality Reduction**  
   - Apply `TruncatedSVD(n_components=50)` to capture the main latent topics without exploding feature space.

4. **Appending NLP Features**  
   - The resulting 50 SVD components become new columns: `tfidf_Descript_0` … `tfidf_Descript_49`.
   - Top terms per component are logged for interpretability.

### 4.2 Spatial Features

Crime incidence often depends on **location**. I engineered:

1. **Grid Cell Assignment**  
   - Divided SF into a 5×5 grid (25 cells).  
   - Each record gets a `GridCell` ID (0–24) based on `(Longitude, Latitude)`.

2. **Crime Density**  
   - For each `GridCell`, computed total crimes and normalized by the maximum count, storing in `CrimeDensity`.
   - Category-specific density columns, e.g. `Density_THEFT`, highlight how concentrated each crime type is in each grid cell.

3. **Time-Space Interactions**  
   - For each `(district, time_period)` or `(grid_cell, time_period)`, I computed a **risk score** (how frequently crimes occur in that spatio-temporal slice).
   - **`TimeLocationRisk`, `GridTimeRisk`, `WeekendLocationRisk`, GridWeekendRisk`** columns capture these interactions.

4. **Distance from City Center**  
   - Euclidean distance to a reference point `(-122.4194, 37.7749)`, stored as `DistanceFromCenter`.

### 4.3 Temporal Features

1. **Extract Hour, Day, Month, DayOfWeek** from `Dates`.
2. **Weekend Flag** to distinguish weekends vs. weekdays.
3. **TimePeriod** (0=Morning, 1=Afternoon, 2=Evening, 3=Night) to capture daily crime cycles.

### 4.4 Combining Features

A master function (e.g., `process_data()`) was created to:

- Run textual cleaning + TF-IDF + SVD (with already-fitted models).
- Construct or look up advanced spatial features if `X`/`Y` columns exist.
- Build a final numeric feature set used by the model.

---

## 5. Model Training & Evaluation

### 5.1 Training Pipeline Overview

The **training script** `train.py` demonstrates the pipeline:

1. **Load & Split Data**  

   ```python
   train, test = train_test_split(data, test_size=0.2, stratify=data["Category"])
   ```

2. **NLP Feature Extraction**  

   ```python
   train_nlp, tfidf_vectorizer, svd_transformer, feature_names = \
       extract_nlp_features(train, "Descript", max_features=200, n_components=50)
   ```

3. **Feature Processing**  

   ```python
   X_full = process_data(train_nlp, tfidf_vectorizer, svd_transformer)
   y_full = train_nlp["Category"]
   X_test = process_data(test, tfidf_vectorizer, svd_transformer)
   ```

4. **Second Split** (Train/Val):

   ```python
   X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, test_size=0.2, stratify=y_full)
   ```

5. **Define CatBoost & Fit**  

   ```python
   model_params = {
       "iterations": 1000, "learning_rate": 0.05, "depth": 8,
       "l2_leaf_reg": 3, "loss_function": "MultiClass", "eval_metric": "TotalF1",
       "random_seed": 42
   }
   model = CatBoostClassifier(**model_params, task_type="GPU")
   model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
   ```

6. **Evaluate** using F1 (macro, weighted) and Accuracy.

### 5.2 Performance Evaluation

- **Accuracy**  

  ```python
  accuracy = accuracy_score(y_val, val_predictions)
  ```

- **Macro F1**  

  ```python
  macro_f1 = f1_score(y_val, val_predictions, average="macro")
  ```

- **Weighted F1**  

  ```python
  weighted_f1 = f1_score(y_val, val_predictions, average="weighted")
  ```

Metrics are logged to MLflow for easy comparison across experiment runs.

### 5.3 Feature Importance Analysis

Using CatBoost's **`get_feature_importance()`**:

```python
importance = model.get_feature_importance()
```

I plotted a bar chart where **NLP features** are highlighted in a different color. This reveals how textual info compares to spatial-temporal features in importance. Some top contributors often include time-of-day risk features, district-based risk, and certain TF-IDF components related to "theft," "vehicle," or "assault."

---

## 6. MLflow Integration & Experiment Tracking

### 6.1 Logging Artifacts

I leveraged MLflow for:

- **Parameters & Metrics**:

  ```python
  mlflow.log_params(model_params)
  mlflow.log_metrics({"accuracy": accuracy, "macro_f1": macro_f1, ...})
  ```

- **Plots**: Feature importance plot saved and logged as an artifact.
- **Model Checkpoint**:

  ```python
  mlflow.sklearn.log_model(model, "model", registered_model_name="crime_classification_model")
  ```

- **Preprocessing Objects**:  
  - TF-IDF vectorizer & SVD transformer pickled and logged.
- **Datasets**: Snapshots of train/test CSVs for reproducibility.

### 6.2 Metadata & Environment Tracking

- **JSON Metadata** capturing feature names, date of training, model run ID, etc.
- **Environment Files**: `requirements.txt` or pip freeze logs for environment reproducibility.

---

## 7. Interactive Streamlit Application

### 7.1 Overview of the Streamlit App

I built a **Streamlit** dashboard (`app.py`) to combine:

- Real-time **crime category predictions**.
- **Geospatial** and **temporal** interactive visualizations.
- PDF ingestion that **extracts key fields** from police reports.
- Optional **Google Gemini** integration to rephrase descriptions into standardized formats before prediction.

Users can navigate among four main sections:

1. **Map Visualization**
2. **Crime Prediction**
3. **Crime Statistics**
4. **PDF Upload**

### 7.2 Key Features & User Flow

1. **Map Visualization**  
   - Filters by date range, crime categories, and police districts.
   - Option to display points (ScatterPlotLayer) and a heatmap (HeatmapLayer) on a deck.gl map.
   - A legend to clarify color codes for different crime categories.
   - A date/time range filter for **temporal** exploration.

2. **Crime Prediction**  
   - Users enter a *crime description*, date/time, district, and coordinates.
   - The app processes these inputs (including advanced spatial features and text transformations) and returns:
     - **Predicted Crime Category**  
     - **Confidence** & top 5 probabilities  
     - A brief explanation of influential features (spatial/temporal/text).
     - A mini map showing the user's provided location.

3. **Crime Statistics**  
   - **Time-series** visualizations (line charts) aggregated by day, week, month, or year.
   - **Bar plots** showing distribution by Category and District.
   - Ability to refine the dataset using the same filters from the sidebar.

4. **PDF Upload**  
   - Allows uploading **PDF** crime reports (ex: official police documents).
   - Extracts labeled fields (Report Number, Date & Time, Reporting Officer, etc.) via `pdfplumber` and regex patterns.
   - Pre-fills a form with the extracted fields so the user can review/edit them, then run the crime prediction.

### 7.3 Integration with Model & Gemini API

1. **Model Artifacts Loading**  

   ```python
   model, tfidf, svd, metadata = get_model_artifacts()
   ```

   - Locates the "latest" model artifacts directory, loads the CatBoost model (`.cbm`), TF-IDF, SVD, and metadata.

2. **Prediction Functions**  
   - **`make_prediction()`**: The standard route that processes text, adds engineered features, and generates a prediction.
   - **`make_prediction_with_gemini()`**: If "Use Gemini" is enabled, the text is first **rephrased** into a standardized format using Google's Gemini. Then it is fed into the same feature pipeline.

3. **Google Gemini**  
   - Called via `rephrase_description_with_gemini()`.
   - Takes the user's raw description and prompts the Gemini API to convert it to a recognized "standardized" police format.
   - If the response does not exactly match a known format, fallback or retry logic is triggered.

### 7.4 PDF Upload & Crime Report Extraction

- Users upload a PDF.  
- **`pdfplumber`** extracts text from each page, concatenated into one string.  
- A set of **regex patterns** captures labeled fields, e.g.:
  - **Report Number:** `Report Number:\s*(.*?)\s*Date & Time:`
  - **Coordinates:** `Coordinates:\s*(\([^)]*\))\s*Detailed Description:`
- The relevant fields are displayed and can be edited before making a final prediction. Coordinates are parsed and (if needed) adjusted for the final location-based features.

### 7.5 Running the Application

**Prerequisites**:

- Python 3.8+  
- All required packages from `requirements.txt`.
- **Gemini API key** set as `GEMINI_API_KEY` in a `.env` file if you intend to use Gemini-based rephrasing.

**Steps to Launch**:

1. `pip install -r requirements.txt`  
2. Ensure your **model artifacts** are in `model_artifacts/latest` or another recognized subfolder.
3. Run `streamlit run app.py`.
4. Access the app via the local URL provided in the console (default: <http://localhost:8501>).

**Demo Workflow**:

1. **Map Visualization**: Explore the dataset interactively.
2. **Crime Prediction**: Paste a crime description, pick a date/time and location, then predict.
3. **Crime Statistics**: Dive deeper into the dataset's aggregated stats.
4. **PDF Upload**: Upload a sample PDF to see how the app extracts data and makes predictions.

---

## 8. Conclusion & Future Work

**Summary**  

- I explored the dataset, noting key spatial-temporal trends.
- I engineered advanced features—**NLP** (TF-IDF + SVD), **grid-based crime density**, **time-of-day risk**, etc.
- I trained a **CatBoost** classifier, logging everything with **MLflow**.
- The final **Streamlit** app provides real-time prediction, interactive mapping, PDF parsing, and an optional **Gemini** integration for text rephrasing into standardized crime formats.

**Lessons Learned**  

- Combining textual data with well-thought-out spatial and temporal features significantly boosts crime classification performance.
- Logging preprocessing objects (TF-IDF, SVD, feature mappings) is crucial for reproducibility.
- The advanced PDF ingestion highlights the potential to automate real-world police reporting workflows.

**Next Steps**  

1. **Further Tuning**: Systematic hyperparameter tuning or using advanced search methods (Optuna, Hyperopt) for CatBoost.  
2. **Extended Geospatial Modeling**: Possibly integrating more robust geospatial libraries or using clustering (DBSCAN, HDBSCAN) for more nuanced location patterns.  
3. **Deployment**: Dockerizing the Streamlit app or providing an AWS/GCP-based hosting solution.  
4. **Additional Data**: Incorporating external datasets, such as weather or demographic data, to uncover deeper crime correlations.

---

*This concludes the comprehensive documentation of the **Crime Classification** project, including data exploration, feature engineering, model training with MLflow, and the interactive Streamlit application.*  
