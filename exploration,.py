# Exploratory Data Analysis

# * Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# For example, count common words in descriptions:
from collections import Counter
import re

# * Suppress warnings
import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import geopandas as gp
import matplotlib.pyplot as plt
from shapely.geometry import Point
import matplotlib.cm as cm
import numpy as np

# * Set plotting styles
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

# * Load the dataset
df = pd.read_csv("Competition_Dataset.csv")
print(df.head())

## * General Preprocessing
# Convert 'Dates' column to datetime type
df["Dates"] = pd.to_datetime(df["Dates"], errors="coerce")

# Check if conversion was successful
print(df["Dates"].head())

# Remove any rows where date conversion failed
df = df.dropna(subset=["Dates"])

# Reset index after cleaning
df.reset_index(drop=True, inplace=True)

print("Cleaned dataset shape:", df.shape)

# Remove duplicates if any exist
df = df.drop_duplicates()

# Reset index after cleaning
df.reset_index(drop=True, inplace=True)

print("Cleaned dataset shape:", df.shape)

# Extract temporal features from the Dates column
df["Hour"] = df["Dates"].dt.hour
df["Day"] = df["Dates"].dt.day
df["Month"] = df["Dates"].dt.month
df["Year"] = df["Dates"].dt.year
df["DayOfWeek_Num"] = df["Dates"].dt.weekday  # Monday=0, Sunday=6


## * Feature Engineering for Temporal Features
# Create time period bins (morning, afternoon, evening, night)
def assign_time_period(hour):
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"


df["TimePeriod"] = df["Hour"].apply(assign_time_period)

# Check the newly created features
df[["Dates", "Hour", "Month", "Year", "DayOfWeek_Num", "TimePeriod"]].head()

## * Feature Engineering for Geographical Data
# Verify coordinate ranges and create a column for geohash clusters if needed
# For simplicity, we use rounding to group nearby coordinates (e.g., to 3 decimal places)
df["Lat_round"] = df["Latitude (Y)"].round(3)
df["Lon_round"] = df["Longitude (X)"].round(3)

# Create a combined geolocation cluster identifier
df["GeoCluster"] = df["Lat_round"].astype(str) + "_" + df["Lon_round"].astype(str)

# Check a few examples of the cluster IDs
df[["Latitude (Y)", "Longitude (X)", "GeoCluster"]].head()

## * Feature Engineering for Text Data
# Check a few examples of crime descriptions
print("Sample Crime Descriptions:")
print(df["Descript"].dropna().sample(5, random_state=42))

# Optional: Simple keyword extraction using string methods (more advanced analysis can be done with NLP libraries)
# Combine all descriptions into a single string and extract words
descriptions = " ".join(df["Descript"].dropna().tolist()).lower()
words = re.findall(r"\w+", descriptions)
word_counts = Counter(words)

# Display the 10 most common words
print("\nTop 10 common words in descriptions:")
print(word_counts.most_common(10))


## * Geospatial Analysis
# Load your neighborhood shapefile data
sftracts = gp.read_file("geo_export_43227ac5-047f-4fe8-aa84-8d60f1ff2159.shp")

# Create a GeoDataFrame from crime data with corrected coordinates
geometry = [Point(xy) for xy in zip(df["Latitude (Y)"], df["Longitude (X)"])]
crime_gdf = gp.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

# Ensure same CRS
sftracts = sftracts.to_crs("EPSG:4326")

# Create a spatial boundary for San Francisco
sf_boundary = sftracts.unary_union

# Use this boundary to filter points that are WITHIN San Francisco only
within_sf = crime_gdf.within(sf_boundary)
crime_gdf_filtered = crime_gdf[within_sf]

print(f"Points within SF boundaries: {len(crime_gdf_filtered)} out of {len(crime_gdf)}")

# Spatial join using the filtered crime data
joined = gp.sjoin(crime_gdf_filtered, sftracts, how="inner", predicate="within")

# Count crimes by neighborhood
crime_counts = joined.groupby("neighborho").size().reset_index(name="crime_count")
print("Top 5 neighborhoods by crime count:")
print(crime_counts.sort_values("crime_count", ascending=False).head())

# Join counts back to neighborhoods for choropleth
sftracts_with_counts = sftracts.merge(crime_counts, on="neighborho", how="left")
sftracts_with_counts["crime_count"] = sftracts_with_counts["crime_count"].fillna(0)

# Create a plot with both neighborhoods and crime density
fig, ax = plt.subplots(figsize=(15, 15))

# Plot neighborhoods with base color
sftracts.plot(ax=ax, edgecolor="black", linewidth=0.5, color="lightgray")

# Plot choropleth of crime counts
sftracts_with_counts.plot(
    column="crime_count",
    ax=ax,
    legend=True,
    cmap="OrRd",
    scheme="quantiles",
    k=5,
    alpha=0.6,
    edgecolor="black",
    linewidth=0.5,
    legend_kwds={"title": "Number of Crimes"},
)

# Add neighborhood labels for top crime areas only
top_n_neighborhoods = 10
top_neighborhoods = (
    crime_counts.sort_values("crime_count", ascending=False)
    .head(top_n_neighborhoods)["neighborho"]
    .tolist()
)

for idx, row in sftracts_with_counts.iterrows():
    if row["neighborho"] in top_neighborhoods:
        centroid = row.geometry.centroid
        ax.text(
            centroid.x,
            centroid.y,
            row["neighborho"],
            fontsize=8,
            ha="center",
            va="center",
            color="black",
            bbox=dict(facecolor="white", alpha=0.7, boxstyle="round,pad=0.1"),
        )

# Add a sample of crime points FROM THE FILTERED DATA
sample_size = min(3000, len(crime_gdf_filtered))
crime_sample = crime_gdf_filtered.sample(sample_size)
crime_sample.plot(ax=ax, markersize=0.5, color="blue", alpha=0.1)

# Set map boundaries to focus on San Francisco
ax.set_xlim([-122.52, -122.35])
ax.set_ylim([37.70, 37.83])

# Set title and labels
plt.title("Crime Density by San Francisco Neighborhood", fontsize=16)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.show()

# Now create category-specific maps
top_categories = df["Category"].value_counts().head(3).index.tolist()
print(f"Top 3 crime categories: {top_categories}")

fig, axes = plt.subplots(1, 3, figsize=(21, 7))

for i, category in enumerate(top_categories):
    # Filter crime data for this category - USE THE FILTERED DATA
    category_crimes = crime_gdf_filtered[crime_gdf_filtered["Category"] == category]

    # Spatial join for this category
    cat_joined = gp.sjoin(category_crimes, sftracts, how="inner", predicate="within")
    cat_counts = cat_joined.groupby("neighborho").size().reset_index(name="crime_count")

    # Join to neighborhoods
    cat_neighborhoods = sftracts.merge(cat_counts, on="neighborho", how="left")
    cat_neighborhoods["crime_count"] = cat_neighborhoods["crime_count"].fillna(0)

    # Plot on this subplot
    ax = axes[i]
    sftracts.plot(ax=ax, edgecolor="black", linewidth=0.5, color="lightgray")
    cat_neighborhoods.plot(
        column="crime_count",
        ax=ax,
        legend=True,
        cmap="OrRd",
        scheme="quantiles",
        k=5,
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
        legend_kwds={"title": f"{category} Crimes"},
    )

    # Set map boundaries consistently for each subplot
    ax.set_xlim([-122.52, -122.35])
    ax.set_ylim([37.70, 37.83])

    ax.set_title(f"{category} Crimes", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
