# Backend - FastAPI

# First, create a file called main.py

"""

# main.py - FastAPI backend for crime classification

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any, Union
import pandas as pd
import numpy as np
import os
import pickle
import json
from datetime import datetime, date, time
import re
import string
from catboost import CatBoostClassifier, Pool
import uvicorn

# Initialize FastAPI app

app = FastAPI(
    title="Crime Classification API",
    description="API for crime classification and data visualization",
    version="1.0.0"
)

# Allow CORS for frontend

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with actual frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models

class PredictionRequest(BaseModel):
    description: str
    date: str
    time: str
    district: str

class PredictionResponse(BaseModel):
    prediction: str
    probabilities: List[Dict[str, Union[str, float]]]

class FilterParams(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    categories: Optional[List[str]] = None
    districts: Optional[List[str]] = None

# Global variables

crime_data = None
model = None
tfidf_vectorizer = None
svd_transformer = None
metadata = None

# Helper functions

def preprocess_text(text):
    """Clean and preprocess text for NLP features"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", " ", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_model_artifacts():
    """Load model artifacts from disk"""
    global model, tfidf_vectorizer, svd_transformer, metadata

    # Find most recent model directory
    base_dir = "model_artifacts"
    if not os.path.exists(base_dir):
        raise FileNotFoundError("Model artifacts directory not found!")
        
    # Get the most recent model directory
    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) 
               if os.path.isdir(os.path.join(base_dir, d))]
    
    if not subdirs:
        raise FileNotFoundError("No model directories found!")
        
    latest_dir = max(subdirs, key=os.path.getmtime)
    
    # Load model
    model_path = os.path.join(latest_dir, "catboost_model.cbm")
    model = CatBoostClassifier()
    model.load_model(model_path)
    
    # Load TF-IDF vectorizer
    tfidf_path = os.path.join(latest_dir, "tfidf_vectorizer.pkl")
    with open(tfidf_path, "rb") as f:
        tfidf_vectorizer = pickle.load(f)
    
    # Load SVD transformer
    svd_path = os.path.join(latest_dir, "svd_transformer.pkl")
    with open(svd_path, "rb") as f:
        svd_transformer = pickle.load(f)
    
    # Load metadata
    metadata_path = os.path.join(latest_dir, "metadata.pkl")
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    
    return True

def load_crime_data():
    """Load crime dataset"""
    global crime_data

    try:
        # Replace with your actual file path
        df = pd.read_csv("Competition_Dataset.csv")
        
        # Convert dates to datetime
        df['Dates'] = pd.to_datetime(df['Dates'])
        
        # Extract coordinates
        if 'Longitude (X)' in df.columns and 'Latitude (Y)' in df.columns:
            df['lon'] = df['Longitude (X)']
            df['lat'] = df['Latitude (Y)']
        elif 'X' in df.columns and 'Y' in df.columns:
            df['lon'] = df['X']
            df['lat'] = df['Y']
        else:
            # Generate random coordinates for demo
            df['lon'] = np.random.uniform(-122.51, -122.36, len(df))
            df['lat'] = np.random.uniform(37.71, 37.83, len(df))
        
        crime_data = df
        return True
    except Exception as e:
        print(f"Error loading data: {e}")
        # Create demo data as fallback
        crime_data = create_demo_data()
        return True

def create_demo_data(n_samples=1000):
    """Create demo data if real data can't be loaded"""
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    categories = ['LARCENY/THEFT', 'OTHER OFFENSES', 'NON-CRIMINAL', 'ASSAULT',
                 'DRUG/NARCOTIC', 'VEHICLE THEFT', 'BURGLARY', 'VANDALISM']
    districts = ['CENTRAL', 'SOUTHERN', 'BAYVIEW', 'MISSION', 'PARK', 'RICHMOND',
                'INGLESIDE', 'TARAVAL', 'NORTHERN']
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    df = pd.DataFrame({
        'Dates': np.random.choice(dates, n_samples),
        'Category': np.random.choice(categories, n_samples),
        'Descript': ['Sample crime description'] * n_samples,
        'DayOfWeek': np.random.choice(days, n_samples),
        'PdDistrict': np.random.choice(districts, n_samples),
        'lon': np.random.uniform(-122.51, -122.36, n_samples),
        'lat': np.random.uniform(37.71, 37.83, n_samples)
    })
    
    return df

def process_for_prediction(description, date_time, district):
    """Process input for prediction"""
    # Feature names and categorical features
    feature_names = metadata['feature_names']
    categorical_features = metadata['categorical_features']

    # Create DataFrame with features
    X = pd.DataFrame(columns=feature_names)
    
    # Preprocess text for NLP features
    processed_text = preprocess_text(description)
    tfidf_matrix = tfidf_vectorizer.transform([processed_text])
    tfidf_svd = svd_transformer.transform(tfidf_matrix)
    
    # Add temporal features
    X['Hour'] = [date_time.hour]
    X['Day'] = [date_time.day]
    X['Month'] = [date_time.month]
    X['DayOfWeek_Num'] = [date_time.weekday()]
    X['WeekendFlag'] = [1 if date_time.weekday() >= 5 else 0]
    
    # Time period
    if 5 <= date_time.hour < 12:
        time_period = 0  # Morning
    elif 12 <= date_time.hour < 17:
        time_period = 1  # Afternoon
    elif 17 <= date_time.hour < 21:
        time_period = 2  # Evening
    else:
        time_period = 3  # Night
    X['TimePeriod'] = [time_period]
    
    # Categorical features
    X['DayOfWeek'] = [date_time.weekday()]
    X['PdDistrict'] = [district]
    
    # Default values for spatial features
    X['DistanceFromCenter'] = [0]
    X['QuadrantNS'] = [0]
    X['QuadrantEW'] = [0]
    
    # Add TF-IDF SVD features
    for i in range(tfidf_svd.shape[1]):
        if i < tfidf_svd.shape[1]:
            X[f'tfidf_Descript_{i}'] = [tfidf_svd[0, i]]
    
    # Fill any missing columns with 0
    for col in feature_names:
        if col not in X.columns:
            X[col] = [0]
    
    # Ensure columns are in correct order
    X = X[feature_names]
    
    return X

def make_prediction(description, date_str, time_str, district):
    """Make a prediction using the model"""
    try:
        # Parse date and time
        date_time = pd.to_datetime(f"{date_str} {time_str}")

        # Process input
        X = process_for_prediction(description, date_time, district)
        
        # Create Pool for CatBoost
        pool = Pool(X, cat_features=metadata['categorical_features'])
        
        # Get prediction and probabilities
        prediction = model.predict(pool)[0]
        probabilities = model.predict_proba(pool)[0]
        
        # Get class names
        class_names = model.classes_
        
        # Create dictionary of class probabilities
        proba_dict = [{"category": str(class_names[i]), "probability": float(probabilities[i])}
                     for i in range(len(class_names))]
        
        # Sort by probability (descending)
        proba_dict = sorted(proba_dict, key=lambda x: x["probability"], reverse=True)
        
        return str(prediction), proba_dict
        
    except Exception as e:
        print(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def filter_crime_data(filters: FilterParams):
    """Filter crime data based on parameters"""
    filtered_data = crime_data.copy()

    # Apply date filters
    if filters.start_date and filters.end_date:
        start_date = pd.to_datetime(filters.start_date)
        end_date = pd.to_datetime(filters.end_date)
        filtered_data = filtered_data[(filtered_data["Dates"] >= start_date) & 
                                    (filtered_data["Dates"] <= end_date)]
    
    # Apply category filter
    if filters.categories and len(filters.categories) > 0:
        filtered_data = filtered_data[filtered_data["Category"].isin(filters.categories)]
    
    # Apply district filter
    if filters.districts and len(filters.districts) > 0:
        filtered_data = filtered_data[filtered_data["PdDistrict"].isin(filters.districts)]
    
    return filtered_data

# Startup event

@app.on_event("startup")
async def startup_event():
    """Load model and data when the API starts"""
    try:
        load_model_artifacts()
        load_crime_data()
        print("Model and data loaded successfully!")
    except Exception as e:
        print(f"Error during startup: {e}")

# API endpoints

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "online", "message": "Crime Classification API is running"}

@app.get("/api/metadata")
async def get_metadata():
    """Get metadata about the dataset and model"""
    if crime_data is None or model is None:
        raise HTTPException(status_code=500, detail="Data or model not loaded")

    return {
        "total_crimes": len(crime_data),
        "date_range": {
            "start": crime_data["Dates"].min().strftime("%Y-%m-%d"),
            "end": crime_data["Dates"].max().strftime("%Y-%m-%d")
        },
        "categories": crime_data["Category"].unique().tolist(),
        "districts": crime_data["PdDistrict"].unique().tolist(),
        "model_info": {
            "type": "CatBoost Classifier",
            "features": len(metadata["feature_names"]) if metadata else 0
        }
    }

@app.post("/api/predict")
async def predict(request: PredictionRequest):
    """Make a crime category prediction"""
    if model is None or tfidf_vectorizer is None or svd_transformer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    prediction, probabilities = make_prediction(
        request.description, request.date, request.time, request.district
    )
    
    return {
        "prediction": prediction,
        "probabilities": probabilities
    }

@app.post("/api/crime_data")
async def get_crime_data(filters: FilterParams, limit: int = Query(1000, ge=1, le=5000)):
    """Get filtered crime data for visualization"""
    if crime_data is None:
        raise HTTPException(status_code=500, detail="Data not loaded")

    filtered_data = filter_crime_data(filters)
    
    # Limit data points for performance
    if len(filtered_data) > limit:
        filtered_data = filtered_data.sample(limit)
    
    # Format data for response
    result = []
    for _, row in filtered_data.iterrows():
        result.append({
            "id": int(row.name),
            "category": row["Category"],
            "description": row["Descript"] if "Descript" in row else "",
            "date": row["Dates"].strftime("%Y-%m-%d %H:%M:%S"),
            "day_of_week": row["DayOfWeek"] if "DayOfWeek" in row else "",
            "district": row["PdDistrict"],
            "location": {
                "lat": float(row["lat"]),
                "lng": float(row["lon"])
            }
        })
    
    return {
        "total": len(filtered_data),
        "limit": limit,
        "data": result
    }

@app.post("/api/crime_stats")
async def get_crime_stats(filters: FilterParams):
    """Get crime statistics for visualization"""
    if crime_data is None:
        raise HTTPException(status_code=500, detail="Data not loaded")

    filtered_data = filter_crime_data(filters)
    
    # Crime by category
    category_counts = filtered_data["Category"].value_counts().reset_index()
    category_counts.columns = ["category", "count"]
    category_stats = category_counts.to_dict(orient="records")
    
    # Crime by district
    district_counts = filtered_data["PdDistrict"].value_counts().reset_index()
    district_counts.columns = ["district", "count"]
    district_stats = district_counts.to_dict(orient="records")
    
    # Crime by time of day
    filtered_data["hour"] = filtered_data["Dates"].dt.hour
    hour_counts = filtered_data.groupby("hour").size().reset_index()
    hour_counts.columns = ["hour", "count"]
    time_stats = hour_counts.to_dict(orient="records")
    
    # Crime by day of week
    day_counts = filtered_data["DayOfWeek"].value_counts().reset_index()
    day_counts.columns = ["day", "count"]
    day_stats = day_counts.to_dict(orient="records")
    
    # Time series data (by month)
    filtered_data["month"] = filtered_data["Dates"].dt.to_period("M").astype(str)
    month_counts = filtered_data.groupby(["month", "Category"]).size().reset_index()
    month_counts.columns = ["month", "category", "count"]
    time_series = month_counts.to_dict(orient="records")
    
    return {
        "total_crimes": len(filtered_data),
        "by_category": category_stats,
        "by_district": district_stats,
        "by_hour": time_stats,
        "by_day": day_stats,
        "time_series": time_series
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
"""

# Now, create a frontend with React + Vite

# First, set up the project structure

"""

# Project setup instructions

# 1. Install Node.js and npm if not already installed

# 2. Create a new Vite project with React

npm create vite@latest crime-dashboard -- --template react-ts

# 3. Navigate to the project directory

cd crime-dashboard

# 4. Install dependencies

npm install axios react-router-dom mapbox-gl @mui/material @mui/icons-material @emotion/react @emotion/styled recharts date-fns

# 5. Install dev dependencies

npm install -D @types/mapbox-gl @types/react-router-dom

# 6. Create the project structure

mkdir -p src/components src/hooks src/pages src/services src/types src/utils
"""

# Create the frontend files

# 1. src/types/index.ts - Type definitions

"""
// src/types/index.ts

export interface Location {
  lat: number;
  lng: number;
}

export interface CrimeData {
  id: number;
  category: string;
  description: string;
  date: string;
  day_of_week: string;
  district: string;
  location: Location;
}

export interface FilterParams {
  start_date: string | null;
  end_date: string | null;
  categories: string[] | null;
  districts: string[] | null;
}

export interface CrimeDataResponse {
  total: number;
  limit: number;
  data: CrimeData[];
}

export interface CategoryStat {
  category: string;
  count: number;
}

export interface DistrictStat {
  district: string;
  count: number;
}

export interface HourStat {
  hour: number;
  count: number;
}

export interface DayStat {
  day: string;
  count: number;
}

export interface TimeSeriesPoint {
  month: string;
  category: string;
  count: number;
}

export interface CrimeStats {
  total_crimes: number;
  by_category: CategoryStat[];
  by_district: DistrictStat[];
  by_hour: HourStat[];
  by_day: DayStat[];
  time_series: TimeSeriesPoint[];
}

export interface PredictionRequest {
  description: string;
  date: string;
  time: string;
  district: string;
}

export interface CategoryProbability {
  category: string;
  probability: number;
}

export interface PredictionResponse {
  prediction: string;
  probabilities: CategoryProbability[];
}

export interface Metadata {
  total_crimes: number;
  date_range: {
    start: string;
    end: string;
  };
  categories: string[];
  districts: string[];
  model_info: {
    type: string;
    features: number;
  };
}
"""

# 2. src/services/api.ts - API service

"""
// src/services/api.ts
import axios from 'axios';
import {
  CrimeDataResponse,
  CrimeStats,
  FilterParams,
  Metadata,
  PredictionRequest,
  PredictionResponse
} from '../types';

const API_URL = '<http://localhost:8000>';

export const api = {
  // Get API metadata
  getMetadata: async (): Promise<Metadata> => {
    const response = await axios.get(`${API_URL}/api/metadata`);
    return response.data;
  },
  
  // Get crime data for map
  getCrimeData: async (filters: FilterParams, limit: number = 1000): Promise<CrimeDataResponse> => {
    const response = await axios.post(`${API_URL}/api/crime_data?limit=${limit}`, filters);
    return response.data;
  },
  
  // Get crime statistics
  getCrimeStats: async (filters: FilterParams): Promise<CrimeStats> => {
    const response = await axios.post(`${API_URL}/api/crime_stats`, filters);
    return response.data;
  },
  
  // Make a prediction
  predictCrime: async (request: PredictionRequest): Promise<PredictionResponse> => {
    const response = await axios.post(`${API_URL}/api/predict`, request);
    return response.data;
  }
};
"""

# 3. src/App.tsx - Main application component

"""
// src/App.tsx
import { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { api } from './services/api';
import { Metadata } from './types';

// Pages
import Dashboard from './pages/Dashboard';
import Prediction from './pages/Prediction';
import Analytics from './pages/Analytics';

// Components
import Layout from './components/Layout';
import Loading from './components/Loading';

// Create theme
const theme = createTheme({
  palette: {
    primary: {
      main: '#2c3e50',
    },
    secondary: {
      main: '#3498db',
    },
    background: {
      default: '#f5f5f5',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '2.2rem',
      fontWeight: 500,
    },
    h2: {
      fontSize: '1.8rem',
      fontWeight: 500,
    },
    h3: {
      fontSize: '1.5rem',
      fontWeight: 500,
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
        },
      },
    },
  },
});

function App() {
  const [metadata, setMetadata] = useState<Metadata | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadMetadata = async () => {
      try {
        const data = await api.getMetadata();
        setMetadata(data);
        setLoading(false);
      } catch (err) {
        console.error('Error loading metadata:', err);
        setError('Failed to load application data. Please try again later.');
        setLoading(false);
      }
    };

    loadMetadata();
  }, []);

  if (loading) {
    return <Loading message="Loading application data..." />;
  }

  if (error) {
    return (
      <div className="error-container">
        <h2>Error</h2>
        <p>{error}</p>
        <button onClick={() => window.location.reload()}>Retry</button>
      </div>
    );
  }

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Layout metadata={metadata} />}>
            <Route index element={<Dashboard metadata={metadata} />} />
            <Route path="predict" element={<Prediction metadata={metadata} />} />
            <Route path="analytics" element={<Analytics metadata={metadata} />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </ThemeProvider>
  );
}

export default App;
"""

# 4. src/components/Layout.tsx - Layout component

"""
// src/components/Layout.tsx
import { Outlet } from 'react-router-dom';
import { useState } from 'react';
import { styled } from '@mui/material/styles';
import {
  AppBar,
  Box,
  Toolbar,
  Typography,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemButton,
  IconButton,
  Divider
} from '@mui/material';
import {
  Map as MapIcon,
  DataUsage as DataIcon,
  Psychology as PredictIcon,
  Menu as MenuIcon
} from '@mui/icons-material';
import { Link, useLocation } from 'react-router-dom';
import { Metadata } from '../types';

const drawerWidth = 240;

const Main = styled('main', { shouldForwardProp: (prop) => prop !== 'open' })<{
  open?: boolean;
}>(({ theme, open }) => ({
  flexGrow: 1,
  padding: theme.spacing(3),
  transition: theme.transitions.create('margin', {
    easing: theme.transitions.easing.sharp,
    duration: theme.transitions.duration.leavingScreen,
  }),
  marginLeft: 0,
  ...(open && {
    transition: theme.transitions.create('margin', {
      easing: theme.transitions.easing.easeOut,
      duration: theme.transitions.duration.enteringScreen,
    }),
    marginLeft: `${drawerWidth}px`,
  }),
}));

interface LayoutProps {
  metadata: Metadata | null;
}

const Layout = ({ metadata }: LayoutProps) => {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const location = useLocation();
  
  const handleDrawerToggle = () => {
    setDrawerOpen(!drawerOpen);
  };
  
  const menuItems = [
    { text: 'Crime Map', icon: <MapIcon />, path: '/' },
    { text: 'Prediction', icon: <PredictIcon />, path: '/predict' },
    { text: 'Analytics', icon: <DataIcon />, path: '/analytics' },
  ];
  
  return (
    <Box sx={{ display: 'flex' }}>
      <AppBar position="fixed" sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}>
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2, display: { sm: 'none' } }}
          >
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
            SF Crime Classification Dashboard
          </Typography>
          {metadata && (
            <Typography variant="body2" sx={{ display: { xs: 'none', sm: 'block' } }}>
              Total Crimes: {metadata.total_crimes.toLocaleString()}
            </Typography>
          )}
        </Toolbar>
      </AppBar>

      <Drawer
        variant="permanent"
        sx={{
          width: drawerWidth,
          flexShrink: 0,
          display: { xs: 'none', sm: 'block' },
          [`& .MuiDrawer-paper`]: { width: drawerWidth, boxSizing: 'border-box' },
        }}
        open
      >
        <Toolbar />
        <Box sx={{ overflow: 'auto' }}>
          <List>
            {menuItems.map((item) => (
              <ListItem key={item.text} disablePadding>
                <ListItemButton 
                  component={Link} 
                  to={item.path}
                  selected={location.pathname === item.path}
                >
                  <ListItemIcon>{item.icon}</ListItemIcon>
                  <ListItemText primary={item.text} />
                </ListItemButton>
              </ListItem>
            ))}
          </List>
          <Divider />
          {metadata && (
            <Box sx={{ p: 2 }}>
              <Typography variant="body2" color="textSecondary" sx={{ mb: 1 }}>
                Date Range: {metadata.date_range.start} - {metadata.date_range.end}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Model: {metadata.model_info.type}
              </Typography>
            </Box>
          )}
        </Box>
      </Drawer>
      
      <Drawer
        variant="temporary"
        open={drawerOpen}
        onClose={handleDrawerToggle}
        sx={{
          display: { xs: 'block', sm: 'none' },
          '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
        }}
      >
        <Toolbar />
        <Box sx={{ overflow: 'auto' }}>
          <List>
            {menuItems.map((item) => (
              <ListItem key={item.text} disablePadding>
                <ListItemButton 
                  component={Link} 
                  to={item.path}
                  onClick={handleDrawerToggle}
                  selected={location.pathname === item.path}
                >
                  <ListItemIcon>{item.icon}</ListItemIcon>
                  <ListItemText primary={item.text} />
                </ListItemButton>
              </ListItem>
            ))}
          </List>
        </Box>
      </Drawer>
      
      <Main open={drawerOpen}>
        <Toolbar /> {/* Add space for the AppBar */}
        <Outlet />
        
        <Box sx={{ mt: 4, pt: 2, borderTop: '1px solid #eee', textAlign: 'center' }}>
          <Typography variant="body2" color="text.secondary">
            &copy; 2025 Crime Classification Dashboard | Powered by ML
          </Typography>
        </Box>
      </Main>
    </Box>
  );
};

export default Layout;
"""

# 5. src/pages/Dashboard.tsx - Map and visualization page

"""
// src/pages/Dashboard.tsx
import { useState, useEffect, useRef } from 'react';
import { Box, Paper, Typography, Grid, Card, CardContent, FormControl, InputLabel, Select, MenuItem, OutlinedInput, Chip, SelectChangeEvent, Button, Checkbox, FormControlLabel } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
import { CrimeData, FilterParams, Metadata, CrimeStats } from '../types';
import { api } from '../services/api';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line } from 'recharts';
import { format, parseISO } from 'date-fns';

// Replace with your Mapbox token
mapboxgl.accessToken = 'YOUR_MAPBOX_TOKEN';

interface DashboardProps {
  metadata: Metadata | null;
}

const Dashboard = ({ metadata }: DashboardProps) => {
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<mapboxgl.Map | null>(null);
  const [crimeData, setCrimeData] = useState<CrimeData[]>([]);
  const [crimeStats, setCrimeStats] = useState<CrimeStats | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const theme = useTheme();
  
  // Filters
  const [filters, setFilters] = useState<FilterParams>({
    start_date: metadata?.date_range.start || null,
    end_date: metadata?.date_range.end || null,
    categories: null,
    districts: null
  });
  
  // Map display options
  const [showPoints, setShowPoints] = useState<boolean>(true);
  const [showHeatmap, setShowHeatmap] = useState<boolean>(false);
  
  // Time granularity for the time series chart
  const [timeGranularity, setTimeGranularity] = useState<string>('month');
  
  useEffect(() => {
    // Initialize map
    if (map.current) return; // Map already initialized

    map.current = new mapboxgl.Map({
      container: mapContainer.current!,
      style: 'mapbox://styles/mapbox/light-v10',
      center: [-122.4194, 37.7749], // San Francisco
      zoom: 12
    });
    
    // Add navigation controls
    map.current.addControl(new mapboxgl.NavigationControl(), 'top-right');
    
    // Load data after map initialization
    loadData();
    
    // Cleanup
    return () => {
      map.current?.remove();
    };
  }, []);
  
  // Load data when filters change
  const loadData = async () => {
    setLoading(true);
    try {
      // Load crime data for map
      const crimeResponse = await api.getCrimeData(filters);
      setCrimeData(crimeResponse.data);

      // Load crime statistics
      const statsResponse = await api.getCrimeStats(filters);
      setCrimeStats(statsResponse);
      
      // Update map data
      updateMap(crimeResponse.data);
      
      setLoading(false);
    } catch (error) {
      console.error('Error loading data:', error);
      setLoading(false);
    }
  };
  
  // Update map with new data
  const updateMap = (data: CrimeData[]) => {
    if (!map.current) return;

    // Clear existing layers
    if (map.current.getLayer('crimes-heat')) {
      map.current.removeLayer('crimes-heat');
    }
    if (map.current.getLayer('crimes-point')) {
      map.current.removeLayer('crimes-point');
    }
    if (map.current.getSource('crimes')) {
      map.current.removeSource('crimes');
    }
    
    // No data to display
    if (data.length === 0) return;
    
    // Add new data as source
    const geojson = {
      type: 'FeatureCollection',
      features: data.map(crime => ({
        type: 'Feature',
        properties: {
          id: crime.id,
          category: crime.category,
          description: crime.description,
          date: crime.date,
          district: crime.district
        },
        geometry: {
          type: 'Point',
          coordinates: [crime.location.lng, crime.location.lat]
        }
      }))
    };
    
    map.current.addSource('crimes', {
      type: 'geojson',
      data: geojson as any
    });
    
    // Add heatmap layer
    if (showHeatmap) {
      map.current.addLayer({
        id: 'crimes-heat',
        type: 'heatmap',
        source: 'crimes',
        paint: {
          'heatmap-weight': 1,
          'heatmap-intensity': 1,
          'heatmap-color': [
            'interpolate',
            ['linear'],
            ['heatmap-density'],
            0, 'rgba(0, 0, 255, 0)',
            0.2, 'rgb(0, 0, 255)',
            0.4, 'rgb(0, 255, 255)',
            0.6, 'rgb(0, 255, 0)',
            0.8, 'rgb(255, 255, 0)',
            1, 'rgb(255, 0, 0)'
          ],
          'heatmap-radius': 10,
          'heatmap-opacity': 0.75
        }
      });
    }
    
    // Add point layer
    if (showPoints) {
      map.current.addLayer({
        id: 'crimes-point',
        type: 'circle',
        source: 'crimes',
        paint: {
          'circle-radius': 5,
          'circle-color': [
            'match',
            ['get', 'category'],
            'LARCENY/THEFT', '#FF5733',
            'ASSAULT', '#C70039',
            'DRUG/NARCOTIC', '#900C3F',
            'VEHICLE THEFT', '#581845',
            'BURGLARY', '#FFC300',
            'ROBBERY', '#FF5733',
            'OTHER OFFENSES', '#DAF7A6',
            'NON-CRIMINAL', '#FFC300',
            'SUSPICIOUS OCC', '#C70039',
            'MISSING PERSON', '#900C3F',
            'WARRANTS', '#581845',
            '#2471A3' // default color
          ],
          'circle-opacity': 0.7
        }
      });
      
      // Add popup
      const popup = new mapboxgl.Popup({
        closeButton: false,
        closeOnClick: true
      });
      
      map.current.on('mouseenter', 'crimes-point', (e) => {
        map.current!.getCanvas().style.cursor = 'pointer';
        
        const coordinates = (e.features![0].geometry as any).coordinates.slice();
        const properties = e.features![0].properties;
        
        const html = `
          <strong>${properties.category}</strong><br/>
          ${properties.description}<br/>
          <small>${properties.date} - ${properties.district}</small>
        `;
        
        popup.setLngLat(coordinates).setHTML(html).addTo(map.current!);
      });
      
      map.current.on('mouseleave', 'crimes-point', () => {
        map.current!.getCanvas().style.cursor = '';
        popup.remove();
      });
    }
  };
  
  // Handle filter changes
  const handleCategoryChange = (event: SelectChangeEvent<string[]>) => {
    const value = event.target.value;
    setFilters({
      ...filters,
      categories: typeof value === 'string' ? value.split(',') : value
    });
  };
  
  const handleDistrictChange = (event: SelectChangeEvent<string[]>) => {
    const value = event.target.value;
    setFilters({
      ...filters,
      districts: typeof value === 'string' ? value.split(',') : value
    });
  };
  
  const handleDateChange = (field: 'start_date' | 'end_date', value: string) => {
    setFilters({
      ...filters,
      [field]: value
    });
  };
  
  const handleApplyFilters = () => {
    loadData();
  };
  
  // Generate charts from crime stats
  const renderCategoryChart = () => {
    if (!crimeStats) return null;

    return (
      <ResponsiveContainer width="100%" height={300}>
        <BarChart
          data={crimeStats.by_category.sort((a, b) => b.count - a.count).slice(0, 10)}
          layout="vertical"
          margin={{ top: 20, right: 30, left: 100, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" />
          <YAxis type="category" dataKey="category" />
          <Tooltip />
          <Bar dataKey="count" fill={theme.palette.primary.main} />
        </BarChart>
      </ResponsiveContainer>
    );
  };
  
  const renderDistrictChart = () => {
    if (!crimeStats) return null;

    return (
      <ResponsiveContainer width="100%" height={300}>
        <BarChart
          data={crimeStats.by_district.sort((a, b) => b.count - a.count)}
          layout="vertical"
          margin={{ top: 20, right: 30, left: 100, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" />
          <YAxis type="category" dataKey="district" />
          <Tooltip />
          <Bar dataKey="count" fill={theme.palette.secondary.main} />
        </BarChart>
      </ResponsiveContainer>
    );
  };
  
  const renderTimeSeriesChart = () => {
    if (!crimeStats) return null;

    // Process data based on time granularity
    let data: any[] = [];
    
    switch (timeGranularity) {
      case 'month':
        // Group by month and sum all categories
        const monthlyData: { [key: string]: { [key: string]: number } } = {};
        crimeStats.time_series.forEach(point => {
          if (!monthlyData[point.month]) {
            monthlyData[point.month] = {};
          }
          monthlyData[point.month][point.category] = point.count;
        });
        
        data = Object.keys(monthlyData).map(month => ({
          month,
          ...monthlyData[month]
        })).sort((a, b) => a.month.localeCompare(b.month));
        break;
        
      default:
        data = [];
    }
    
    // Get top 5 categories for legend
    const topCategories = crimeStats.by_category
      .sort((a, b) => b.count - a.count)
      .slice(0, 5)
      .map(item => item.category);
    
    return (
      <ResponsiveContainer width="100%" height={400}>
        <LineChart
          data={data}
          margin={{ top: 20, right: 30, left: 20, bottom: 30 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="month" />
          <YAxis />
          <Tooltip />
          <Legend />
          {topCategories.map((category, index) => (
            <Line 
              key={category}
              type="monotone"
              dataKey={category}
              stroke={theme.palette.primary.main}
              strokeOpacity={(5 - index) / 5}
              activeDot={{ r: 8 }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    );
  };
  
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        San Francisco Crime Map
      </Typography>

      <Grid container spacing={3}>
        {/* Filters */}
        <Grid item xs={12} md={3}>
          <Paper elevation={2} sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Filters
            </Typography>
            
            <FormControl fullWidth margin="normal" size="small">
              <InputLabel>Start Date</InputLabel>
              <OutlinedInput
                type="date"
                value={filters.start_date || ''}
                onChange={(e) => handleDateChange('start_date', e.target.value)}
                label="Start Date"
              />
            </FormControl>
            
            <FormControl fullWidth margin="normal" size="small">
              <InputLabel>End Date</InputLabel>
              <OutlinedInput
                type="date"
                value={filters.end_date || ''}
                onChange={(e) => handleDateChange('end_date', e.target.value)}
                label="End Date"
              />
            </FormControl>
            
            <FormControl fullWidth margin="normal" size="small">
              <InputLabel>Crime Categories</InputLabel>
              <Select
                multiple
                value={filters.categories || []}
                onChange={handleCategoryChange}
                input={<OutlinedInput label="Crime Categories" />}
                renderValue={(selected) => (
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {selected.map((value) => (
                      <Chip key={value} label={value} size="small" />
                    ))}
                  </Box>
                )}
              >
                {metadata?.categories.map((category) => (
                  <MenuItem key={category} value={category}>
                    {category}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            
            <FormControl fullWidth margin="normal" size="small">
              <InputLabel>Police Districts</InputLabel>
              <Select
                multiple
                value={filters.districts || []}
                onChange={handleDistrictChange}
                input={<OutlinedInput label="Police Districts" />}
                renderValue={(selected) => (
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {selected.map((value) => (
                      <Chip key={value} label={value} size="small" />
                    ))}
                  </Box>
                )}
              >
                {metadata?.districts.map((district) => (
                  <MenuItem key={district} value={district}>
                    {district}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            
            <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>
              Map Options
            </Typography>
            
            <FormControlLabel
              control={
                <Checkbox
                  checked={showPoints}
                  onChange={(e) => setShowPoints(e.target.checked)}
                />
              }
              label="Show Points"
            />
            
            <FormControlLabel
              control={
                <Checkbox
                  checked={showHeatmap}
                  onChange={(e) => setShowHeatmap(e.target.checked)}
                />
              }
              label="Show Heatmap"
            />
            
            <Button 
              variant="contained" 
              fullWidth 
              sx={{ mt: 2 }}
              onClick={handleApplyFilters}
            >
              Apply Filters
            </Button>
          </Paper>
        </Grid>
        
        {/* Map */}
        <Grid item xs={12} md={9}>
          <Paper elevation={2} sx={{ height: '600px', position: 'relative' }}>
            <Box ref={mapContainer} sx={{ width: '100%', height: '100%' }} />
            {loading && (
              <Box
                sx={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  right: 0,
                  bottom: 0,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  backgroundColor: 'rgba(255, 255, 255, 0.7)',
                  zIndex: 1000
                }}
              >
                <Typography>Loading data...</Typography>
              </Box>
            )}
          </Paper>
        </Grid>
        
        {/* Statistics */}
        <Grid item xs={12}>
          <Paper elevation={2} sx={{ p: 2, mt: 2 }}>
            <Typography variant="h5" gutterBottom>
              Crime Statistics
            </Typography>
            
            {crimeStats && (
              <Typography variant="subtitle1" gutterBottom>
                Showing {crimeStats.total_crimes.toLocaleString()} crimes
              </Typography>
            )}
            
            <Grid container spacing={3}>
              {/* Category breakdown */}
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Top Crime Categories
                    </Typography>
                    {renderCategoryChart()}
                  </CardContent>
                </Card>
              </Grid>
              
              {/* District breakdown */}
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Crimes by District
                    </Typography>
                    {renderDistrictChart()}
                  </CardContent>
                </Card>
              </Grid>
              
              {/* Time series */}
              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                      <Typography variant="h6">
                        Crimes Over Time
                      </Typography>
                      <FormControl sx={{ minWidth: 120 }} size="small">
                        <InputLabel>Time Period</InputLabel>
                        <Select
                          value={timeGranularity}
                          label="Time Period"
                          onChange={(e) => setTimeGranularity(e.target.value)}
                        >
                          <MenuItem value="month">Month</MenuItem>
                        </Select>
                      </FormControl>
                    </Box>
                    {renderTimeSeriesChart()}
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;
"""

# 6. src/pages/Prediction.tsx - Prediction page

"""
// src/pages/Prediction.tsx
import { useState } from 'react';
import { Box, Paper, Typography, TextField, Grid, Button, FormControl, InputLabel, Select, MenuItem, SelectChangeEvent, Card, CardContent, LinearProgress, Alert } from '@mui/material';
import { PredictionRequest, PredictionResponse, Metadata, CategoryProbability } from '../types';
import { api } from '../services/api';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { format } from 'date-fns';

interface PredictionProps {
  metadata: Metadata | null;
}

const Prediction = ({ metadata }: PredictionProps) => {
  const [description, setDescription] = useState<string>('Suspect broke car window and stole laptop from the back seat.');
  const [date, setDate] = useState<string>(format(new Date(), 'yyyy-MM-dd'));
  const [time, setTime] = useState<string>(format(new Date(), 'HH:mm'));
  const [district, setDistrict] = useState<string>(metadata?.districts[0] || '');
  
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const request: PredictionRequest = {
        description,
        date,
        time,
        district
      };
      
      const response = await api.predictCrime(request);
      setResult(response);
    } catch (err: any) {
      console.error('Error making prediction:', err);
      setError(err.message || 'An error occurred while making the prediction.');
    } finally {
      setLoading(false);
    }
  };
  
  const renderProbabilityChart = (probabilities: CategoryProbability[]) => {
    // Sort and take top 5 for visualization
    const topProbabilities = [...probabilities]
      .sort((a, b) => b.probability - a.probability)
      .slice(0, 5);

    // Format for chart
    const data = topProbabilities.map(item => ({
      category: item.category,
      probability: (item.probability * 100).toFixed(2),
      fill: item.category === result?.prediction ? '#2c3e50' : '#3498db'
    }));
    
    return (
      <ResponsiveContainer width="100%" height={300}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 20, right: 30, left: 150, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" domain={[0, 100]} label={{ value: 'Probability (%)', position: 'insideBottom', offset: -5 }} />
          <YAxis type="category" dataKey="category" />
          <Tooltip formatter={(value) => [`${value}%`, 'Probability']} />
          <Bar dataKey="probability" fill="#3498db" />
        </BarChart>
      </ResponsiveContainer>
    );
  };
  
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Crime Category Prediction
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Paper elevation={2} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Enter Crime Details
            </Typography>
            
            <form onSubmit={handleSubmit}>
              <TextField
                fullWidth
                label="Crime Description"
                multiline
                rows={4}
                margin="normal"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                required
              />
              
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <TextField
                    fullWidth
                    label="Date"
                    type="date"
                    margin="normal"
                    InputLabelProps={{ shrink: true }}
                    value={date}
                    onChange={(e) => setDate(e.target.value)}
                    required
                  />
                </Grid>
                <Grid item xs={6}>
                  <TextField
                    fullWidth
                    label="Time"
                    type="time"
                    margin="normal"
                    InputLabelProps={{ shrink: true }}
                    value={time}
                    onChange={(e) => setTime(e.target.value)}
                    required
                  />
                </Grid>
              </Grid>
              
              <FormControl fullWidth margin="normal">
                <InputLabel>Police District</InputLabel>
                <Select
                  value={district}
                  label="Police District"
                  onChange={(e) => setDistrict(e.target.value)}
                  required
                >
                  {metadata?.districts.map((d) => (
                    <MenuItem key={d} value={d}>
                      {d}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              
              <Button 
                type="submit" 
                variant="contained" 
                fullWidth 
                sx={{ mt: 3 }}
                disabled={loading}
              >
                Predict Crime Category
              </Button>
            </form>
            
            {loading && <LinearProgress sx={{ mt: 2 }} />}
            {error && <Alert severity="error" sx={{ mt: 2 }}>{error}</Alert>}
          </Paper>
        </Grid>
        
        <Grid item xs={12} md={6}>
          {result ? (
            <Paper elevation={2} sx={{ p: 3, height: '100%' }}>
              <Typography variant="h6" gutterBottom>
                Prediction Results
              </Typography>
              
              <Box sx={{ mb: 3 }}>
                <Typography variant="subtitle1" gutterBottom>
                  Predicted Crime Category:
                </Typography>
                <Typography variant="h5" color="primary">
                  {result.prediction}
                </Typography>
              </Box>
              
              <Typography variant="subtitle1" gutterBottom>
                Probability Distribution:
              </Typography>
              
              {renderProbabilityChart(result.probabilities)}
              
              <Typography variant="body2" sx={{ mt: 3, color: 'text.secondary' }}>
                The model uses both the text description and contextual features (time, location) 
                to make predictions. Text features are particularly important for classification.
              </Typography>
            </Paper>
          ) : (
            <Paper elevation={2} sx={{ p: 3, height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h6" color="text.secondary" gutterBottom>
                  Enter crime details and click "Predict Crime Category" to see results.
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  The model analyzes the description and contextual information to classify the crime.
                </Typography>
              </Box>
            </Paper>
          )}
        </Grid>
      </Grid>
    </Box>
  );
};

export default Prediction;
"""

# 7. src/pages/Analytics.tsx - Analytics page

"""
// src/pages/Analytics.tsx
import { useState, useEffect } from 'react';
import { Box, Paper, Typography, Grid, Card, CardContent, FormControl, InputLabel, Select, MenuItem, SelectChangeEvent, Button, TextField } from '@mui/material';
import { format } from 'date-fns';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line } from 'recharts';
import { api } from '../services/api';
import { FilterParams, Metadata, CrimeStats } from '../types';
import { useTheme } from '@mui/material/styles';

interface AnalyticsProps {
  metadata: Metadata | null;
}

const Analytics = ({ metadata }: AnalyticsProps) => {
  const [crimeStats, setCrimeStats] = useState<CrimeStats | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const theme = useTheme();
  
  // Filters
  const [filters, setFilters] = useState<FilterParams>({
    start_date: metadata?.date_range.start || null,
    end_date: metadata?.date_range.end || null,
    categories: null,
    districts: null
  });
  
  useEffect(() => {
    if (metadata) {
      loadData();
    }
  }, [metadata]);
  
  const loadData = async () => {
    setLoading(true);
    try {
      const statsResponse = await api.getCrimeStats(filters);
      setCrimeStats(statsResponse);
      setLoading(false);
    } catch (error) {
      console.error('Error loading data:', error);
      setLoading(false);
    }
  };
  
  const handleDateChange = (field: 'start_date' | 'end_date', value: string) => {
    setFilters({
      ...filters,
      [field]: value
    });
  };
  
  const handleApplyFilters = () => {
    loadData();
  };
  
  // Charts and visualizations
  const renderCategoryPieChart = () => {
    if (!crimeStats) return null;

    // Get top categories
    const topCategories = crimeStats.by_category
      .sort((a, b) => b.count - a.count)
      .slice(0, 7);
    
    // Add "Other" category
    const topCategoriesSum = topCategories.reduce((sum, item) => sum + item.count, 0);
    const otherCount = crimeStats.total_crimes - topCategoriesSum;
    
    if (otherCount > 0) {
      topCategories.push({ category: 'Other', count: otherCount });
    }
    
    // Colors
    const COLORS = [
      '#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', 
      '#8DD1E1', '#A4DE6C', '#D0ED57'
    ];
    
    return (
      <ResponsiveContainer width="100%" height={300}>
        <PieChart>
          <Pie
            data={topCategories}
            cx="50%"
            cy="50%"
            labelLine={false}
            label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
            outerRadius={100}
            fill="#8884d8"
            dataKey="count"
            nameKey="category"
          >
            {topCategories.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip 
            formatter={(value) => [`${value} crimes`, 'Count']} 
            labelFormatter={(label) => `Category: ${label}`}
          />
        </PieChart>
      </ResponsiveContainer>
    );
  };
  
  const renderHourOfDayChart = () => {
    if (!crimeStats) return null;

    // Map for time period labels
    const timePeriodMap: { [key: number]: string } = {
      0: 'Early Morning (12am-5am)',
      1: 'Morning (5am-12pm)',
      2: 'Afternoon (12pm-5pm)',
      3: 'Evening (5pm-9pm)',
      4: 'Night (9pm-12am)'
    };
    
    // Group data by time period
    const hourData = [...crimeStats.by_hour];
    
    const timePeriodsData = [
      { period: 0, count: hourData.filter(h => h.hour >= 0 && h.hour < 5).reduce((sum, h) => sum + h.count, 0) },
      { period: 1, count: hourData.filter(h => h.hour >= 5 && h.hour < 12).reduce((sum, h) => sum + h.count, 0) },
      { period: 2, count: hourData.filter(h => h.hour >= 12 && h.hour < 17).reduce((sum, h) => sum + h.count, 0) },
      { period: 3, count: hourData.filter(h => h.hour >= 17 && h.hour < 21).reduce((sum, h) => sum + h.count, 0) },
      { period: 4, count: hourData.filter(h => h.hour >= 21 && h.hour < 24).reduce((sum, h) => sum + h.count, 0) }
    ];
    
    const formattedData = timePeriodsData.map(item => ({
      ...item,
      name: timePeriodMap[item.period]
    }));
    
    return (
      <ResponsiveContainer width="100%" height={300}>
        <BarChart
          data={formattedData}
          margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" />
          <YAxis />
          <Tooltip formatter={(value) => [`${value} crimes`, 'Count']} />
          <Bar dataKey="count" fill={theme.palette.primary.main} />
        </BarChart>
      </ResponsiveContainer>
    );
  };
  
  const renderDayOfWeekChart = () => {
    if (!crimeStats) return null;

    // Ensure proper day order
    const dayOrder = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
    const dayMap: { [key: string]: number } = {};
    dayOrder.forEach((day, index) => { dayMap[day] = index; });
    
    // Sort days in correct order
    const sortedDays = [...crimeStats.by_day]
      .sort((a, b) => {
        const dayA = a.day;
        const dayB = b.day;
        return dayMap[dayA] - dayMap[dayB];
      });
    
    // Format for bar chart
    const data = sortedDays.map(item => ({
      day: item.day,
      count: item.count,
      isWeekend: item.day === 'Saturday' || item.day === 'Sunday'
    }));
    
    return (
      <ResponsiveContainer width="100%" height={300}>
        <BarChart
          data={data}
          margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="day" />
          <YAxis />
          <Tooltip formatter={(value) => [`${value} crimes`, 'Count']} />
          <Bar 
            dataKey="count" 
            fill={theme.palette.secondary.main}
            // Highlight weekends
            fillOpacity={(entry) => entry.isWeekend ? 1 : 0.6}
          />
        </BarChart>
      </ResponsiveContainer>
    );
  };
  
  const renderDistrictChart = () => {
    if (!crimeStats) return null;

    const data = crimeStats.by_district
      .sort((a, b) => b.count - a.count);
    
    return (
      <ResponsiveContainer width="100%" height={350}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 20, right: 30, left: 100, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" />
          <YAxis type="category" dataKey="district" />
          <Tooltip formatter={(value) => [`${value} crimes`, 'Count']} />
          <Bar dataKey="count" fill={theme.palette.primary.main} />
        </BarChart>
      </ResponsiveContainer>
    );
  };
  
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Crime Analytics
      </Typography>

      <Paper elevation={2} sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Filter Data
        </Typography>
        
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} sm={4}>
            <TextField
              fullWidth
              label="Start Date"
              type="date"
              value={filters.start_date || ''}
              onChange={(e) => handleDateChange('start_date', e.target.value)}
              InputLabelProps={{ shrink: true }}
            />
          </Grid>
          
          <Grid item xs={12} sm={4}>
            <TextField
              fullWidth
              label="End Date"
              type="date"
              value={filters.end_date || ''}
              onChange={(e) => handleDateChange('end_date', e.target.value)}
              InputLabelProps={{ shrink: true }}
            />
          </Grid>
          
          <Grid item xs={12} sm={4}>
            <Button 
              variant="contained" 
              fullWidth
              onClick={handleApplyFilters}
            >
              Apply Filters
            </Button>
          </Grid>
        </Grid>
      </Paper>
      
      {crimeStats && (
        <Box sx={{ mb: 2 }}>
          <Typography variant="h6">
            Analyzing {crimeStats.total_crimes.toLocaleString()} crime incidents
          </Typography>
        </Box>
      )}
      
      <Grid container spacing={3}>
        {/* Crime by Category */}
        <Grid item xs={12} md={6}>
          <Card elevation={2}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Crime Categories Distribution
              </Typography>
              {renderCategoryPieChart()}
            </CardContent>
          </Card>
        </Grid>
        
        {/* Crime by Time of Day */}
        <Grid item xs={12} md={6}>
          <Card elevation={2}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Crimes by Time of Day
              </Typography>
              {renderHourOfDayChart()}
            </CardContent>
          </Card>
        </Grid>
        
        {/* Crime by Day of Week */}
        <Grid item xs={12} md={6}>
          <Card elevation={2}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Crimes by Day of Week
              </Typography>
              {renderDayOfWeekChart()}
            </CardContent>
          </Card>
        </Grid>
        
        {/* Crime by District */}
        <Grid item xs={12} md={6}>
          <Card elevation={2}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Crimes by Police District
              </Typography>
              {renderDistrictChart()}
            </CardContent>
          </Card>
        </Grid>
        
        {/* Key Insights */}
        <Grid item xs={12}>
          <Card elevation={2}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Key Insights
              </Typography>
              
              {crimeStats ? (
                <Grid container spacing={2}>
                  <Grid item xs={12} md={4}>
                    <Box sx={{ p: 2, backgroundColor: theme.palette.grey[100], borderRadius: 1 }}>
                      <Typography variant="subtitle1" color="primary" gutterBottom>
                        Most Common Crime
                      </Typography>
                      <Typography variant="h6">
                        {crimeStats.by_category.sort((a, b) => b.count - a.count)[0].category}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {crimeStats.by_category.sort((a, b) => b.count - a.count)[0].count.toLocaleString()} incidents
                      </Typography>
                    </Box>
                  </Grid>
                  
                  <Grid item xs={12} md={4}>
                    <Box sx={{ p: 2, backgroundColor: theme.palette.grey[100], borderRadius: 1 }}>
                      <Typography variant="subtitle1" color="primary" gutterBottom>
                        Highest Crime District
                      </Typography>
                      <Typography variant="h6">
                        {crimeStats.by_district.sort((a, b) => b.count - a.count)[0].district}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {crimeStats.by_district.sort((a, b) => b.count - a.count)[0].count.toLocaleString()} incidents
                      </Typography>
                    </Box>
                  </Grid>
                  
                  <Grid item xs={12} md={4}>
                    <Box sx={{ p: 2, backgroundColor: theme.palette.grey[100], borderRadius: 1 }}>
                      <Typography variant="subtitle1" color="primary" gutterBottom>
                        Peak Time for Crime
                      </Typography>
                      <Typography variant="h6">
                        {crimeStats.by_hour.sort((a, b) => b.count - a.count)[0].hour}:00 - {crimeStats.by_hour.sort((a, b) => b.count - a.count)[0].hour + 1}:00
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {crimeStats.by_hour.sort((a, b) => b.count - a.count)[0].count.toLocaleString()} incidents
                      </Typography>
                    </Box>
                  </Grid>
                </Grid>
              ) : (
                <Typography>Loading insights...</Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Analytics;
"""

# 8. src/components/Loading.tsx - Loading component

"""
// src/components/Loading.tsx
import { Box, CircularProgress, Typography } from '@mui/material';

interface LoadingProps {
  message?: string;
}

const Loading = ({ message = 'Loading...' }: LoadingProps) => {
  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100vh',
      }}
    >
      <CircularProgress size={60} thickness={4} />
      <Typography variant="h6" sx={{ mt: 2 }}>
        {message}
      </Typography>
    </Box>
  );
};

export default Loading;
"""

# 9. Create deployment instructions for the full-stack application

"""

# Deployment Instructions

## Backend Setup

1. Install backend dependencies:

```bash
pip install fastapi uvicorn pandas numpy catboost scikit-learn
```

2. Make sure you have your model artifacts in a directory named "model_artifacts" in the project root.

3. Run the FastAPI backend:

```bash
uvicorn main:app --reload
```

The API will run on <http://localhost:8000>

## Frontend Setup

1. Navigate to the frontend directory:

```bash
cd crime-dashboard
```

2. Install dependencies:

```bash
npm install
```

3. Update the Mapbox token in src/pages/Dashboard.tsx with your own token from mapbox.com.

4. Start the development server:

```bash
npm run dev
```

The frontend will run on <http://localhost:5173>

## Production Deployment

### Backend (FastAPI)

1. For production deployment, you might want to use Gunicorn with Uvicorn workers:

```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
```

2. Or deploy to a cloud provider that supports Python applications, like Heroku, AWS, or Google Cloud.

### Frontend (React)

1. Build the production bundle:

```bash
npm run build
```

2. The build files will be in the "dist" directory, which you can serve with any web server.

3. For static site deployment, you can use services like Netlify, Vercel, or GitHub Pages.

4. Make sure to update the API_URL in src/services/api.ts to point to your production backend URL.
"""

## Recommendation and Implementation Plan

Based on your requirements for a frontend/backend setup with interactive maps, I recommend the __React + FastAPI__ option as the most professional and scalable solution. Here's why and how to implement it:

## Why React + FastAPI?

- __Clean Separation__: True frontend/backend architecture with independent scaling
- __Most Professional__: Industry-standard tech stack used by companies worldwide
- __Rich Interactivity__: React's component system offers the best map and chart customization
- __Future-proof__: Easily extensible as your project grows
- __Deployment Flexibility__: Can deploy frontend and backend to different services

## Step-by-Step Implementation Plan

### 1. Setup Environment (Day 1)

```bash
# Create project structure
mkdir -p crime-classification/{frontend,backend}
cd crime-classification

# Backend setup
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install fastapi uvicorn pandas numpy scikit-learn catboost

# Frontend setup
cd ../frontend
npm create vite@latest . -- --template react-ts
npm install axios react-router-dom mapbox-gl @mui/material @mui/icons-material @emotion/react @emotion/styled recharts date-fns
```

### 2. Backend Implementation (Day 1-2)

1. Create `backend/main.py` using the FastAPI code I provided earlier
2. Move your model artifacts to `backend/model_artifacts/`
3. Test API endpoints:

```bash
cd backend
uvicorn main:app --reload
```

Visit <http://localhost:8000/docs> to see the interactive API documentation.

### 3. Frontend Implementation (Day 2-3)

1. Set up the folder structure in the frontend directory:

```bash
cd frontend
mkdir -p src/{components,pages,services,types,utils}
```

2. Copy the React files from my previous code into their respective directories
3. Get a free Mapbox token at <https://mapbox.com> and replace `YOUR_MAPBOX_TOKEN` in Dashboard.tsx
4. Start the development server:

```bash
npm run dev
```

### 4. Connect Frontend to Backend (Day 3)

1. Make sure your backend is running on <http://localhost:8000>
2. Ensure the API_URL in src/services/api.ts is set correctly
3. Test basic API calls and verify data is flowing between systems

### 5. Map Integration (Day 4)

1. Focus on the Dashboard.tsx component and map integration
2. Test with your model's crime predictions and data visualization
3. Add custom styling for the crime points based on categories

### 6. Production Deployment (Day 5)

#### Backend Deployment

```bash
# Build for production
cd backend
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
```

#### Frontend Deployment

```bash
cd frontend
npm run build
```

Deploy the built files from the `dist` directory to any static hosting service.

## Alternative Options

If you want a quicker solution with less setup:

- __Dash__: Single-file solution with Python only, good balance of features and simplicity
- __Streamlit__: Fastest to implement but less customizable and professional-looking

Let me know which route you'd like to take, and I can provide more detailed implementation steps for that specific technology!
