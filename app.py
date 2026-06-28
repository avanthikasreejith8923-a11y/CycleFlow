"""
CycleFlow - Streamlit Web App
=============================
Interactive Bike-Sharing Demand Prediction App
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler
import json

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="CycleFlow - Bike Rental Predictor",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        color: #1f77b4;
        text-align: center;
        font-size: 3em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .subheader {
        color: #666;
        text-align: center;
        font-size: 1.2em;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL AND DATA
# ============================================================================
@st.cache_resource
def load_model():
    """Load the trained Gradient Boosting model"""
    return joblib.load('models/gradient_boosting_engineered.pkl')

@st.cache_data
def load_training_data():
    """Load training data for reference"""
    X_train = pd.read_csv('data/processed/X_train_engineered.csv')
    y_train = pd.read_csv('data/processed/y_train_engineered.csv').values.ravel()
    return X_train, y_train

@st.cache_data
def load_test_data():
    """Load test data for visualization"""
    X_test = pd.read_csv('data/processed/X_test_engineered.csv')
    y_test = pd.read_csv('data/processed/y_test_engineered.csv').values.ravel()
    return X_test, y_test

@st.cache_data
def load_hyperparameters():
    """Load best hyperparameters"""
    with open('results/best_hyperparameters.json', 'r') as f:
        return json.load(f)

# Load everything
model = load_model()
X_train, y_train = load_training_data()
X_test, y_test = load_test_data()
best_hyperparams = load_hyperparameters()

# ============================================================================
# HEADER
# ============================================================================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<h1 class="main-header">🚴 CycleFlow</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">AI-Powered Bike-Sharing Demand Prediction</p>', unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# SIDEBAR: PROJECT INFO
# ============================================================================
with st.sidebar:
    st.header("📊 Project Info")
    
    st.markdown("""
    **CycleFlow** predicts daily bike rental demand using machine learning.
    
    ### Model Details
    - **Algorithm:** Gradient Boosting Regressor
    - **Features:** 15 (11 original + 4 engineered)
    - **Training Samples:** 584
    - **Test Samples:** 147
    
    ### Performance
    - **Test R² Score:** 0.8995
    - **Test RMSE:** ±435 rentals
    - **Explanation:** Model explains 89.95% of rental variation
    
    ### Features Used
    - Weather conditions
    - Temperature & Humidity
    - Day type (weekend/weekday)
    - Season & Month
    - Engineered interactions
    """)
    
    st.markdown("---")
    st.subheader("📁 Resources")
    st.markdown("""
    - [GitHub Repo](https://github.com/avanthikasreejith8923-a11y/cycleflow)
    - [Dataset](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset)
    """)

# ============================================================================
# MAIN CONTENT: PREDICTION INTERFACE
# ============================================================================
st.header("🎯 Make a Prediction")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Weather Information")
    
    # Temperature input
    temp = st.slider(
        "🌡️ Temperature (normalized 0-1)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Normalized temperature (0=cold, 1=hot)"
    )
    
    # Humidity input
    humidity = st.slider(
        "💧 Humidity (normalized 0-1)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Normalized humidity level"
    )
    
    # Weather situation
    weather_sit = st.selectbox(
        "☁️ Weather Condition",
        [1, 2, 3, 4],
        format_func=lambda x: {
            1: "Clear",
            2: "Mist/Cloudy",
            3: "Light Rain/Snow",
            4: "Heavy Rain/Snow"
        }[x]
    )
    
    # Wind speed
    windspeed = st.slider(
        "💨 Wind Speed (normalized 0-1)",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05
    )

with col2:
    st.subheader("Date & Season Information")
    
    # Season
    season = st.selectbox(
        "🌿 Season",
        [1, 2, 3, 4],
        format_func=lambda x: {
            1: "Spring",
            2: "Summer",
            3: "Fall",
            4: "Winter"
        }[x]
    )
    
    # Month
    month = st.slider(
        "📅 Month",
        min_value=1,
        max_value=12,
        value=6,
        step=1,
        format="Month %d"
    )
    
    # Weekday
    weekday = st.selectbox(
        "📆 Day of Week",
        [0, 1, 2, 3, 4, 5, 6],
        format_func=lambda x: {
            0: "Sunday",
            1: "Monday",
            2: "Tuesday",
            3: "Wednesday",
            4: "Thursday",
            5: "Friday",
            6: "Saturday"
        }[x]
    )
    
    # Holiday
    is_holiday = st.checkbox("🎉 Is Holiday?", value=False)
    holiday = 1 if is_holiday else 0

# ============================================================================
# MAKE PREDICTION
# ============================================================================
st.markdown("---")

# Prepare input features (in correct order)
yr = 1  # Assuming current year (2012 in dataset)
workingday = 0 if (weekday == 0 or weekday == 6 or holiday == 1) else 1
atemp = temp * 0.9  # Feel-like temperature slightly less than actual

# Create feature vector (same order as training)
features_list = [
    season, yr, month, holiday, weekday, workingday, weather_sit, 
    temp, atemp, humidity, windspeed,
    # Engineered features
    temp * (1 - humidity),  # temp_humidity_comfort
    season * temp,          # season_temp
    int((weekday == 0) or (weekday == 6)),  # is_weekend
    (5 - weather_sit) * temp  # weather_temp
]

feature_names = [
    'season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit',
    'temp', 'atemp', 'hum', 'windspeed',
    'temp_humidity_comfort', 'season_temp', 'is_weekend', 'weather_temp'
]

# Scale features (using training data stats)
scaler = StandardScaler()
X_train_values = X_train.values
scaler.fit(X_train_values)

features_array = np.array(features_list).reshape(1, -1)
features_scaled = scaler.transform(features_array)

# Make prediction
prediction = model.predict(features_scaled)[0]

# ============================================================================
# DISPLAY PREDICTION
# ============================================================================
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="🎯 Predicted Rentals",
        value=f"{int(prediction):,}",
        delta=f"bikes/day"
    )

with col2:
    # Compare to historical mean
    historical_mean = y_train.mean()
    diff_from_mean = prediction - historical_mean
    st.metric(
        label="📊 vs Historical Average",
        value=f"{int(historical_mean):,}",
        delta=f"{diff_from_mean:+.0f} bikes"
    )

with col3:
    # Confidence based on model performance
    confidence = 0.90  # 90% based on R² = 0.8995
    st.metric(
        label="🎓 Model Confidence",
        value=f"{confidence*100:.0f}%",
        delta="R² = 0.8995"
    )

# ============================================================================
# PREDICTION INSIGHTS
# ============================================================================
st.markdown("---")
st.header("📈 Prediction Insights")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Summary")
    
    input_summary = f"""
    **Weather:** {['Clear', 'Mist', 'Light Rain', 'Heavy Rain'][weather_sit-1]}
    - Temperature: {temp:.2f} (normalized)
    - Humidity: {humidity:.2f}
    - Wind Speed: {windspeed:.2f}
    
    **Date:** {['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'][weekday]}, Month {month}
    - Season: {['Spring', 'Summer', 'Fall', 'Winter'][season-1]}
    - Holiday: {'Yes' if is_holiday else 'No'}
    - Working Day: {'Yes' if workingday else 'No'}
    """
    st.markdown(input_summary)

with col2:
    st.subheader("Prediction Range")
    
    # Calculate error margin based on test RMSE
    test_rmse = 435
    lower_bound = prediction - (1.96 * test_rmse)
    upper_bound = prediction + (1.96 * test_rmse)
    
    st.markdown(f"""
    **95% Confidence Interval:**
    - Lower Bound: {int(lower_bound):,} rentals
    - Point Estimate: {int(prediction):,} rentals
    - Upper Bound: {int(upper_bound):,} rentals
    
    **Interpretation:** We're 95% confident actual rentals will fall in this range.
    """)

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================
st.markdown("---")
st.header("🔍 What Influenced This Prediction?")

# Get feature importances from model
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

col1, col2 = st.columns([2, 1])

with col1:
    # Plot top 10 features
    top_10 = feature_importance_df.head(10)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#2ca02c' if feat in ['temp_humidity_comfort', 'season_temp', 'is_weekend', 'weather_temp'] 
              else '#1f77b4' for feat in top_10['Feature']]
    
    ax.barh(range(len(top_10)), top_10['Importance'].values, color=colors, edgecolor='black', linewidth=1)
    ax.set_yticks(range(len(top_10)))
    ax.set_yticklabels(top_10['Feature'].values, fontsize=10)
    ax.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
    ax.set_title('Top 10 Most Influential Features\n(Green = Engineered Features)', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader("💡 Key Drivers")
    top_3 = feature_importance_df.head(3)
    for idx, row in top_3.iterrows():
        st.write(f"**{row['Feature']}** → {row['Importance']:.2%}")

# ============================================================================
# MODEL PERFORMANCE
# ============================================================================
st.markdown("---")
st.header("📊 Model Performance Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="R² Score", value="0.8995", delta="89.95% variance explained")

with col2:
    st.metric(label="RMSE", value="435 rentals", delta="average error")

with col3:
    st.metric(label="MAE", value="288 rentals", delta="median error")

with col4:
    st.metric(label="Samples", value="731 days", delta="training + test")

# ============================================================================
# COMPARISON PLOT
# ============================================================================
st.markdown("---")
st.header("📉 Historical Performance")

# Plot actual vs predicted on test set
fig, ax = plt.subplots(figsize=(14, 5))

# Get test predictions for visualization
X_test_array = X_test.values
X_test_scaled = scaler.transform(X_test_array)
test_predictions = model.predict(X_test_scaled)

x_axis = range(len(y_test))
ax.plot(x_axis, y_test, 'o-', label='Actual Rentals', color='steelblue', alpha=0.7, linewidth=2, markersize=5)
ax.plot(x_axis, test_predictions, 's--', label='Predicted Rentals', color='coral', alpha=0.7, linewidth=2, markersize=4)

ax.set_xlabel('Test Sample Index', fontsize=11, fontweight='bold')
ax.set_ylabel('Bike Rentals', fontsize=11, fontweight='bold')
ax.set_title('Model Performance on Test Set: Actual vs Predicted', fontsize=12, fontweight='bold')
ax.legend(fontsize=11, loc='best')
ax.grid(alpha=0.3)

st.pyplot(fig)
plt.close()

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
### About CycleFlow
CycleFlow is an AI-powered prediction system for bike-sharing demand. 
It uses Gradient Boosting with engineered features to predict daily rental counts 
based on weather conditions and temporal patterns.

**Built with:** Python, Scikit-learn, Streamlit  
**Dataset:** UCI Bike Sharing Dataset  
**Author:** Avanthika Sreejith

---
*This app is part of a machine learning portfolio project. 
[View on GitHub](https://github.com/avanthikasreejith8923-a11y/cycleflow)*
""")