"""
CycleFlow - Session 1: Data Loading, EDA & Preprocessing
========================================================
Goal: Load bike-sharing data, explore it, preprocess safely (no data leakage)
Time: ~2 hours
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import urllib.request

# ============================================================================
# STEP 1: Download the Dataset
# ============================================================================
print("\n[STEP 1] Downloading UCI Bike Sharing Dataset...")

dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"
dataset_path = "data/raw/bike_sharing.zip"

# Create raw folder if it doesn't exist
os.makedirs("data/raw", exist_ok=True)

# Download the dataset
try:
    urllib.request.urlretrieve(dataset_url, dataset_path)
    print(f"✓ Downloaded to {dataset_path}")
except Exception as e:
    print(f"Error downloading: {e}")
    print("Alternative: Download manually from https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset")

# Extract the zip
import zipfile
with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
    zip_ref.extractall("data/raw/")
print("✓ Extracted dataset")

# ============================================================================
# STEP 2: Load Data
# ============================================================================
print("\n[STEP 2] Loading data...")

# The dataset has day.csv (daily aggregates) and hour.csv (hourly)
# We'll use day.csv for simplicity
df = pd.read_csv("data/raw/day.csv")
print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nFirst 5 rows:")
print(df.head())

# ============================================================================
# STEP 3: Exploratory Data Analysis (EDA)
# ============================================================================
print("\n[STEP 3] Exploratory Data Analysis...")

print("\n--- Dataset Info ---")
print(df.info())

print("\n--- Summary Statistics ---")
print(df.describe())

print("\n--- Missing Values ---")
print(df.isnull().sum())

# Check columns
print("\n--- Column Names & Meaning ---")
print("""
instant: record index
dteday: date
season: 1=spring, 2=summer, 3=fall, 4=winter
yr: year (0=2011, 1=2012)
mnth: month (1-12)
holiday: 1=holiday, 0=not
weekday: day of week (0=Sunday to 6=Saturday)
workingday: 1=working day, 0=weekend/holiday
weathersit: 1=clear, 2=mist, 3=light rain/snow, 4=heavy rain
temp: normalized temperature
atemp: normalized feel-like temperature
hum: normalized humidity
windspeed: normalized wind speed
casual: count of casual users
registered: count of registered users
cnt: total rental count (TARGET VARIABLE)
""")

# Visualize target distribution
print("\n[VISUALIZATION] Creating plots...")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1: Distribution of bike rentals (target)
axes[0, 0].hist(df['cnt'], bins=30, color='steelblue', edgecolor='black')
axes[0, 0].set_xlabel('Bike Rentals (cnt)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Distribution of Daily Bike Rentals')

# Plot 2: Rentals by season
df.boxplot(column='cnt', by='season', ax=axes[0, 1])
axes[0, 1].set_xlabel('Season (1=Spring, 2=Summer, 3=Fall, 4=Winter)')
axes[0, 1].set_ylabel('Bike Rentals')
axes[0, 1].set_title('Rentals by Season')
plt.sca(axes[0, 1])
plt.xticks([1, 2, 3, 4])

# Plot 3: Temperature vs Rentals (correlation check)
axes[1, 0].scatter(df['temp'], df['cnt'], alpha=0.5, color='coral')
axes[1, 0].set_xlabel('Normalized Temperature')
axes[1, 0].set_ylabel('Bike Rentals')
axes[1, 0].set_title('Temperature vs Rentals (Correlation)')

# Plot 4: Correlation heatmap
corr_cols = ['temp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']
correlation_matrix = df[corr_cols].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1, 1])
axes[1, 1].set_title('Correlation Matrix')

plt.tight_layout()
plt.savefig('results/session_1_eda.png', dpi=100, bbox_inches='tight')
print("✓ Saved plots to results/session_1_eda.png")
plt.close()

# ============================================================================
# STEP 4: Data Preprocessing (Leakage-Safe)
# ============================================================================
print("\n[STEP 4] Data Preprocessing...")

# Create a copy to avoid modifying original
df_processed = df.copy()

# Drop columns that won't help or could cause leakage:
# - instant, dteday: just identifiers
# - casual, registered: These sum to cnt, so they'd be perfect predictors (data leakage!)
#   If we use them, we're not really "predicting" — we already know the answer.
drop_cols = ['instant', 'dteday', 'casual', 'registered']
df_processed = df_processed.drop(columns=drop_cols)

print(f"✓ Dropped leakage-prone columns: {drop_cols}")

# Separate features (X) and target (y)
X = df_processed.drop(columns=['cnt'])
y = df_processed['cnt']

print(f"✓ Features shape: {X.shape}")
print(f"✓ Target shape: {y.shape}")

# ============================================================================
# STEP 5: Train-Test Split (Before Scaling!)
# ============================================================================
print("\n[STEP 5] Train-Test Split (80-20)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"✓ Train set: {X_train.shape[0]} samples")
print(f"✓ Test set: {X_test.shape[0]} samples")

# ============================================================================
# STEP 6: Feature Scaling (Fit Only on Train!)
# ============================================================================
print("\n[STEP 6] Feature Scaling...")
print("   (Fitting scaler on TRAIN data only, then applying to TEST)")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit AND transform on train
X_test_scaled = scaler.transform(X_test)        # ONLY transform test (no fit!)

print("✓ Scaling complete (no data leakage!)")

# Convert back to DataFrames for clarity
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

# ============================================================================
# STEP 7: Save Processed Data
# ============================================================================
print("\n[STEP 7] Saving processed data...")

os.makedirs("data/processed", exist_ok=True)

X_train_scaled.to_csv("data/processed/X_train.csv", index=False)
X_test_scaled.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print("✓ Saved:")
print("  - data/processed/X_train.csv")
print("  - data/processed/X_test.csv")
print("  - data/processed/y_train.csv")
print("  - data/processed/y_test.csv")



print(f"\nDataset: Bike Sharing (UCI)")
print(f"Total samples: {len(df)}")
print(f"Train samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features: {list(X.columns)}")
print(f"Target: cnt (bike rentals)")
