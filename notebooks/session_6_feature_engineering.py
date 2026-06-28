"""
CycleFlow - Session 6: Feature Engineering
===========================================
Goal: Create domain-specific features, retrain models, measure improvement
Time: ~1 hour
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os

# ============================================================================
# STEP 1: Load Original Data (Raw, before scaling)
# ============================================================================
print("\n[STEP 1] Loading raw data for feature engineering...")

# Load raw data (before scaling, so we can create features from raw values)
df_raw = pd.read_csv("data/raw/day.csv")

# Drop leakage columns (same as Session 1)
drop_cols = ['instant', 'dteday', 'casual', 'registered']
df_engineered = df_raw.drop(columns=drop_cols)

# Separate features and target
X_engineered = df_engineered.drop(columns=['cnt'])
y = df_engineered['cnt']

print(f"✓ Loaded raw data: {X_engineered.shape}")
print(f"   Original features: {list(X_engineered.columns)}")

# ============================================================================
# STEP 2: Create Domain-Specific Features
# ============================================================================
print("\n[STEP 2] Creating domain-specific features...")

# Feature 1: Temperature-Humidity Comfort Index
# Logic: High temp + Low humidity = comfortable = more rentals
# Formula: temp * (1 - humidity) gives us "comfort"
X_engineered['temp_humidity_comfort'] = X_engineered['temp'] * (1 - X_engineered['hum'])
print(f"\n   Feature 1: temp_humidity_comfort")
print(f"      Logic: High temp × Low humidity = comfortable weather")
print(f"      Formula: temp × (1 - humidity)")
print(f"      Example values: {X_engineered['temp_humidity_comfort'].head(3).values}")

# Feature 2: Season-Temperature Interaction
# Logic: Temperature effect is different in each season
# Summer: high temp = good
# Winter: high temp = unusual, bad
X_engineered['season_temp'] = X_engineered['season'] * X_engineered['temp']
print(f"\n   Feature 2: season_temp")
print(f"      Logic: Temperature matters differently by season")
print(f"      Formula: season × temp")
print(f"      Example values: {X_engineered['season_temp'].head(3).values}")

# Feature 3: Weekend Flag
# Logic: Weekends might have different rental patterns than weekdays
# weekday: 0=Sunday, 6=Saturday (weekends)
# weekday: 1-5 = weekdays
X_engineered['is_weekend'] = ((X_engineered['weekday'] == 0) | (X_engineered['weekday'] == 6)).astype(int)
print(f"\n   Feature 3: is_weekend")
print(f"      Logic: Weekend vs weekday patterns differ")
print(f"      Formula: 1 if weekday in [0, 6], else 0")
print(f"      Example values: {X_engineered['is_weekend'].head(3).values}")

# Feature 4: Weather-Temperature Combined
# Logic: Bad weather + cold = very few rentals
# Good weather + warm = many rentals
X_engineered['weather_temp'] = (5 - X_engineered['weathersit']) * X_engineered['temp']
print(f"\n   Feature 4: weather_temp")
print(f"      Logic: Good weather amplifies temperature effect")
print(f"      Formula: (5 - weathersit) × temp")
print(f"      Example values: {X_engineered['weather_temp'].head(3).values}")

print(f"\n✓ Created 4 new features")
print(f"✓ Total features now: {X_engineered.shape[1]} (was 11, now {X_engineered.shape[1]})")

# ============================================================================
# STEP 3: Prepare Data (Train/Test Split + Scaling)
# ============================================================================
print("\n[STEP 3] Preparing data (split + scale)...")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train_eng, X_test_eng, y_train_eng, y_test_eng = train_test_split(
    X_engineered, y, test_size=0.2, random_state=42
)

# Fit scaler ONLY on training data (no leakage)
scaler_eng = StandardScaler()
X_train_eng_scaled = scaler_eng.fit_transform(X_train_eng)
X_test_eng_scaled = scaler_eng.transform(X_test_eng)

# Convert back to DataFrame
X_train_eng_scaled = pd.DataFrame(X_train_eng_scaled, columns=X_engineered.columns, index=X_train_eng.index)
X_test_eng_scaled = pd.DataFrame(X_test_eng_scaled, columns=X_engineered.columns, index=X_test_eng.index)

print(f"✓ Train set: {X_train_eng_scaled.shape}")
print(f"✓ Test set: {X_test_eng_scaled.shape}")

# ============================================================================
# STEP 4: Load Original Preprocessed Data for Comparison
# ============================================================================
print("\n[STEP 4] Loading original data (without engineered features)...")

X_train_orig = pd.read_csv("data/processed/X_train.csv")
X_test_orig = pd.read_csv("data/processed/X_test.csv")
y_train_orig = pd.read_csv("data/processed/y_train.csv").values.ravel()
y_test_orig = pd.read_csv("data/processed/y_test.csv").values.ravel()

print(f"✓ Original features: {X_train_orig.shape[1]}")
print(f"✓ Engineered features: {X_train_eng_scaled.shape[1]}")

# ============================================================================
# STEP 5: Train Gradient Boosting (Original Features)
# ============================================================================
print("\n[STEP 5] Training Gradient Boosting with ORIGINAL features...")

# Load best params from Session 5
gb_orig = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=5,
    random_state=42
)

gb_orig.fit(X_train_orig, y_train_orig)
y_pred_orig = gb_orig.predict(X_test_orig)

r2_orig = r2_score(y_test_orig, y_pred_orig)
rmse_orig = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))

print(f"   R² Score (Original): {r2_orig:.4f}")
print(f"   RMSE (Original): {rmse_orig:.2f}")

# Cross-validation on original
cv_scores_orig = cross_val_score(gb_orig, X_train_orig, y_train_orig, cv=5, scoring='r2', n_jobs=-1)
cv_mean_orig = cv_scores_orig.mean()
print(f"   5-Fold CV R² (Original): {cv_mean_orig:.4f}")

# ============================================================================
# STEP 6: Train Gradient Boosting (Engineered Features)
# ============================================================================
print("\n[STEP 6] Training Gradient Boosting with ENGINEERED features...")

gb_eng = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=5,
    random_state=42
)

gb_eng.fit(X_train_eng_scaled, y_train_eng)
y_pred_eng = gb_eng.predict(X_test_eng_scaled)

r2_eng = r2_score(y_test_eng, y_pred_eng)
rmse_eng = np.sqrt(mean_squared_error(y_test_eng, y_pred_eng))

print(f"   R² Score (Engineered): {r2_eng:.4f}")
print(f"   RMSE (Engineered): {rmse_eng:.2f}")

# Cross-validation on engineered
cv_scores_eng = cross_val_score(gb_eng, X_train_eng_scaled, y_train_eng, cv=5, scoring='r2', n_jobs=-1)
cv_mean_eng = cv_scores_eng.mean()
print(f"   5-Fold CV R² (Engineered): {cv_mean_eng:.4f}")

# ============================================================================
# STEP 7: Compare Results
# ============================================================================
print("\n[STEP 7] Comparing original vs engineered features...")
print("\n" + "="*70)
print("FEATURE ENGINEERING IMPACT")
print("="*70)

comparison = pd.DataFrame({
    'Metric': ['Test R²', 'Test RMSE', 'CV R² Mean', 'CV R² Std'],
    'Original Features (11)': [f'{r2_orig:.4f}', f'{rmse_orig:.2f}', f'{cv_mean_orig:.4f}', f'{cv_scores_orig.std():.4f}'],
    'Engineered Features (15)': [f'{r2_eng:.4f}', f'{rmse_eng:.2f}', f'{cv_mean_eng:.4f}', f'{cv_scores_eng.std():.4f}'],
    'Improvement': [
        f'{r2_eng - r2_orig:+.4f}',
        f'{rmse_eng - rmse_orig:+.2f}',
        f'{cv_mean_eng - cv_mean_orig:+.4f}',
        f'{cv_scores_eng.std() - cv_scores_orig.std():+.4f}'
    ]
})

print("\n" + comparison.to_string(index=False))

# ============================================================================
# STEP 8: Feature Importance Analysis
# ============================================================================
print("\n[STEP 8] Analyzing feature importance...")

# Get feature importances from engineered model
feature_importance = pd.DataFrame({
    'Feature': X_train_eng_scaled.columns,
    'Importance': gb_eng.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n--- Top 10 Most Important Features ---")
print(feature_importance.head(10).to_string(index=False))

# Highlight new engineered features
engineered_features = ['temp_humidity_comfort', 'season_temp', 'is_weekend', 'weather_temp']
engineered_importances = feature_importance[feature_importance['Feature'].isin(engineered_features)]

print("\n--- Engineered Features Importance ---")
print(engineered_importances.to_string(index=False))
print(f"\nEngineered features rank:")
for idx, (i, row) in enumerate(engineered_importances.iterrows(), 1):
    overall_rank = feature_importance[feature_importance['Feature'] == row['Feature']].index[0] + 1
    print(f"  {row['Feature']}: #{overall_rank} out of 15 features")

# ============================================================================
# STEP 9: Visualizations
# ============================================================================
print("\n[STEP 9] Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Test R² Comparison
ax = axes[0, 0]
models = ['Original\nFeatures (11)', 'Engineered\nFeatures (15)']
r2_scores = [r2_orig, r2_eng]
colors = ['#ff7f0e', '#2ca02c']
bars = ax.bar(models, r2_scores, color=colors, edgecolor='black', linewidth=2, alpha=0.8)
ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax.set_title('Test Set R² Comparison', fontsize=13, fontweight='bold')
ax.set_ylim([0.88, 0.92])
ax.grid(axis='y', alpha=0.3)

for bar, score in zip(bars, r2_scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.0005,
            f'{score:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add improvement annotation
improvement = r2_eng - r2_orig
ax.annotate(f'Improvement:\n{improvement:+.4f}', 
            xy=(0.5, 0.90), fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
            ha='center')

# Plot 2: CV R² Comparison (with error bars)
ax = axes[0, 1]
cv_means = [cv_mean_orig, cv_mean_eng]
cv_stds = [cv_scores_orig.std(), cv_scores_eng.std()]
bars = ax.bar(models, cv_means, yerr=cv_stds, capsize=5, color=colors, edgecolor='black', linewidth=2, alpha=0.8)
ax.set_ylabel('CV R² Score (5-fold)', fontsize=12, fontweight='bold')
ax.set_title('Cross-Validation R² Comparison', fontsize=13, fontweight='bold')
ax.set_ylim([0.85, 0.92])
ax.grid(axis='y', alpha=0.3)

for bar, mean in zip(bars, cv_means):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.003,
            f'{mean:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Plot 3: Top 10 Feature Importance
ax = axes[1, 0]
top_10 = feature_importance.head(10)
colors_feat = ['#2ca02c' if feat in engineered_features else '#1f77b4' 
               for feat in top_10['Feature']]
ax.barh(range(len(top_10)), top_10['Importance'].values, color=colors_feat, edgecolor='black', linewidth=1)
ax.set_yticks(range(len(top_10)))
ax.set_yticklabels(top_10['Feature'].values, fontsize=10)
ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
ax.set_title('Top 10 Most Important Features\n(Green = Engineered)', fontsize=13, fontweight='bold')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

# Plot 4: Engineered Features Importance
ax = axes[1, 1]
eng_feats = engineered_importances.sort_values('Importance', ascending=True)
ax.barh(range(len(eng_feats)), eng_feats['Importance'].values, color='#2ca02c', edgecolor='black', linewidth=1.5)
ax.set_yticks(range(len(eng_feats)))
ax.set_yticklabels(eng_feats['Feature'].values, fontsize=11, fontweight='bold')
ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
ax.set_title('Engineered Features: Relative Importance', fontsize=13, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('results/session_6_feature_engineering.png', dpi=100, bbox_inches='tight')
print("✓ Saved plot to results/session_6_feature_engineering.png")
plt.close()

# ============================================================================
# STEP 10: Save Engineered Model
# ============================================================================
print("\n[STEP 10] Saving engineered model and data...")

joblib.dump(gb_eng, 'models/gradient_boosting_engineered.pkl')
print("✓ Saved engineered model to models/gradient_boosting_engineered.pkl")

# Save engineered data for Session 7
X_train_eng_scaled.to_csv('data/processed/X_train_engineered.csv', index=False)
X_test_eng_scaled.to_csv('data/processed/X_test_engineered.csv', index=False)
pd.DataFrame(y_train_eng).to_csv('data/processed/y_train_engineered.csv', index=False)
pd.DataFrame(y_test_eng).to_csv('data/processed/y_test_engineered.csv', index=False)
print("✓ Saved engineered data to data/processed/")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("SESSION 6 COMPLETE ✓")
print("="*70)

print(f"\n--- Feature Engineering Summary ---")
print(f"Original features: 11")
print(f"New engineered features: 4")
print(f"Total features: 15")

print(f"\n--- Performance Improvement ---")
print(f"Test R² improvement: {r2_eng - r2_orig:+.4f}")
print(f"Test RMSE change: {rmse_eng - rmse_orig:+.2f} rentals")
print(f"CV R² improvement: {cv_mean_eng - cv_mean_orig:+.4f}")

print(f"\n--- Best Engineered Feature ---")
best_eng = engineered_importances.iloc[0]
print(f"{best_eng['Feature']}: importance = {best_eng['Importance']:.4f}")

