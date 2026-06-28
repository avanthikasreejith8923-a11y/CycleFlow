"""
CycleFlow - Session 3: Train Ensemble Models (Random Forest, Gradient Boosting, XGBoost)
=======================================================================================
Goal: Train 3 ensemble models, compare with Session 2 results
Time: ~2 hours
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
import os

# ============================================================================
# STEP 1: Load Preprocessed Data
# ============================================================================
print("\n[STEP 1] Loading preprocessed data...")

X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

print(f"✓ Loaded training data: {X_train.shape}")
print(f"✓ Loaded test data: {X_test.shape}")

# ============================================================================
# STEP 2: Train Random Forest
# ============================================================================
print("\n[STEP 2] Training Random Forest...")
print("   (100 decision trees, each trained on random data subsets)")
print("   (Final prediction = average of all 100 trees)")

rf_model = RandomForestRegressor(
    n_estimators=100,      # 100 trees
    max_depth=15,          # Max depth of each tree (prevents overfitting)
    min_samples_split=5,   # Min samples to split a node (prevents overfitting)
    random_state=42,
    n_jobs=-1              # Use all CPU cores (faster)
)

rf_model.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test)

# Calculate metrics
r2_rf = r2_score(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

print(f"   R² Score: {r2_rf:.4f}")
print(f"   RMSE: {rmse_rf:.2f} rentals")

# Feature importance in Random Forest
rf_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n   Top 5 Most Important Features:")
print(rf_importance.head().to_string(index=False))

# ============================================================================
# STEP 3: Train Gradient Boosting
# ============================================================================
print("\n[STEP 3] Training Gradient Boosting...")
print("   (Trees built sequentially)")
print("   (Each tree learns from previous tree's mistakes)")

gb_model = GradientBoostingRegressor(
    n_estimators=100,      # 100 trees built one at a time
    learning_rate=0.1,     # Step size (how much each tree corrects)
    max_depth=5,           # Shallow trees (prevents overfitting)
    min_samples_split=5,
    random_state=42
)

gb_model.fit(X_train, y_train)

# Make predictions
y_pred_gb = gb_model.predict(X_test)

# Calculate metrics
r2_gb = r2_score(y_test, y_pred_gb)
rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))

print(f"   R² Score: {r2_gb:.4f}")
print(f"   RMSE: {rmse_gb:.2f} rentals")

# Feature importance in Gradient Boosting
gb_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': gb_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n   Top 5 Most Important Features:")
print(gb_importance.head().to_string(index=False))

# ============================================================================
# STEP 4: Train XGBoost
# ============================================================================
print("\n[STEP 4] Training XGBoost (Extreme Gradient Boosting)...")
print("   (Improved version of Gradient Boosting)")
print("   (Faster + better regularization to prevent overfitting)")

xgb_model = XGBRegressor(
    n_estimators=100,      # 100 trees
    learning_rate=0.1,     # Step size (same as GB for fair comparison)
    max_depth=5,           # Shallow trees
    min_child_weight=1,    # Min samples in leaf (overfitting prevention)
    subsample=0.8,         # Use 80% of data per tree (overfitting prevention)
    colsample_bytree=0.8,  # Use 80% of features per tree (overfitting prevention)
    random_state=42,
    verbosity=0            # Suppress verbose output
)

xgb_model.fit(X_train, y_train)

# Make predictions
y_pred_xgb = xgb_model.predict(X_test)

# Calculate metrics
r2_xgb = r2_score(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))

print(f"   R² Score: {r2_xgb:.4f}")
print(f"   RMSE: {rmse_xgb:.2f} rentals")

# Feature importance in XGBoost
xgb_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n   Top 5 Most Important Features:")
print(xgb_importance.head().to_string(index=False))

# ============================================================================
# STEP 5: Load Session 2 Results for Comparison
# ============================================================================
print("\n[STEP 5] Loading Session 2 results for comparison...")

session2_results = pd.DataFrame({
    'Model': ['Linear Regression', 'Ridge', 'Lasso', 'SVR'],
    'R² Score': [0.827667, 0.827649, 0.827657, 0.810893],
    'RMSE': [831.285155, 831.329431, 831.309812, 870.801398]
})

print("\nSession 2 Results:")
print(session2_results.to_string(index=False))

# ============================================================================
# STEP 6: Compare ALL 7 Models (Sessions 2 + 3)
# ============================================================================
print("\n" + "="*70)
print("COMPLETE COMPARISON: ALL 7 MODELS")
print("="*70)

all_results = pd.DataFrame({
    'Model': ['Linear Regression', 'Ridge', 'Lasso', 'SVR', 'Random Forest', 'Gradient Boosting', 'XGBoost'],
    'R² Score': [0.827667, 0.827649, 0.827657, 0.810893, r2_rf, r2_gb, r2_xgb],
    'RMSE': [831.285155, 831.329431, 831.309812, 870.801398, rmse_rf, rmse_gb, rmse_xgb]
})

print("\n" + all_results.to_string(index=False))

# Find best overall model
best_model_idx = all_results['R² Score'].idxmax()
best_model_name = all_results.loc[best_model_idx, 'Model']
best_r2 = all_results.loc[best_model_idx, 'R² Score']

print(f"\n🏆 OVERALL BEST MODEL: {best_model_name} (R² = {best_r2:.4f})")

# ============================================================================
# STEP 7: Visualize Comparison
# ============================================================================
print("\n[STEP 7] Creating comparison visualization...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: R² Comparison (All 7 Models)
colors_session2 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
colors_session3 = ['#9467bd', '#8c564b', '#e377c2']
all_colors = colors_session2 + colors_session3

axes[0].bar(range(len(all_results)), all_results['R² Score'], color=all_colors, edgecolor='black', linewidth=1.5)
axes[0].set_ylabel('R² Score', fontsize=13, fontweight='bold')
axes[0].set_title('R² Score: All 7 Models (Higher is Better)', fontsize=14, fontweight='bold')
axes[0].set_xticks(range(len(all_results)))
axes[0].set_xticklabels(all_results['Model'], rotation=45, ha='right', fontsize=10)
axes[0].set_ylim([0.75, 0.85])
axes[0].axhline(y=0.8277, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Linear Reg baseline (0.8277)')
axes[0].grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, v in enumerate(all_results['R² Score']):
    axes[0].text(i, v + 0.001, f'{v:.4f}', ha='center', fontsize=9, fontweight='bold')

axes[0].legend(fontsize=10)

# Plot 2: RMSE Comparison (All 7 Models)
axes[1].bar(range(len(all_results)), all_results['RMSE'], color=all_colors, edgecolor='black', linewidth=1.5)
axes[1].set_ylabel('RMSE (rentals)', fontsize=13, fontweight='bold')
axes[1].set_title('RMSE: All 7 Models (Lower is Better)', fontsize=14, fontweight='bold')
axes[1].set_xticks(range(len(all_results)))
axes[1].set_xticklabels(all_results['Model'], rotation=45, ha='right', fontsize=10)
axes[1].grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, v in enumerate(all_results['RMSE']):
    axes[1].text(i, v + 10, f'{v:.0f}', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('results/session_3_all_models_comparison.png', dpi=100, bbox_inches='tight')
print("✓ Saved plot to results/session_3_all_models_comparison.png")
plt.close()

# ============================================================================
# STEP 8: Feature Importance Comparison
# ============================================================================
print("\n[STEP 8] Creating feature importance comparison...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, importance_df, title in zip(
    axes,
    [rf_importance, gb_importance, xgb_importance],
    ['Random Forest', 'Gradient Boosting', 'XGBoost']
):
    top_features = importance_df.head(8)
    ax.barh(range(len(top_features)), top_features['Importance'].values, color='steelblue', edgecolor='black')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['Feature'].values, fontsize=10)
    ax.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
    ax.set_title(f'{title} - Top 8 Features', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('results/session_3_feature_importance.png', dpi=100, bbox_inches='tight')
print("✓ Saved feature importance plot to results/session_3_feature_importance.png")
plt.close()

# ============================================================================
# STEP 9: Save Results
# ============================================================================
print("\n[STEP 9] Saving results...")

all_results.to_csv('results/all_7_models_comparison.csv', index=False)
print("✓ Saved results to results/all_7_models_comparison.csv")

# Save ensemble predictions
ensemble_predictions = pd.DataFrame({
    'y_true': y_test,
    'RandomForest_pred': y_pred_rf,
    'GradientBoosting_pred': y_pred_gb,
    'XGBoost_pred': y_pred_xgb
})
ensemble_predictions.to_csv('results/ensemble_predictions.csv', index=False)
print("✓ Saved predictions to results/ensemble_predictions.csv")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("SESSION 3 COMPLETE ✓")
print("="*70)

print(f"\nDataset: Bike Sharing")
print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
print(f"Features: {X_train.shape[1]}")

print(f"\n--- SESSION 3 RESULTS (Ensemble Models) ---")
print(f"Random Forest:      R² = {r2_rf:.4f}, RMSE = {rmse_rf:.2f}")
print(f"Gradient Boosting:  R² = {r2_gb:.4f}, RMSE = {rmse_gb:.2f}")
print(f"XGBoost:            R² = {r2_xgb:.4f}, RMSE = {rmse_xgb:.2f}")

print(f"\n--- BEST FROM SESSION 2 (Linear Models) ---")
print(f"Linear Regression:  R² = 0.8277, RMSE = 831.29")

print(f"\n--- YOUR PREDICTION (Session 3 Ranking) ---")
print(f"#1: Gradient Boosting")
print(f"#2: XGBoost")
print(f"#3: Random Forest")

print(f"\n--- ACTUAL RANKING (Session 3) ---")
session3_only = all_results.iloc[4:].sort_values('R² Score', ascending=False)
for idx, (i, row) in enumerate(session3_only.iterrows(), 1):
    print(f"#{idx}: {row['Model']} (R² = {row['R² Score']:.4f})")

print(f"\n🏆 OVERALL WINNER: {best_model_name} (R² = {best_r2:.4f})")

# Check if user's Session 3 ranking was correct
session3_ranking = all_results.iloc[4:].sort_values('R² Score', ascending=False)['Model'].tolist()
user_prediction = ['Gradient Boosting', 'XGBoost', 'Random Forest']

if session3_ranking == user_prediction:
    print("\n✓ YOUR SESSION 3 RANKING WAS CORRECT!")
else:
    print(f"\n✗ Your ranking was different. Actual: {session3_ranking}")

