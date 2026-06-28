"""
CycleFlow - Session 2: Train Linear Models (Linear, Ridge, Lasso, SVR)
======================================================================
Goal: Train 4 models, compare their R² and RMSE scores
Time: ~1.5-2 hours
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
import os

# ============================================================================
# STEP 1: Load Preprocessed Data from Session 1
# ============================================================================
print("\n[STEP 1] Loading preprocessed data...")

X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

print(f"✓ Loaded training data: {X_train.shape}")
print(f"✓ Loaded test data: {X_test.shape}")

# ============================================================================
# STEP 2: Train Linear Regression (Baseline)
# ============================================================================
print("\n[STEP 2] Training Linear Regression...")
print("   (No regularization — just fits a straight line through all features)")

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions
y_pred_lr = lr_model.predict(X_test)

# Calculate metrics
r2_lr = r2_score(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))

print(f"   R² Score: {r2_lr:.4f}")
print(f"   RMSE: {rmse_lr:.2f} rentals")
print(f"   (R² = 1.0 is perfect, 0.0 is terrible)")

# ============================================================================
# STEP 3: Train Ridge Regression
# ============================================================================
print("\n[STEP 3] Training Ridge Regression...")
print("   (Penalizes large coefficients — safer than Linear Regression)")
print("   (Keeps all 11 features, but doesn't rely too heavily on any one)")

# Ridge has a hyperparameter 'alpha' (regularization strength)
# Higher alpha = stronger penalty = more conservative
ridge_model = Ridge(alpha=1.0)  # Standard value; we can tune this later
ridge_model.fit(X_train, y_train)

# Make predictions
y_pred_ridge = ridge_model.predict(X_test)

# Calculate metrics
r2_ridge = r2_score(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))

print(f"   R² Score: {r2_ridge:.4f}")
print(f"   RMSE: {rmse_ridge:.2f} rentals")

# ============================================================================
# STEP 4: Train Lasso Regression
# ============================================================================
print("\n[STEP 4] Training Lasso Regression...")
print("   (Like Ridge, but REMOVES useless features by setting coeff to 0)")
print("   (Automatic feature selection)")

lasso_model = Lasso(alpha=0.1)  # Lower alpha than Ridge (Lasso is harsher)
lasso_model.fit(X_train, y_train)

# Make predictions
y_pred_lasso = lasso_model.predict(X_test)

# Calculate metrics
r2_lasso = r2_score(y_test, y_pred_lasso)
rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))

print(f"   R² Score: {r2_lasso:.4f}")
print(f"   RMSE: {rmse_lasso:.2f} rentals")

# Check which features Lasso kept
lasso_coefs = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': lasso_model.coef_
})
print("\n   Lasso Feature Coefficients:")
print(lasso_coefs)
print(f"\n   Features removed (coef = 0): {(lasso_coefs['Coefficient'] == 0).sum()}")

# ============================================================================
# STEP 5: Train SVR (Support Vector Regression)
# ============================================================================
print("\n[STEP 5] Training SVR (Nonlinear)...")
print("   (Doesn't assume linear relationship)")
print("   (Can capture curved patterns)")

# SVR kernel options: 'linear', 'rbf' (curved), 'poly' (polynomial)
# 'rbf' = Radial Basis Function (most flexible)
svr_model = SVR(kernel='rbf', C=100, gamma='scale')
svr_model.fit(X_train, y_train)

# Make predictions
y_pred_svr = svr_model.predict(X_test)

# Calculate metrics
r2_svr = r2_score(y_test, y_pred_svr)
rmse_svr = np.sqrt(mean_squared_error(y_test, y_pred_svr))

print(f"   R² Score: {r2_svr:.4f}")
print(f"   RMSE: {rmse_svr:.2f} rentals")

# ============================================================================
# STEP 6: Compare All 4 Models
# ============================================================================
print("\n" + "="*70)
print("MODEL COMPARISON")
print("="*70)

results_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Ridge', 'Lasso', 'SVR'],
    'R² Score': [r2_lr, r2_ridge, r2_lasso, r2_svr],
    'RMSE': [rmse_lr, rmse_ridge, rmse_lasso, rmse_svr]
})

print("\n" + results_df.to_string(index=False))

# Find best model
best_model_idx = results_df['R² Score'].idxmax()
best_model_name = results_df.loc[best_model_idx, 'Model']
best_r2 = results_df.loc[best_model_idx, 'R² Score']

print(f"\n🏆 Best Model: {best_model_name} (R² = {best_r2:.4f})")

# ============================================================================
# STEP 7: Visualize Comparison
# ============================================================================
print("\n[STEP 7] Creating comparison visualization...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: R² Comparison
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
axes[0].bar(results_df['Model'], results_df['R² Score'], color=colors, edgecolor='black')
axes[0].set_ylabel('R² Score', fontsize=12)
axes[0].set_title('R² Score Comparison (Higher is Better)', fontsize=13, fontweight='bold')
axes[0].set_ylim([0, 1])
axes[0].axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Good threshold')
for i, v in enumerate(results_df['R² Score']):
    axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 2: RMSE Comparison
axes[1].bar(results_df['Model'], results_df['RMSE'], color=colors, edgecolor='black')
axes[1].set_ylabel('RMSE (rentals)', fontsize=12)
axes[1].set_title('RMSE Comparison (Lower is Better)', fontsize=13, fontweight='bold')
for i, v in enumerate(results_df['RMSE']):
    axes[1].text(i, v + 50, f'{v:.0f}', ha='center', fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('results/session_2_model_comparison.png', dpi=100, bbox_inches='tight')
print("✓ Saved plot to results/session_2_model_comparison.png")
plt.close()

# ============================================================================
# STEP 8: Save Results
# ============================================================================
print("\n[STEP 8] Saving results...")

results_df.to_csv('results/model_comparison_results.csv', index=False)
print("✓ Saved results to results/model_comparison_results.csv")

# Save model predictions for later analysis
predictions_df = pd.DataFrame({
    'y_true': y_test,
    'LinearReg_pred': y_pred_lr,
    'Ridge_pred': y_pred_ridge,
    'Lasso_pred': y_pred_lasso,
    'SVR_pred': y_pred_svr
})
predictions_df.to_csv('results/model_predictions.csv', index=False)
print("✓ Saved predictions to results/model_predictions.csv")


