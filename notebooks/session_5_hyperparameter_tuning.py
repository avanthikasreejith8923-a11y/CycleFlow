"""
CycleFlow - Session 5: Hyperparameter Tuning + Cross-Validation
================================================================
Goal: Find best hyperparameters using RandomSearchCV + 5-fold CV
Time: ~2 hours (includes training time)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')
import os
import joblib

# ============================================================================
# STEP 1: Load Data
# ============================================================================
print("\n[STEP 1] Loading preprocessed data...")

X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

print(f"✓ Loaded training data: {X_train.shape}")
print(f"✓ Loaded test data: {X_test.shape}")

# ============================================================================
# STEP 2: 5-Fold Cross-Validation on All 7 Models (Default Params)
# ============================================================================
print("\n[STEP 2] Testing all 7 models with 5-fold cross-validation (default params)...")
print("   (This trains each model 5 times on different data splits)")

# Default models
models_default = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1),
    'SVR': SVR(kernel='rbf', C=100, gamma='scale'),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, verbosity=0)
}

cv_results_default = {}

for name, model in models_default.items():
    print(f"\n   Training {name}...")
    
    # 5-fold cross-validation (scoring metric = R²)
    cv_scores = cross_val_score(
        model, 
        X_train, 
        y_train, 
        cv=5,           # 5 folds
        scoring='r2',   # R² score
        n_jobs=-1       # Use all CPU cores
    )
    
    mean_cv_score = cv_scores.mean()
    std_cv_score = cv_scores.std()
    
    cv_results_default[name] = {
        'cv_scores': cv_scores,
        'mean': mean_cv_score,
        'std': std_cv_score
    }
    
    print(f"      CV Scores (5 folds): {[f'{s:.4f}' for s in cv_scores]}")
    print(f"      Mean CV R²: {mean_cv_score:.4f} (±{std_cv_score:.4f})")

# ============================================================================
# STEP 3: Hyperparameter Tuning for Gradient Boosting (RandomSearchCV)
# ============================================================================
print("\n[STEP 3] Tuning Gradient Boosting with RandomSearchCV...")
print("   (Testing 20 random hyperparameter combinations)")

# Define hyperparameter distribution to search
param_dist_gb = {
    'n_estimators': [50, 75, 100, 150, 200],
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'max_depth': [3, 4, 5, 6, 7],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.7, 0.8, 0.9, 1.0]
}

gb_base = GradientBoostingRegressor(random_state=42)

# RandomSearchCV: try 20 random combinations
random_search_gb = RandomizedSearchCV(
    gb_base,
    param_distributions=param_dist_gb,
    n_iter=20,                    # Try 20 random combinations
    cv=5,                         # 5-fold cross-validation
    scoring='r2',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

print("\n   Running RandomSearchCV (this takes ~2-3 minutes)...")
random_search_gb.fit(X_train, y_train)

best_params_gb = random_search_gb.best_params_
best_cv_score_gb = random_search_gb.best_score_

print(f"\n   ✓ Best hyperparameters found:")
for param, value in best_params_gb.items():
    print(f"      {param}: {value}")
print(f"\n   Best CV R² Score: {best_cv_score_gb:.4f}")

# ============================================================================
# STEP 4: Train Tuned Gradient Boosting Model
# ============================================================================
print("\n[STEP 4] Training Gradient Boosting with best hyperparameters...")

gb_tuned = GradientBoostingRegressor(**best_params_gb, random_state=42)
gb_tuned.fit(X_train, y_train)

# Evaluate on test set
y_pred_gb_tuned = gb_tuned.predict(X_test)
r2_gb_tuned = r2_score(y_test, y_pred_gb_tuned)
rmse_gb_tuned = np.sqrt(mean_squared_error(y_test, y_pred_gb_tuned))

print(f"   Test R² Score: {r2_gb_tuned:.4f}")
print(f"   Test RMSE: {rmse_gb_tuned:.2f}")

# Compare with default
gb_default = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
gb_default.fit(X_train, y_train)
y_pred_gb_default = gb_default.predict(X_test)
r2_gb_default = r2_score(y_test, y_pred_gb_default)
rmse_gb_default = np.sqrt(mean_squared_error(y_test, y_pred_gb_default))

print(f"\n   Default Gradient Boosting:")
print(f"      Test R²: {r2_gb_default:.4f}")
print(f"      Test RMSE: {rmse_gb_default:.2f}")
print(f"\n   Improvement from tuning:")
print(f"      ΔR²: {r2_gb_tuned - r2_gb_default:+.4f}")
print(f"      ΔRMSE: {rmse_gb_tuned - rmse_gb_default:+.2f} rentals")

# ============================================================================
# STEP 5: Hyperparameter Tuning for Other Key Models
# ============================================================================
print("\n[STEP 5] Tuning Ridge, Lasso, and XGBoost...")

# Ridge Tuning
print("\n   Tuning Ridge...")
param_dist_ridge = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
ridge_search = RandomizedSearchCV(
    Ridge(), 
    param_dist_ridge, 
    n_iter=6, 
    cv=5, 
    scoring='r2', 
    n_jobs=-1,
    random_state=42
)
ridge_search.fit(X_train, y_train)
best_params_ridge = ridge_search.best_params_
best_cv_score_ridge = ridge_search.best_score_
print(f"      Best alpha: {best_params_ridge['alpha']}")
print(f"      Best CV R²: {best_cv_score_ridge:.4f}")

# Lasso Tuning
print("\n   Tuning Lasso...")
param_dist_lasso = {'alpha': [0.001, 0.01, 0.05, 0.1, 0.5]}
lasso_search = RandomizedSearchCV(
    Lasso(), 
    param_dist_lasso, 
    n_iter=5, 
    cv=5, 
    scoring='r2', 
    n_jobs=-1,
    random_state=42
)
lasso_search.fit(X_train, y_train)
best_params_lasso = lasso_search.best_params_
best_cv_score_lasso = lasso_search.best_score_
print(f"      Best alpha: {best_params_lasso['alpha']}")
print(f"      Best CV R²: {best_cv_score_lasso:.4f}")

# XGBoost Tuning
print("\n   Tuning XGBoost...")
param_dist_xgb = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.05, 0.1, 0.15],
    'max_depth': [4, 5, 6],
    'subsample': [0.7, 0.8, 0.9]
}
xgb_search = RandomizedSearchCV(
    XGBRegressor(random_state=42, verbosity=0),
    param_dist_xgb,
    n_iter=15,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    random_state=42,
    verbose=0
)
xgb_search.fit(X_train, y_train)
best_params_xgb = xgb_search.best_params_
best_cv_score_xgb = xgb_search.best_score_
print(f"      Best learning_rate: {best_params_xgb['learning_rate']}")
print(f"      Best max_depth: {best_params_xgb['max_depth']}")
print(f"      Best CV R²: {best_cv_score_xgb:.4f}")

# ============================================================================
# STEP 6: Compare Default vs Tuned Models
# ============================================================================
print("\n[STEP 6] Comparing default vs tuned hyperparameters...")
print("\n" + "="*70)
print("5-FOLD CROSS-VALIDATION COMPARISON (Training Set)")
print("="*70)

comparison_data = []

for name, model in models_default.items():
    mean_default = cv_results_default[name]['mean']
    std_default = cv_results_default[name]['std']
    
    comparison_data.append({
        'Model': name,
        'Mean CV R² (Default)': mean_default,
        'Std Dev': std_default
    })

comparison_df = pd.DataFrame(comparison_data)
print("\n" + comparison_df.to_string(index=False))

# ============================================================================
# STEP 7: Evaluate All Tuned Models on Test Set
# ============================================================================
print("\n[STEP 7] Evaluating tuned models on test set...")

# Train tuned models
ridge_tuned = Ridge(**best_params_ridge)
ridge_tuned.fit(X_train, y_train)
y_pred_ridge_tuned = ridge_tuned.predict(X_test)
r2_ridge_tuned = r2_score(y_test, y_pred_ridge_tuned)

lasso_tuned = Lasso(**best_params_lasso)
lasso_tuned.fit(X_train, y_train)
y_pred_lasso_tuned = lasso_tuned.predict(X_test)
r2_lasso_tuned = r2_score(y_test, y_pred_lasso_tuned)

xgb_tuned = XGBRegressor(**best_params_xgb, random_state=42, verbosity=0)
xgb_tuned.fit(X_train, y_train)
y_pred_xgb_tuned = xgb_tuned.predict(X_test)
r2_xgb_tuned = r2_score(y_test, y_pred_xgb_tuned)

print("\n" + "="*70)
print("TEST SET PERFORMANCE: DEFAULT vs TUNED")
print("="*70)

test_comparison = pd.DataFrame({
    'Model': ['Ridge', 'Lasso', 'Gradient Boosting', 'XGBoost'],
    'Default R²': [0.8276, 0.8277, 0.8971, 0.8958],
    'Tuned R²': [r2_ridge_tuned, r2_lasso_tuned, r2_gb_tuned, r2_xgb_tuned],
    'Improvement': [
        r2_ridge_tuned - 0.8276,
        r2_lasso_tuned - 0.8277,
        r2_gb_tuned - 0.8971,
        r2_xgb_tuned - 0.8958
    ]
})

print("\n" + test_comparison.to_string(index=False))

# ============================================================================
# STEP 8: Visualize Cross-Validation Results
# ============================================================================
print("\n[STEP 8] Creating visualization...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: CV Scores for Default Models
cv_means = [cv_results_default[name]['mean'] for name in models_default.keys()]
cv_stds = [cv_results_default[name]['std'] for name in models_default.keys()]

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
axes[0].bar(range(len(models_default)), cv_means, yerr=cv_stds, capsize=5, 
            color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
axes[0].set_ylabel('Mean CV R² Score', fontsize=12, fontweight='bold')
axes[0].set_title('5-Fold Cross-Validation: All Models (Default Params)', fontsize=13, fontweight='bold')
axes[0].set_xticks(range(len(models_default)))
axes[0].set_xticklabels(models_default.keys(), rotation=45, ha='right', fontsize=10)
axes[0].set_ylim([0.75, 0.95])
axes[0].grid(axis='y', alpha=0.3)

# Add value labels
for i, (mean, std) in enumerate(zip(cv_means, cv_stds)):
    axes[0].text(i, mean + std + 0.005, f'{mean:.4f}', ha='center', fontsize=9, fontweight='bold')

# Plot 2: Default vs Tuned (Test Set)
x = np.arange(len(test_comparison))
width = 0.35

axes[1].bar(x - width/2, test_comparison['Default R²'], width, label='Default', 
            color='lightcoral', edgecolor='black', linewidth=1.5)
axes[1].bar(x + width/2, test_comparison['Tuned R²'], width, label='Tuned', 
            color='lightgreen', edgecolor='black', linewidth=1.5)

axes[1].set_ylabel('R² Score (Test Set)', fontsize=12, fontweight='bold')
axes[1].set_title('Default vs Tuned Hyperparameters (Test Set)', fontsize=13, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(test_comparison['Model'], rotation=45, ha='right', fontsize=10)
axes[1].legend(fontsize=11)
axes[1].set_ylim([0.82, 0.92])
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/session_5_hyperparameter_tuning.png', dpi=100, bbox_inches='tight')
print("✓ Saved plot to results/session_5_hyperparameter_tuning.png")
plt.close()

# ============================================================================
# STEP 9: Save Tuned Models
# ============================================================================
print("\n[STEP 9] Saving tuned models...")

os.makedirs("models", exist_ok=True)

joblib.dump(gb_tuned, 'models/gradient_boosting_tuned.pkl')
joblib.dump(ridge_tuned, 'models/ridge_tuned.pkl')
joblib.dump(lasso_tuned, 'models/lasso_tuned.pkl')
joblib.dump(xgb_tuned, 'models/xgboost_tuned.pkl')

print("✓ Saved tuned models:")
print("  - models/gradient_boosting_tuned.pkl")
print("  - models/ridge_tuned.pkl")
print("  - models/lasso_tuned.pkl")
print("  - models/xgboost_tuned.pkl")

# ============================================================================
# STEP 10: Save Hyperparameters
# ============================================================================
print("\n[STEP 10] Saving best hyperparameters...")

hyperparams_summary = {
    'Ridge': best_params_ridge,
    'Lasso': best_params_lasso,
    'Gradient Boosting': best_params_gb,
    'XGBoost': best_params_xgb
}

import json
with open('results/best_hyperparameters.json', 'w') as f:
    json.dump(hyperparams_summary, f, indent=2)

print("✓ Saved to results/best_hyperparameters.json")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("SESSION 5 COMPLETE ✓")
print("="*70)

print(f"\n--- 5-Fold Cross-Validation Insights ---")
print(f"Best 5-fold CV score: {max(cv_means):.4f} ({list(models_default.keys())[cv_means.index(max(cv_means))]}")
print(f"Most consistent model (lowest std): {list(models_default.keys())[cv_stds.index(min(cv_stds))]}")

print(f"\n--- Hyperparameter Tuning Results ---")
print(f"Gradient Boosting improvement: {r2_gb_tuned - r2_gb_default:+.4f}")
print(f"Ridge improvement: {r2_ridge_tuned - 0.8276:+.4f}")
print(f"Lasso improvement: {r2_lasso_tuned - 0.8277:+.4f}")
print(f"XGBoost improvement: {r2_xgb_tuned - 0.8958:+.4f}")

print(f"\n--- Best Tuned Model ---")
best_tuned_model = test_comparison.loc[test_comparison['Tuned R²'].idxmax()]
print(f"Model: {best_tuned_model['Model']}")
print(f"Tuned R²: {best_tuned_model['Tuned R²']:.4f}")

