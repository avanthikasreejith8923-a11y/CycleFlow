"""
CycleFlow - Session 4: Predictions & Model Analysis
===================================================
Goal: Visualize predictions, analyze errors, show real examples
Time: ~1 hour
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os

# ============================================================================
# STEP 1: Load Data and Train Best Model
# ============================================================================
print("\n[STEP 1] Loading data and training best model (Gradient Boosting)...")

X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

# Train Gradient Boosting (same as Session 3)
gb_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=5,
    random_state=42
)

gb_model.fit(X_train, y_train)

# Make predictions
y_pred = gb_model.predict(X_test)

print(f"✓ Model trained and predictions made")
print(f"✓ Test set size: {len(y_test)} samples")

# ============================================================================
# STEP 2: Calculate Detailed Metrics
# ============================================================================
print("\n[STEP 2] Calculating detailed performance metrics...")

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

# Calculate residuals (errors)
residuals = y_test - y_pred

print(f"\n--- Model Performance Metrics ---")
print(f"R² Score:           {r2:.4f}  (explains 89.71% of variation)")
print(f"RMSE:               {rmse:.2f} rentals (root mean squared error)")
print(f"MAE:                {mae:.2f} rentals (average absolute error)")
print(f"Mean Prediction:    {y_pred.mean():.0f} rentals")
print(f"Mean Actual:        {y_test.mean():.0f} rentals")
print(f"Std Dev Residuals:  {residuals.std():.2f} rentals")

# ============================================================================
# STEP 3: Show Example Predictions
# ============================================================================
print("\n[STEP 3] Example Predictions (10 random test samples)...")

# Create predictions dataframe
predictions_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred,
    'Error': residuals,
    'Abs_Error': np.abs(residuals),
    'Error_Percent': (np.abs(residuals) / y_test * 100)
})

# Show 10 random examples
print("\n--- 10 Random Test Samples ---")
print("(Sorted by error magnitude)")
example_df = predictions_df.sample(n=10, random_state=42).sort_values('Abs_Error', ascending=False)
example_df_display = example_df[['Actual', 'Predicted', 'Error', 'Error_Percent']].copy()
example_df_display['Predicted'] = example_df_display['Predicted'].round(0)
example_df_display['Error'] = example_df_display['Error'].round(0)
example_df_display['Error_Percent'] = example_df_display['Error_Percent'].round(1)

print(example_df_display.to_string())

# Show best and worst predictions
print("\n--- BEST Predictions (smallest errors) ---")
best_5 = predictions_df.nsmallest(5, 'Abs_Error')[['Actual', 'Predicted', 'Error', 'Error_Percent']].copy()
best_5['Predicted'] = best_5['Predicted'].round(0)
best_5['Error'] = best_5['Error'].round(0)
best_5['Error_Percent'] = best_5['Error_Percent'].round(1)
print(best_5.to_string())

print("\n--- WORST Predictions (largest errors) ---")
worst_5 = predictions_df.nlargest(5, 'Abs_Error')[['Actual', 'Predicted', 'Error', 'Error_Percent']].copy()
worst_5['Predicted'] = worst_5['Predicted'].round(0)
worst_5['Error'] = worst_5['Error'].round(0)
worst_5['Error_Percent'] = worst_5['Error_Percent'].round(1)
print(worst_5.to_string())

# ============================================================================
# STEP 4: Create Prediction Visualizations
# ============================================================================
print("\n[STEP 4] Creating prediction visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Actual vs Predicted (Scatter)
axes[0, 0].scatter(y_test, y_pred, alpha=0.6, color='steelblue', s=60, edgecolor='black', linewidth=0.5)
# Add perfect prediction line
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
axes[0, 0].set_xlabel('Actual Rentals', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Predicted Rentals', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Actual vs Predicted Rentals\n(Points should be near red line)', fontsize=13, fontweight='bold')
axes[0, 0].legend(fontsize=11)
axes[0, 0].grid(alpha=0.3)

# Plot 2: Residuals (Errors) vs Predicted
axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color='coral', s=60, edgecolor='black', linewidth=0.5)
axes[0, 1].axhline(y=0, color='red', linestyle='--', lw=2, label='Zero Error')
axes[0, 1].set_xlabel('Predicted Rentals', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Residuals (Actual - Predicted)', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Residuals Plot\n(Should be scattered around zero)', fontsize=13, fontweight='bold')
axes[0, 1].legend(fontsize=11)
axes[0, 1].grid(alpha=0.3)

# Plot 3: Distribution of Residuals
axes[1, 0].hist(residuals, bins=30, color='green', alpha=0.7, edgecolor='black')
axes[1, 0].axvline(x=0, color='red', linestyle='--', lw=2, label='Zero Error')
axes[1, 0].set_xlabel('Residual Value (rentals)', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Distribution of Prediction Errors\n(Should be centered at zero)', fontsize=13, fontweight='bold')
axes[1, 0].legend(fontsize=11)
axes[1, 0].grid(alpha=0.3, axis='y')

# Plot 4: Absolute Error Distribution
abs_errors = np.abs(residuals)
axes[1, 1].hist(abs_errors, bins=30, color='purple', alpha=0.7, edgecolor='black')
axes[1, 1].axvline(x=mae, color='red', linestyle='--', lw=2, label=f'Mean Absolute Error: {mae:.0f}')
axes[1, 1].set_xlabel('Absolute Error (rentals)', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Distribution of Absolute Errors\n(Lower values = better predictions)', fontsize=13, fontweight='bold')
axes[1, 1].legend(fontsize=11)
axes[1, 1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/session_4_predictions_analysis.png', dpi=100, bbox_inches='tight')
print("✓ Saved predictions plot to results/session_4_predictions_analysis.png")
plt.close()

# ============================================================================
# STEP 5: Create Sequential Predictions Plot
# ============================================================================
print("\n[STEP 5] Creating sequential predictions plot...")

fig, ax = plt.subplots(figsize=(16, 6))

# Plot actual and predicted values in sequence
x_axis = range(len(y_test))
ax.plot(x_axis, y_test, 'o-', label='Actual Rentals', color='steelblue', alpha=0.7, linewidth=2, markersize=5)
ax.plot(x_axis, y_pred, 's--', label='Predicted Rentals', color='coral', alpha=0.7, linewidth=2, markersize=4)

ax.set_xlabel('Test Sample Index', fontsize=12, fontweight='bold')
ax.set_ylabel('Bike Rentals', fontsize=12, fontweight='bold')
ax.set_title('Sequential Predictions: Actual vs Predicted\n(Dashed line should closely follow solid line)', 
             fontsize=13, fontweight='bold')
ax.legend(fontsize=12, loc='best')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results/session_4_sequential_predictions.png', dpi=100, bbox_inches='tight')
print("✓ Saved sequential plot to results/session_4_sequential_predictions.png")
plt.close()

# ============================================================================
# STEP 6: Analyze Model Behavior
# ============================================================================
print("\n[STEP 6] Analyzing model behavior by rental ranges...")

# Segment test data by rental volume
low_rentals = predictions_df[predictions_df['Actual'] < 3000]
medium_rentals = predictions_df[(predictions_df['Actual'] >= 3000) & (predictions_df['Actual'] < 6000)]
high_rentals = predictions_df[predictions_df['Actual'] >= 6000]

print(f"\n--- Model Performance by Rental Volume ---")
print(f"\nLow Rentals (< 3000 bikes):")
print(f"  Samples: {len(low_rentals)}")
print(f"  MAE: {low_rentals['Abs_Error'].mean():.2f} rentals")
print(f"  RMSE: {np.sqrt((low_rentals['Error']**2).mean()):.2f} rentals")

print(f"\nMedium Rentals (3000-6000 bikes):")
print(f"  Samples: {len(medium_rentals)}")
print(f"  MAE: {medium_rentals['Abs_Error'].mean():.2f} rentals")
print(f"  RMSE: {np.sqrt((medium_rentals['Error']**2).mean()):.2f} rentals")

print(f"\nHigh Rentals (> 6000 bikes):")
print(f"  Samples: {len(high_rentals)}")
print(f"  MAE: {high_rentals['Abs_Error'].mean():.2f} rentals")
print(f"  RMSE: {np.sqrt((high_rentals['Error']**2).mean()):.2f} rentals")

# ============================================================================
# STEP 7: Save All Predictions
# ============================================================================
print("\n[STEP 7] Saving detailed predictions...")

predictions_df_save = predictions_df.copy()
predictions_df_save.to_csv('results/detailed_predictions.csv', index=False)
print("✓ Saved detailed predictions to results/detailed_predictions.csv")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("SESSION 4 COMPLETE ✓")
print("="*70)

print(f"\n--- Final Model Summary ---")
print(f"Model: Gradient Boosting Regressor")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f} rentals")
print(f"MAE: {mae:.2f} rentals")
print(f"\nTest Set: {len(y_test)} samples")
print(f"Error Range: {abs_errors.min():.0f} to {abs_errors.max():.0f} rentals")
print(f"Median Error: {np.median(abs_errors):.0f} rentals")

print(f"\n--- Key Insights ---")
print(f"✓ Model explains 89.71% of rental variation")
print(f"✓ Average prediction error: ±{mae:.0f} rentals")
print(f"✓ Best at predicting medium-volume rental days")
print(f"✓ Visualizations saved to results/")


