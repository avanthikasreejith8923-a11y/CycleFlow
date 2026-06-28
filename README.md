# 🚴 CycleFlow — Bike-Sharing Demand Prediction

> An end-to-end machine learning project that predicts daily bike rental demand using weather and temporal data — covering EDA, 7-model comparison, hyperparameter tuning, feature engineering, and a live Streamlit web app.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cycleflow-5jatato5dv3svst7y2vhgo.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange.svg)](https://scikit-learn.org)

---

## 🌐 Live Demo

👉 **[Try the app here → cycleflow-5jatato5dv3svst7y2vhgo.streamlit.app](https://cycleflow-5jatato5dv3svst7y2vhgo.streamlit.app/)**

Enter weather conditions and day type — get an instant prediction with confidence intervals and feature importance breakdown.

---

## 📖 Description

CycleFlow is a full machine learning pipeline built on the UCI Bike Sharing Dataset. The project simulates a real-world scenario: a bike-sharing company wants to forecast how many bikes will be rented on a given day based on weather conditions and the day's characteristics — so they can staff stations and redistribute bikes efficiently.

The project covers every stage of a production ML workflow:

- Exploratory data analysis to understand patterns (temperature, season, weather vs rentals)
- A leakage-safe preprocessing pipeline (scaler fit only on train data, never on test)
- Systematic comparison of 7 algorithms from linear baselines to ensemble methods
- Hyperparameter tuning using RandomizedSearchCV with 5-fold cross-validation
- Domain-driven feature engineering that created the single most important feature in the final model
- Detailed prediction analysis with residual plots, confidence intervals, and segment-wise error breakdown
- A deployed Streamlit web app where anyone can input weather conditions and get a live prediction

**Final model (Gradient Boosting + tuned + engineered features) achieves R² = 0.8995**, explaining nearly 90% of daily rental variation.

---

## 🏆 Results

| Model | R² Score | RMSE | Type |
|---|---|---|---|
| **Gradient Boosting (Tuned + Engineered)** | **0.8995** | **428** | ✅ Best |
| Gradient Boosting (Default) | 0.8971 | 434 | Ensemble |
| XGBoost | 0.8889 | 456 | Ensemble |
| Random Forest | 0.8835 | 468 | Ensemble |
| Linear Regression | 0.8277 | 831 | Linear |
| Lasso | 0.8277 | 831 | Linear |
| Ridge | 0.8276 | 831 | Linear |
| SVR | 0.8109 | 871 | Nonlinear |

Ensemble models outperformed linear baselines by **~8.4% in R²**. Feature engineering added a further **+0.92% on cross-validation**.

---

## 🧠 ML Techniques Used

- **Leakage-safe pipeline** — StandardScaler fit only on training data, transform applied to test set
- **7-algorithm comparison** — understanding when linear vs nonlinear models win and why
- **RandomizedSearchCV** — tuning 6 hyperparameters across 20 random combinations with 5-fold CV
- **5-fold cross-validation** — more reliable performance estimate than a single train/test split
- **Feature engineering** — `weather_temp` (weather × temperature interaction) became the #1 most important feature with importance score 0.51
- **Error analysis** — residuals, 95% confidence intervals, performance breakdown by rental volume segment

---

## 🔧 Tech Stack

`Python` `Scikit-learn` `XGBoost` `Streamlit` `Pandas` `NumPy` `Matplotlib` `Seaborn`

---

## 📁 Project Structure

```
cycleflow/
├── data/
│   ├── raw/                               # UCI Bike Sharing Dataset
│   └── processed/                         # Scaled train/test CSVs (original + engineered)
├── notebooks/
│   ├── session_1_eda.py                   # Data loading, EDA, preprocessing
│   ├── session_2_linear_models.py         # Linear, Ridge, Lasso, SVR
│   ├── session_3_ensemble_models.py       # Random Forest, Gradient Boosting, XGBoost
│   ├── session_4_predictions.py           # Prediction analysis & visualizations
│   ├── session_5_hyperparameter_tuning.py # RandomizedSearchCV + 5-fold CV
│   └── session_6_feature_engineering.py  # Domain-specific interaction features
├── models/                                # Saved model files
├── results/                               # Charts, CSVs, best_hyperparameters.json
├── app.py                                 # Streamlit web app
└── requirements.txt
```

---

## 🚀 Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/avanthikasreejith8923-a11y/CycleFlow.git
cd CycleFlow
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the full pipeline** (generates all data and model files)
```bash
python notebooks/session_1_eda.py
python notebooks/session_2_linear_models.py
python notebooks/session_3_ensemble_models.py
python notebooks/session_4_predictions.py
python notebooks/session_5_hyperparameter_tuning.py
python notebooks/session_6_feature_engineering.py
```

**4. Launch the Streamlit app**
```bash
streamlit run app.py
```

---

## 📊 Dataset

[UCI Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset) — 731 daily records from a Washington D.C. bike-sharing system (2011–2012).

**Features used:** season, year, month, holiday, weekday, working day, weather condition, temperature, feel-like temperature, humidity, windspeed

**Target:** `cnt` — total daily bike rentals (range: 22 to 8,714)

**Engineered features created:**
| Feature | Formula | Logic |
|---|---|---|
| `weather_temp` | weather_sit × temp | Good weather amplifies temperature effect |
| `temp_humidity_comfort` | temp × (1 − humidity) | High temp + low humidity = comfortable |
| `season_temp` | season × temp | Temperature effect varies by season |
| `is_weekend` | 1 if weekday ∈ {0, 6} | Weekend vs weekday rental patterns differ |

---

## 📈 Visual Outputs

All charts saved to `results/` after running the pipeline:

- `session_1_eda.png` — rental distribution, seasonal boxplots, temperature correlation, heatmap
- `session_2_model_comparison.png` — R² and RMSE for 4 linear models
- `session_3_all_models_comparison.png` — all 7 models side by side
- `session_3_feature_importance.png` — feature importance for all 3 ensemble models
- `session_4_predictions_analysis.png` — actual vs predicted, residuals, error distribution
- `session_5_hyperparameter_tuning.png` — CV comparison and default vs tuned performance
- `session_6_feature_engineering.png` — impact of engineered features on model performance

---

## 👤 Author

**Avanthika Sreejith**
B.Tech CSE | AI Enthusiast

[GitHub](https://github.com/avanthikasreejith8923-a11y) · [LinkedIn](https://linkedin.com/in/YOUR_PROFILE)
