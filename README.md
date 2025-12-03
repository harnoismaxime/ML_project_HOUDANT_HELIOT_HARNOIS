# F1 Lap Time Prediction — Qatar Grand Prix

This project builds a machine-learning pipeline to predict Formula 1 lap times on the Qatar Grand Prix from race, tyre and weather data.

The work is done in a single Jupyter notebook and follows a full ML workflow: data exploration, cleaning, feature engineering, dimensionality reduction, model comparison and evaluation.

---

## Project overview

- **Goal**: predict `LapTime_sec` (lap time in seconds) for each lap.
- **Type of task**: supervised regression on tabular data.
- **Main features**: tyre life and compound, stint, lap number, race position, weather (air/track temperature, pressure, humidity, wind), driver and team.

The project is mainly educational: it shows how to go from raw race data to a reasonably accurate predictive model.

---

## Methods

The notebook implements:

- Data cleaning (missing values, duplicates, outliers, target leakage control).
- Feature preprocessing with `ColumnTransformer` (numeric scaling + categorical one-hot encoding).
- Dimensionality reduction with **PCA** and feature selection with **SelectKBest**.
- Baseline and models:  
  `DummyRegressor`, Ridge Regression, Decision Tree, Random Forest.
- Advanced models and ensembles:  
  XGBoost, Bagging, VotingRegressor, StackingRegressor.
- Model evaluation with train/test split, cross-validation, learning curves and
  metrics: **R²**, **MSE**, **MAE**.

Best models (XGBoost and voting/stacking ensembles) reach **R² > 0.70** on the test set with a typical error below **1 second** per lap.

---

## Repository structure

- `Qatar.csv` — lap-by-lap race dataset (not committed if restricted).
- `projetF1_new.ipynb` — main notebook with all the code, analysis and plots.
- `figures/` (optional) — exported plots for the report / README.

---

## Requirements

- Python 3.10+
- Jupyter
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn`
- `xgboost` (or `HistGradientBoostingRegressor` as a fallback)

You can install the dependencies with:

```bash
pip install -r requirements.txt
How to run

Clone the repository and place Qatar.csv at the root of the project.

Create a virtual environment (optional but recommended).

Install the dependencies.

Launch Jupyter:

jupyter notebook


Open projetF1_new.ipynb and run the cells from top to bottom.

The notebook will reproduce the full analysis: data exploration, model training, performance comparison and plots (feature importance, model performance, learning curves, etc.).
