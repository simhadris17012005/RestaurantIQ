"""
rating_prediction.py
Predicts a restaurant's Aggregate Rating using regression models.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import os
_OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(_OUT_DIR, exist_ok=True)
from preprocess import load_data, clean_data, encode_categoricals

FEATURES = [
    'Country Code', 'City_enc', 'Longitude', 'Latitude', 'Primary Cuisine_enc',
    'Average Cost for two', 'Currency_enc', 'Has Table booking', 'Has Online delivery',
    'Price range', 'Votes'
]
TARGET = 'Aggregate rating'


def prepare(df):
    df, encoders = encode_categoricals(df, ['City', 'Primary Cuisine', 'Currency'])
    X = df[FEATURES]
    y = df[TARGET]
    return X, y, encoders


def train_and_evaluate():
    df = clean_data(load_data())
    X, y, encoders = prepare(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
    }

    results = []
    best_model, best_r2, best_name = None, -np.inf, None

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        results.append({"Model": name, "MSE": round(mse, 4), "MAE": round(mae, 4), "R2": round(r2, 4)})
        if r2 > best_r2:
            best_model, best_r2, best_name = model, r2, name

    results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
    print(results_df.to_string(index=False))

    # Feature importance from the best tree-based model (fallback to RF for interpretability)
    importances_model = best_model if hasattr(best_model, "feature_importances_") else models["Random Forest"]
    if hasattr(importances_model, "feature_importances_"):
        fi = pd.Series(importances_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
        print(f"\nTop features influencing rating ({importances_model.__class__.__name__}):")
        print(fi.to_string())

    joblib.dump({"model": best_model, "encoders": encoders, "features": FEATURES},
                os.path.join(_OUT_DIR, "rating_model.pkl"))
    print(f"\nBest model: {best_name} (R2={best_r2:.4f}) saved to outputs/rating_model.pkl")
    return results_df


if __name__ == "__main__":
    train_and_evaluate()
