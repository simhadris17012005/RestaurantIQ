"""
preprocess.py
Shared data loading & cleaning utilities for RestaurantIQ.
"""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Resolve data path relative to the project root regardless of cwd
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_PATH = os.path.join(_PROJECT_ROOT, "data", "restaurants.csv")


def load_data(path=None):
    df = pd.read_csv(path or DEFAULT_DATA_PATH)
    return df


def clean_data(df):
    df = df.copy()

    # Handle missing values
    df['Cuisines'] = df['Cuisines'].fillna('Not Specified')

    # Drop rows with 0 aggregate rating (no votes / not rated) for supervised tasks
    df = df[df['Votes'] > 0].reset_index(drop=True)

    # Binary Yes/No -> 1/0
    for col in ['Has Table booking', 'Has Online delivery', 'Is delivering now', 'Switch to order menu']:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

    # Primary cuisine (first listed) - useful for classification target
    df['Primary Cuisine'] = df['Cuisines'].apply(lambda x: x.split(',')[0].strip())

    return df


def encode_categoricals(df, columns):
    """Label-encode given categorical columns, returns df + fitted encoders dict."""
    df = df.copy()
    encoders = {}
    for col in columns:
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    print("Shape after cleaning:", df.shape)
    print(df.head())
