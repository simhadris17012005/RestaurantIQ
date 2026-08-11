"""
cuisine_classification.py
Classifies a restaurant's primary cuisine category using its
non-cuisine features (location, cost, rating behaviour, services).
Limited to the top N most frequent cuisines to keep classes balanced.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

from preprocess import load_data, clean_data, encode_categoricals

TOP_N_CUISINES = 10

FEATURES = [
    'Country Code', 'City_enc', 'Longitude', 'Latitude',
    'Average Cost for two', 'Currency_enc', 'Has Table booking',
    'Has Online delivery', 'Price range', 'Votes', 'Aggregate rating'
]


def prepare(df):
    top_cuisines = df['Primary Cuisine'].value_counts().nlargest(TOP_N_CUISINES).index
    df = df[df['Primary Cuisine'].isin(top_cuisines)].reset_index(drop=True)

    df, encoders = encode_categoricals(df, ['City', 'Currency', 'Primary Cuisine'])
    X = df[FEATURES]
    y = df['Primary Cuisine_enc']
    return X, y, encoders, df


def train_and_evaluate():
    df = clean_data(load_data())
    X, y, encoders, df = prepare(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=3000),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
    }

    label_names = encoders['Primary Cuisine'].classes_
    best_model, best_acc, best_name, best_preds = None, -1, None, None

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average='weighted', zero_division=0)
        rec = recall_score(y_test, preds, average='weighted', zero_division=0)
        print(f"{name}: Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}")
        if acc > best_acc:
            best_model, best_acc, best_name, best_preds = model, acc, name, preds

    print(f"\nBest model: {best_name} (Accuracy={best_acc:.4f})")
    print("\nPer-cuisine performance:")
    present_labels = sorted(set(y_test) | set(best_preds))
    print(classification_report(
        y_test, best_preds,
        labels=present_labels,
        target_names=[label_names[i] for i in present_labels],
        zero_division=0
    ))
    return best_model


if __name__ == "__main__":
    train_and_evaluate()
