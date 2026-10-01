import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def load_data(path: str) -> pd.DataFrame:
    """Load biomedical dataset from a CSV file."""
    return pd.read_csv(path)


def train_model(df: pd.DataFrame, target: str):
    """Preprocess data and train a classification model."""

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    return pipeline


if __name__ == "__main__":
    data = load_data("data/biomedical_data.csv")
    model = train_model(data, target="diagnosis")
