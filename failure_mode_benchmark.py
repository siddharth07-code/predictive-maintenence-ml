from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF"]
NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
DATASET_PATH = Path(__file__).resolve().parent / "data" / "ai4i2020.csv"


def main():
    data = pd.read_csv(DATASET_PATH)
    counts = data[FAILURE_MODES].eq(1).sum()
    target = counts.idxmax()
    X = data[NUMERIC_FEATURES + ["Type"]]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessing = ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ("type", OneHotEncoder(handle_unknown="ignore"), ["Type"]),
    ])
    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        )),
    ])
    model.fit(X_train, y_train)
    scores = model.predict_proba(X_test)[:, 1]
    average_precision = average_precision_score(y_test, scores)
    prevalence = float(y_test.mean())

    print(f"Target: {target} ({int(counts[target])} positives overall)")
    print("Split: stratified 80/20, random_state=42")
    print(f"Train positives: {int(y_train.sum())}/{len(y_train)}")
    print(f"Test positives: {int(y_test.sum())}/{len(y_test)}")
    print(f"Test average precision: {average_precision:.6f}")
    print(f"Test prevalence (no-skill AP reference): {prevalence:.6f} ({prevalence:.2%})")


if __name__ == "__main__":
    main()
