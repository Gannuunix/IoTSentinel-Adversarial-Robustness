"""
Baseline IDS model — Gradient Boosting classifier with SMOTE oversampling
and PCA dimensionality reduction, mirroring the IoTSentinel/dissertation
pipeline (22 -> 10 features via Mutual Information in the original work;
here we use PCA for a comparable, reproducible reduction).

Run standalone to train + save the baseline model and print metrics.
"""
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score, precision_score,
                              recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = "/home/claude/iotsentinel-adversarial/data/iot_flows.csv"
MODEL_DIR = "/home/claude/iotsentinel-adversarial/models"
N_PCA_COMPONENTS = 16


def load_data(path: str = DATA_PATH):
    df = pd.read_csv(path)
    X = df.drop(columns=["label"])
    y = df["label"]
    return X, y


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=N_PCA_COMPONENTS, random_state=42)),
        ("clf", GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42
        )),
    ])


def evaluate(y_true, y_pred, label: str = "Baseline") -> dict:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }
    print(f"\n--- {label} ---")
    for k, v in metrics.items():
        print(f"{k:>10}: {v:.4f}")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=["benign", "attack"]))
    return metrics


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    # SMOTE on training data only — avoids leaking synthetic samples into test set
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"Train class balance after SMOTE: {np.bincount(y_train_res)}")

    pipeline = build_pipeline()
    pipeline.fit(X_train_res, y_train_res)

    y_pred = pipeline.predict(X_test)
    evaluate(y_test, y_pred, label="Baseline GBM (clean test set)")

    joblib.dump(pipeline, f"{MODEL_DIR}/baseline_gbm.joblib")
    X_test.to_csv(f"{MODEL_DIR}/X_test.csv", index=False)
    y_test.to_csv(f"{MODEL_DIR}/y_test.csv", index=False)
    print(f"\nSaved model -> {MODEL_DIR}/baseline_gbm.joblib")


if __name__ == "__main__":
    main()
