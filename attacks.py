"""
Adversarial attacks against the IoT IDS baseline model.

GBM/tree ensembles aren't differentiable, so two attack strategies are used:

1. Surrogate transfer attack (FGSM, PGD): train a differentiable MLP
   surrogate on the same data, craft adversarial examples against it using
   gradient-based methods, then test whether those examples also fool the
   real GBM model (transferability is a well-known property of adversarial
   examples across model families).

2. Direct black-box attack (HopSkipJump): query the GBM model's decisions
   directly, no gradient or surrogate needed. Slower, but a more realistic
   threat model since an attacker rarely has your training data or model
   internals.
"""
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from art.attacks.evasion import FastGradientMethod, HopSkipJump, ProjectedGradientDescent
from art.estimators.classification import KerasClassifier, SklearnClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

MODEL_DIR = "/home/claude/iotsentinel-adversarial/models"
RESULTS_DIR = "/home/claude/iotsentinel-adversarial/results"

tf.compat.v1.disable_eager_execution() if False else None  # kept off; ART TF2 classifier used instead


def load_test_set():
    X_test = pd.read_csv(f"{MODEL_DIR}/X_test.csv")
    y_test = pd.read_csv(f"{MODEL_DIR}/y_test.csv").squeeze()
    return X_test, y_test


def build_surrogate(input_dim: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(2, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def report(y_true, y_pred, label: str):
    print(f"\n--- {label} ---")
    print(f"accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"recall   : {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"f1       : {f1_score(y_true, y_pred, zero_division=0):.4f}")


def run_surrogate_transfer_attack(gbm_pipeline, X_test, y_test):
    # Preprocess with the same scaler+PCA the GBM pipeline uses, so the
    # surrogate operates in the same feature space the attacker would target.
    X_proc = gbm_pipeline[:-1].transform(X_test)  # scaler + PCA, no classifier
    y_arr = y_test.values

    surrogate = build_surrogate(input_dim=X_proc.shape[1])
    surrogate.fit(X_proc, y_arr, epochs=15, batch_size=64, verbose=0)

    art_classifier = KerasClassifier(model=surrogate, clip_values=(X_proc.min(), X_proc.max()))

    results = {}
    for name, attack_cls, kwargs in [
        ("FGSM", FastGradientMethod, dict(eps=0.3)),
        ("PGD", ProjectedGradientDescent, dict(eps=0.3, eps_step=0.05, max_iter=20)),
    ]:
        attack = attack_cls(estimator=art_classifier, **kwargs)
        X_adv = attack.generate(x=X_proc)

        # Surrogate performance under attack (sanity check the attack worked)
        surrogate_preds = np.argmax(surrogate.predict(X_adv, verbose=0), axis=1)
        report(y_arr, surrogate_preds, f"{name} — surrogate NN under attack")

        # Transferability: feed the same adversarial examples to the real GBM
        gbm_preds = gbm_pipeline.named_steps["clf"].predict(X_adv)
        report(y_arr, gbm_preds, f"{name} — transferred to GBM (baseline model)")

        results[name] = dict(X_adv=X_adv, gbm_preds=gbm_preds)

    return results


def run_hopskipjump_attack(gbm_pipeline, X_test, y_test, n_samples: int = 100):
    """True black-box attack: queries the GBM's predictions directly, no
    surrogate or gradients needed. Slower (many queries per sample), so run
    on a subset."""
    X_proc = gbm_pipeline[:-1].transform(X_test)[:n_samples]
    y_arr = y_test.values[:n_samples]

    clf = gbm_pipeline.named_steps["clf"]
    art_classifier = SklearnClassifier(model=clf, clip_values=(X_proc.min(), X_proc.max()))

    attack = HopSkipJump(classifier=art_classifier, max_iter=20, max_eval=200, init_eval=20)
    X_adv = attack.generate(x=X_proc)

    preds = clf.predict(X_adv)
    report(y_arr, preds, "HopSkipJump — direct black-box attack on GBM")
    return dict(X_adv=X_adv, preds=preds)


def main():
    gbm_pipeline = joblib.load(f"{MODEL_DIR}/baseline_gbm.joblib")
    X_test, y_test = load_test_set()

    baseline_preds = gbm_pipeline.predict(X_test)
    report(y_test, baseline_preds, "Baseline GBM — clean test set (no attack)")

    transfer_results = run_surrogate_transfer_attack(gbm_pipeline, X_test, y_test)
    hsj_results = run_hopskipjump_attack(gbm_pipeline, X_test, y_test, n_samples=100)

    # Persist adversarial examples for use in hardening.py
    np.save(f"{RESULTS_DIR}/X_adv_fgsm.npy", transfer_results["FGSM"]["X_adv"])
    np.save(f"{RESULTS_DIR}/X_adv_pgd.npy", transfer_results["PGD"]["X_adv"])
    np.save(f"{RESULTS_DIR}/X_adv_hsj.npy", hsj_results["X_adv"])
    print(f"\nSaved adversarial examples -> {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
