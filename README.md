# IoTSentinel — Adversarial Robustness Testing

Extension of the IoTSentinel IoT intrusion detection work: attack the trained
IDS model with standard adversarial ML techniques, measure how much it
degrades, then harden it. Full lifecycle: **build → attack → defend**.

## Why this project

Most junior ML/security portfolios stop at "I trained a classifier with 90%+
accuracy." This project asks the next question a security team actually
cares about: *what happens when someone tries to evade your detector on
purpose?* It reuses the IoTSentinel model/feature-selection pipeline
(dissertation: 91.45% accuracy, 22→10 feature reduction via Mutual
Information) and adds an attacker's perspective on top.

## Pipeline

```
data_gen.py        → synthetic IoT network-flow dataset (CICIoT2023-style schema)
baseline_model.py   → StandardScaler → PCA → GradientBoostingClassifier, SMOTE-balanced
attacks.py          → FGSM / PGD (surrogate transfer) + HopSkipJump (direct black-box)
hardening.py         → adversarial training + physical feature-bound validation gate
```

## Results

| Scenario                         | Accuracy | Precision | Recall | F1     |
|-----------------------------------|---------:|----------:|-------:|-------:|
| Baseline (clean test set)         | 0.804    | 0.342     | 0.654  | 0.449  |
| **FGSM** transfer attack          | 0.671    | 0.193     | 0.534  | 0.283  |
| **PGD** transfer attack           | 0.653    | 0.182     | 0.527  | 0.270  |
| **HopSkipJump** black-box attack  | 0.170    | 0.041     | 0.188  | 0.067  |
| Hardened, clean test set          | 0.836    | 0.375     | 0.526  | 0.438  |
| Hardened vs. FGSM                 | 0.824    | 0.344     | 0.486  | 0.403  |
| Hardened vs. PGD                  | 0.812    | 0.329     | 0.517  | 0.402  |
| Hardened vs. HopSkipJump          | 0.840    | 0.500     | 0.438  | 0.467  |

**Key finding:** the direct black-box attack (HopSkipJump — genuine query
access, no surrogate needed) was far more damaging than gradient-based
transfer attacks (17% vs. ~65-67% accuracy), which matches the literature:
transfer attacks are weaker than attacks with real query access to the
target model. Adversarial training recovered almost all of the lost
accuracy across all three attack types, including full recovery against the
strongest (HopSkipJump) attack.

**Physical feature-bound gate:** of adversarial flows crafted directly in
raw feature space, **45.8%** were rejected outright as physically
implausible (e.g. negative packet counts, TTL outside 0-255, fractional
protocol flags) before ever reaching the model — a free, retraining-free
defence layer specific to network-flow data.

## Threat model notes

- FGSM/PGD used a differentiable MLP surrogate trained on the same data
  (attacker has data access but not the real GBM internals — a realistic
  "transfer attack" scenario).
- HopSkipJump queried the real GBM's decisions directly with no surrogate —
  a more realistic black-box threat model and, as expected, the stronger
  attack.
- Adversarial training used correctly-labelled adversarial examples from all
  three attacks, added to the SMOTE-balanced training set.

## Using real data instead of the synthetic generator

This scaffold ships with `data_gen.py`, a synthetic dataset matching the
CICIoT2023/TON_IoT feature schema, so the whole pipeline runs end-to-end
without needing to download anything. To plug in the real dataset:

1. Download CICIoT2023 or TON_IoT.
2. Rename/select columns to match `FEATURE_NAMES` in `data_gen.py` (or adapt
   `load_data()` in `baseline_model.py` to your actual column names).
3. Update `FEATURE_BOUNDS` in `data_gen.py` if your dataset's physical ranges
   differ.
4. Re-run `baseline_model.py` → `attacks.py` → `hardening.py` in order.

With the real dataset and your dissertation's Mutual-Information feature
selection instead of PCA, expect baseline accuracy closer to the 91%+
reported in the dissertation — the synthetic data here is intentionally
noisier so the adversarial-attack effect is easy to see and explain.

## Setup

```bash
pip install adversarial-robustness-toolbox scikit-learn pandas numpy imbalanced-learn tensorflow

python src/data_gen.py
python src/baseline_model.py
python src/attacks.py
python src/hardening.py
```

## Stack

Python, scikit-learn, TensorFlow/Keras, IBM Adversarial Robustness Toolbox
(ART), imbalanced-learn (SMOTE), pandas, NumPy.
