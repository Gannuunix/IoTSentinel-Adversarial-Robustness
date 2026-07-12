"""
Synthetic IoT network-flow dataset generator.

Mirrors the feature schema used in CICIoT2023 / TON_IoT (flow duration, packet
counts, byte counts, inter-arrival times, protocol flags) so the rest of the
pipeline is a drop-in replacement once the real CSV is available.

To use the real dataset instead: load your CICIoT2023 / TON_IoT CSV, rename
columns to match FEATURE_NAMES below (or just skip this generator and adapt
the loader in baseline_model.py to read your file directly).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

BASE_DIR = Path(__file__).resolve().parent / "iotsentinel-adversarial"
DATA_DIR = BASE_DIR / "data"

FEATURE_NAMES = [
    "flow_duration", "fwd_pkt_count", "bwd_pkt_count", "fwd_byte_count",
    "bwd_byte_count", "pkt_size_avg", "pkt_size_std", "iat_mean", "iat_std",
    "syn_flag_count", "ack_flag_count", "psh_flag_count", "rst_flag_count",
    "fwd_pkt_rate", "bwd_pkt_rate", "active_mean", "idle_mean",
    "protocol_tcp", "protocol_udp", "protocol_mqtt", "dst_port_entropy",
    "ttl_mean",
]

# Physical bounds per feature — used later by hardening.py to reject
# physically implausible adversarial perturbations (an IoT-specific defence).
FEATURE_BOUNDS = {
    "flow_duration": (0, None),
    "fwd_pkt_count": (0, None),
    "bwd_pkt_count": (0, None),
    "fwd_byte_count": (0, None),
    "bwd_byte_count": (0, None),
    "pkt_size_avg": (0, 65535),
    "pkt_size_std": (0, None),
    "iat_mean": (0, None),
    "iat_std": (0, None),
    "syn_flag_count": (0, None),
    "ack_flag_count": (0, None),
    "psh_flag_count": (0, None),
    "rst_flag_count": (0, None),
    "fwd_pkt_rate": (0, None),
    "bwd_pkt_rate": (0, None),
    "active_mean": (0, None),
    "idle_mean": (0, None),
    "protocol_tcp": (0, 1),
    "protocol_udp": (0, 1),
    "protocol_mqtt": (0, 1),
    "dst_port_entropy": (0, 1),
    "ttl_mean": (0, 255),
}


def generate_iot_dataset(n_samples: int = 20000, imbalance_ratio: float = 0.12,
                          random_state: int = 42) -> pd.DataFrame:
    """Generate a synthetic binary IoT intrusion dataset (benign=0, attack=1)
    with class imbalance matching typical CICIoT2023 proportions."""
    n_features = len(FEATURE_NAMES)
    weights = [1 - imbalance_ratio, imbalance_ratio]

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=18,
        n_redundant=3,
        n_repeated=0,
        n_classes=2,
        weights=weights,
        class_sep=2.2,
        flip_y=0.005,
        random_state=random_state,
    )

    df = pd.DataFrame(X, columns=FEATURE_NAMES)

    # Push generated values into physically plausible IoT ranges.
    df["flow_duration"] = np.abs(df["flow_duration"]) * 500
    df["fwd_pkt_count"] = np.abs(df["fwd_pkt_count"]) * 20
    df["bwd_pkt_count"] = np.abs(df["bwd_pkt_count"]) * 20
    df["fwd_byte_count"] = np.abs(df["fwd_byte_count"]) * 1500
    df["bwd_byte_count"] = np.abs(df["bwd_byte_count"]) * 1500
    df["pkt_size_avg"] = np.clip(np.abs(df["pkt_size_avg"]) * 300, 0, 65535)
    df["pkt_size_std"] = np.abs(df["pkt_size_std"]) * 50
    df["iat_mean"] = np.abs(df["iat_mean"]) * 100
    df["iat_std"] = np.abs(df["iat_std"]) * 20
    for flag in ["syn_flag_count", "ack_flag_count", "psh_flag_count", "rst_flag_count"]:
        df[flag] = np.abs(df[flag]).round().astype(int)
    df["fwd_pkt_rate"] = np.abs(df["fwd_pkt_rate"]) * 10
    df["bwd_pkt_rate"] = np.abs(df["bwd_pkt_rate"]) * 10
    df["active_mean"] = np.abs(df["active_mean"]) * 200
    df["idle_mean"] = np.abs(df["idle_mean"]) * 200
    for proto in ["protocol_tcp", "protocol_udp", "protocol_mqtt"]:
        df[proto] = (df[proto] > df[proto].median()).astype(int)
    df["dst_port_entropy"] = np.clip(np.abs(df["dst_port_entropy"]) / df["dst_port_entropy"].abs().max(), 0, 1)
    df["ttl_mean"] = np.clip(np.abs(df["ttl_mean"]) * 40 + 32, 0, 255)

    df["label"] = y
    return df


if __name__ == "__main__":
    df = generate_iot_dataset()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "iot_flows.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows -> {out_path}")
    print(df["label"].value_counts(normalize=True))
