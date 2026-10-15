"""Exploratory offline reinforcement-learning experiment.

Tabular Double Q-learning on real 2012 SHARP features, with the asymmetric
reward from the project paper (TP +5, TN +1, FP -1, FN -8 by default).

State   : the 4 most important XGBoost features (distinct physical
          parameters), each quantile-binned into 5 bins from training data.
Actions : 0 = no alert, 1 = alert.
Stream  : observations in time order within each active region; the next
          state is the region's next observation.
Split   : same chronological split as XGBoost (train <May, validation
          May-Jun, test Jul-Dec 2012).

Honest framing: the reward does not depend on how an action changes later
states, so the learned policy is equivalent to a cost-sensitive threshold on
binned flare rates. This is a research checkpoint, not a production agent.
"""

import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FEATURES_PATH = (
    PROJECT_ROOT / "data" / "processed" / "features"
    / "sharp_goes_temporal_features_2012_full_25params.parquet"
)
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_2012_temporal_features_25params.joblib"
OUTPUT_DIR = PROJECT_ROOT / "results" / "rl_dry_run"

VALIDATION_START = pd.Timestamp("2012-05-01 00:00:00")
TEST_START = pd.Timestamp("2012-07-01 00:00:00")

TARGET = "target_24h"
N_STATE_FEATURES = 4
N_BINS = 5
EPOCHS = 8
ALPHA = 0.05
GAMMA = 0.9
EPS_START = 1.0
EPS_END = 0.05
SEED = 42

# weights are (TP, TN, FP, FN) rewards
REWARD_VARIANTS = [
    {"name": "RL, FN penalty -8 (paper default)", "weights": (5.0, 1.0, -1.0, -8.0)},
    {"name": "RL, FN penalty -16", "weights": (5.0, 1.0, -1.0, -16.0)},
    {"name": "RL, FN penalty -32", "weights": (5.0, 1.0, -1.0, -32.0)},
    {"name": "RL, FN penalty -64", "weights": (5.0, 1.0, -1.0, -64.0)},
]


# ============================================================
# STATE ENCODING
# ============================================================

def base_param(name):
    """Physical parameter behind a feature name (strips LOG_ and temporal suffixes)."""

    if name.startswith("LOG_"):
        name = name[4:]

    for marker in ("_RELCHANGE_", "_CHANGE_", "_ROLLSTD_"):
        if marker in name:
            return name.split(marker)[0]

    return name


def choose_state_features(importances, columns, n):

    chosen, used = [], set()

    for feature in importances.sort_values(ascending=False).index:

        if feature in ("observation_hour", "day_of_year") or feature not in columns:
            continue

        base = base_param(feature)

        if base in used:
            continue

        chosen.append(feature)
        used.add(base)

        if len(chosen) == n:
            break

    return chosen


def fit_encoder(train_df, features, n_bins):

    medians, edges = {}, []

    for f in features:
        values = train_df[f].astype(float)
        medians[f] = float(values.median())
        filled = values.fillna(medians[f]).to_numpy()
        quantiles = np.quantile(filled, np.linspace(0, 1, n_bins + 1)[1:-1])
        edges.append(np.unique(quantiles))

    return {"features": list(features), "medians": medians, "edges": edges}


def n_states_of(encoder):
    return int(np.prod([len(e) + 1 for e in encoder["edges"]]))


def encode_states(df, encoder):

    codes = np.zeros(len(df), dtype=np.int64)

    for f, edges in zip(encoder["features"], encoder["edges"]):
        values = df[f].astype(float).fillna(encoder["medians"][f]).to_numpy()
        bins = np.searchsorted(edges, values, side="right")
        codes = codes * (len(edges) + 1) + bins

    return codes


def decode_state(code, encoder):

    sizes = [len(e) + 1 for e in encoder["edges"]]
    parts = []

    for size in reversed(sizes):
        parts.append(code % size)
        code //= size

    return list(reversed(parts))


def region_bounds(df):

    ar = df["NOAA_AR"].to_numpy()
    change = np.flatnonzero(ar[1:] != ar[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(ar)]))

    return list(zip(starts.tolist(), ends.tolist()))


# ============================================================
# DOUBLE Q-LEARNING
# ============================================================

def train_double_q(states, labels, bounds, n_states, weights,
                   epochs=EPOCHS, alpha=ALPHA, gamma=GAMMA,
                   eps_start=EPS_START, eps_end=EPS_END, seed=SEED):

    tp_r, tn_r, fp_r, fn_r = weights

    rng = random.Random(seed)

    qa = [[0.0, 0.0] for _ in range(n_states)]
    qb = [[0.0, 0.0] for _ in range(n_states)]

    states = [int(s) for s in states]
    labels = [int(y) for y in labels]
    bounds = list(bounds)

    total = max(1, epochs * len(states))
    decay_steps = 0.6 * total
    step = 0

    for _ in range(epochs):

        rng.shuffle(bounds)

        for start, end in bounds:

            last = end - 1

            for i in range(start, end):

                s = states[i]
                y = labels[i]

                eps = max(eps_end, eps_start - (eps_start - eps_end) * step / decay_steps)

                if rng.random() < eps:
                    a = rng.randint(0, 1)
                else:
                    a = 1 if (qa[s][1] + qb[s][1]) > (qa[s][0] + qb[s][0]) else 0

                if a == 1:
                    r = tp_r if y == 1 else fp_r
                else:
                    r = fn_r if y == 1 else tn_r

                if rng.random() < 0.5:
                    online, other = qa, qb
                else:
                    online, other = qb, qa

                if i == last:
                    target = r
                else:
                    s2 = states[i + 1]
                    best = 1 if online[s2][1] > online[s2][0] else 0
                    target = r + gamma * other[s2][best]

                online[s][a] += alpha * (target - online[s][a])

                step += 1

    return np.array(qa), np.array(qb)


def q_margin(qa, qb):
    return (qa[:, 1] + qb[:, 1]) - (qa[:, 0] + qb[:, 0])


# ============================================================
# METRICS
# ============================================================

def skill(y_true, y_pred):

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    far = fp / (fp + tn) if (fp + tn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0

    return {
        "tss": recall - far, "recall": recall, "far": far,
        "precision": precision, "alerts": tp + fp,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def make_row(policy, role, y_val, pred_val, y_test, pred_test, roc):

    v = skill(y_val, pred_val)
    t = skill(y_test, pred_test)

    return {
        "policy": policy, "role": role,
        "val_tss": v["tss"], "val_recall": v["recall"], "val_far": v["far"],
        "val_precision": v["precision"], "val_alerts": v["alerts"],
        "test_tss": t["tss"], "test_recall": t["recall"], "test_far": t["far"],
        "test_precision": t["precision"], "test_alerts": t["alerts"],
        "test_tp": t["tp"], "test_fp": t["fp"], "test_fn": t["fn"], "test_tn": t["tn"],
        "test_roc_auc": roc,
    }


def safe_auc(y, score):
    return float(roc_auc_score(y, score)) if len(np.unique(y)) > 1 else float("nan")


# ============================================================
# MAIN
# ============================================================

def main():

    from scripts.predict import SolarFlarePredictor

    print("=" * 60)
    print(" RL DRY RUN: tabular Double Q-learning on 2012 SHARP features")
    print("=" * 60)

    data = pd.read_parquet(FEATURES_PATH)
    data["observation_time"] = pd.to_datetime(data["observation_time"])
    data = data.dropna(subset=["NOAA_AR", TARGET])

    def split(mask):
        return data[mask].sort_values(["NOAA_AR", "observation_time"]).reset_index(drop=True)

    train = split(data["observation_time"] < VALIDATION_START)
    val = split((data["observation_time"] >= VALIDATION_START) & (data["observation_time"] < TEST_START))
    test = split(data["observation_time"] >= TEST_START)

    print(f"Train {len(train):,} | Validation {len(val):,} | Test {len(test):,}")

    y_train = train[TARGET].astype(int).to_numpy()
    y_val = val[TARGET].astype(int).to_numpy()
    y_test = test[TARGET].astype(int).to_numpy()

    predictor = SolarFlarePredictor(str(MODEL_PATH))
    importances = pd.Series(predictor.model.feature_importances_, index=predictor.features)

    features = choose_state_features(importances, set(data.columns), N_STATE_FEATURES)
    print("State features:", features)

    encoder = fit_encoder(train, features, N_BINS)
    n_states = n_states_of(encoder)
    print(f"Discrete states: {n_states}")

    train_states = encode_states(train, encoder)
    val_states = encode_states(val, encoder)
    test_states = encode_states(test, encoder)

    bounds = region_bounds(train)

    counts = np.bincount(train_states, minlength=n_states)
    flares = np.bincount(train_states, weights=y_train, minlength=n_states)
    flare_rate = flares / np.maximum(counts, 1)

    rows = []

    # ---------------- XGBoost baselines ----------------
    val_prob = predictor.predict(val)["flare_probability"].to_numpy()
    test_prob = predictor.predict(test)["flare_probability"].to_numpy()
    xgb_roc = safe_auc(y_test, test_prob)

    saved = float(predictor.threshold)

    rows.append(make_row(
        f"XGBoost (saved threshold {saved:.3f})", "baseline",
        y_val, (val_prob >= saved).astype(int),
        y_test, (test_prob >= saved).astype(int), xgb_roc,
    ))

    candidates = np.unique(np.concatenate(([saved], np.quantile(val_prob, np.linspace(0.5, 0.999, 250)))))
    best_thr, best_tss = saved, -1.0

    for thr in candidates:
        tss = skill(y_val, (val_prob >= thr).astype(int))["tss"]
        if tss > best_tss:
            best_tss, best_thr = tss, float(thr)

    rows.append(make_row(
        f"XGBoost (validation-TSS threshold {best_thr:.4f})", "baseline",
        y_val, (val_prob >= best_thr).astype(int),
        y_test, (test_prob >= best_thr).astype(int), xgb_roc,
    ))

    # ---------------- RL variants ----------------
    margins = {}
    rl_rows = []

    for variant in REWARD_VARIANTS:

        print(f"\nTraining: {variant['name']} ...")

        qa, qb = train_double_q(train_states, y_train, bounds, n_states, variant["weights"])
        margin = q_margin(qa, qb)
        margins[variant["name"]] = margin

        rl_rows.append(make_row(
            variant["name"], "rl",
            y_val, (margin[val_states] > 0).astype(int),
            y_test, (margin[test_states] > 0).astype(int),
            safe_auc(y_test, margin[test_states]),
        ))

        print(f"  validation TSS {rl_rows[-1]['val_tss']:.4f} | test TSS {rl_rows[-1]['test_tss']:.4f}")

    selected_index = int(np.argmax([r["val_tss"] for r in rl_rows]))
    selected_name = rl_rows[selected_index]["policy"]
    rl_rows[selected_index]["policy"] = selected_name + " [selected on validation]"

    rows.extend(rl_rows)

    results = pd.DataFrame(rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_DIR / "rl_results.csv", index=False)

    # ---------------- policy table for the selected variant ----------------
    margin = margins[selected_name]

    policy = pd.DataFrame({
        "state": np.arange(n_states),
        "bins": ["-".join(str(b) for b in decode_state(c, encoder)) for c in range(n_states)],
        "n_train": counts,
        "train_flare_rate": flare_rate,
        "margin": margin,
        "alert": (margin > 0).astype(int),
    })

    policy.to_csv(OUTPUT_DIR / "rl_policy.csv", index=False)

    config = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "state_features": features,
        "bins_per_feature": [len(e) + 1 for e in encoder["edges"]],
        "bin_edges": [e.tolist() for e in encoder["edges"]],
        "n_states": n_states,
        "actions": ["no alert", "alert"],
        "reward_variants": [{"name": v["name"], "tp_tn_fp_fn": list(v["weights"])} for v in REWARD_VARIANTS],
        "selected_variant": selected_name,
        "epochs": EPOCHS, "alpha": ALPHA, "gamma": GAMMA,
        "epsilon_start": EPS_START, "epsilon_end": EPS_END, "seed": SEED,
        "train_rows": int(len(train)), "validation_rows": int(len(val)), "test_rows": int(len(test)),
        "note": (
            "The reward does not depend on how an action changes later states, so the "
            "learned policy is equivalent to a cost-sensitive threshold on binned flare "
            "rates. This is a research checkpoint, not a production agent."
        ),
    }

    with open(OUTPUT_DIR / "rl_config.json", "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)

    show = results[["policy", "val_tss", "test_tss", "test_recall", "test_far", "test_precision", "test_alerts", "test_roc_auc"]]

    print("\n" + "=" * 60)
    print(" RESULTS (held-out test period, Jul-Dec 2012)")
    print("=" * 60)
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()