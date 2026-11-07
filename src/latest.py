"""
Hybrid RUL model for NASA C-MAPSS FD004: CNN-GRU (seed ensemble) + XGBoost, blended.

Pipeline
  1. Per-operating-condition normalisation (KMeans on the 3 operating settings),
     fitted on TRAIN engines only.
  2. Engine-wise train/validation split (no window leakage).
  3. CNN-GRU on raw 30-step windows  +  XGBoost on engineered window features.
  4. Blend weight tuned on validation engines.
  5. Benchmark evaluation: last window of each engine in test_FD004 vs RUL_FD004.txt.
  6. Artifacts saved for serving (see predict_rul at the bottom).

Run:  python train_hybrid_fd004.py
"""
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from sklearn.cluster import KMeans
from tensorflow.keras import layers, models

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
BASE = Path("E:/P2/Python/TurbofanEngine/git_FD004")
TRAIN_CSV = BASE / "data" / "train_FD004.csv"
TEST_CSV = BASE / "data" / "test_FD004.csv"
RUL_TXT = BASE / "data" / "RUL_FD004.txt"
OUT_DIR = BASE / "artifacts_hybrid"

ENGINE, CYCLE = "Engine", "Cycles"
OP_COLS = ["OP_Setting 1", "OP_Setting 2", "OP_Setting 3"]
# Sensors that carry (almost) no degradation signal; adjust if your column
# numbering differs. Set to [] to keep everything.
DROP_SENSOR_IDS = [1, 5, 6, 10, 16, 18, 19]

WINDOW = 30
RUL_CAP = 125
N_CONDITIONS = 6
VAL_FRACTION = 0.2
N_NN_SEEDS = 3          # size of the CNN-GRU ensemble
EPOCHS = 100
BATCH_SIZE = 128
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def get_sensor_cols(df):
    drop = {f"Sensor_{i}" for i in DROP_SENSOR_IDS}
    return [c for c in df.columns if c.startswith("Sensor_") and c not in drop]


def add_rul(df, cap=RUL_CAP):
    """Run-to-failure data only (train_FD004). NOT for the benchmark test file."""
    df = df.sort_values([ENGINE, CYCLE]).copy()
    max_c = df.groupby(ENGINE)[CYCLE].transform("max")
    df["RUL_Capped"] = (max_c - df[CYCLE]).clip(upper=cap)
    return df


def fit_preprocessor(train_df, sensors):
    """Fit on TRAIN engines only. Returns a dict that can be joblib-dumped."""
    km = KMeans(n_clusters=N_CONDITIONS, n_init=10, random_state=SEED)
    km.fit(train_df[OP_COLS])
    cond = km.predict(train_df[OP_COLS])
    grp = train_df[sensors].groupby(cond)
    mean = grp.mean()
    std = grp.std().replace(0, 1.0).fillna(1.0)
    return {"km": km, "sensors": sensors, "mean": mean, "std": std}


def apply_preprocessor(df, pre):
    df = df.copy()
    cond = pre["km"].predict(df[OP_COLS])
    mu = pre["mean"].loc[cond].values
    sd = pre["std"].loc[cond].values
    df[pre["sensors"]] = (df[pre["sensors"]].values - mu) / sd
    return df


# --------------------------------------------------------------------------- #
# Windowing
# --------------------------------------------------------------------------- #
def make_train_windows(df, feats):
    """All stride-1 windows of length WINDOW. Returns X, y, engine ids, cycle at window end."""
    Xs, ys, es, cs = [], [], [], []
    for eng, g in df.sort_values([ENGINE, CYCLE]).groupby(ENGINE):
        if len(g) < WINDOW:
            continue
        a = g[feats].values.astype(np.float32)
        n = len(g) - WINDOW + 1
        idx = np.arange(WINDOW)[None, :] + np.arange(n)[:, None]
        Xs.append(a[idx])
        ys.append(g["RUL_Capped"].values[WINDOW - 1:])
        es.append(np.full(n, eng))
        cs.append(g[CYCLE].values[WINDOW - 1:])
    return (np.concatenate(Xs), np.concatenate(ys).astype(np.float32),
            np.concatenate(es), np.concatenate(cs).astype(np.float32))


def make_last_windows(df, feats):
    """One window per engine (its last WINDOW cycles), padded with the first row if short.
    Engines come out sorted by id, matching the row order of RUL_FD004.txt."""
    Xs, cs, es = [], [], []
    for eng, g in df.sort_values([ENGINE, CYCLE]).groupby(ENGINE):
        a = g[feats].values.astype(np.float32)[-WINDOW:]
        if len(a) < WINDOW:
            a = np.vstack([np.repeat(a[:1], WINDOW - len(a), axis=0), a])
        Xs.append(a)
        cs.append(g[CYCLE].values[-1])
        es.append(eng)
    return np.stack(Xs), np.array(cs, dtype=np.float32), np.array(es)


def window_features(X, cycles):
    """Tabular summary of each window for XGBoost."""
    last = X[:, -1, :]
    mean10 = X[:, -10:, :].mean(axis=1)
    std10 = X[:, -10:, :].std(axis=1)
    t = np.arange(X.shape[1], dtype=np.float32)
    tc = t - t.mean()
    slope = (tc[None, :, None] * (X - X.mean(axis=1, keepdims=True))).sum(axis=1) / (tc ** 2).sum()
    return np.hstack([last, mean10, std10, slope, cycles[:, None]])


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


def nasa_score(y, p):
    d = np.asarray(p) - np.asarray(y)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)))


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def build_cnn_gru(window, n_feat):
    inp = layers.Input((window, n_feat))
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(inp)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Bidirectional(layers.GRU(64, return_sequences=True))(x)
    x = layers.GRU(32)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1)(x)
    return models.Model(inp, out)


def train_nn_ensemble(X_tr, y_tr, X_val, y_val, out_dir):
    nets = []
    for k in range(N_NN_SEEDS):
        set_seed(SEED + k)
        net = build_cnn_gru(X_tr.shape[1], X_tr.shape[2])
        net.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                    loss=tf.keras.losses.Huber(delta=10.0))
        cbs = [tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
               tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)]
        net.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=EPOCHS,
                batch_size=BATCH_SIZE, callbacks=cbs, verbose=2)
        net.save(out_dir / f"cnn_gru_seed{k}.keras")
        nets.append(net)
    return nets


def predict_nn(nets, X):
    return np.mean([n.predict(X, batch_size=512, verbose=0).ravel() for n in nets], axis=0)


def train_xgb(Xt_tr, y_tr, Xt_val, y_val):
    m = xgb.XGBRegressor(n_estimators=1500, learning_rate=0.03, max_depth=5,
                         subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                         early_stopping_rounds=50, random_state=SEED, n_jobs=-1)
    m.fit(Xt_tr, y_tr, eval_set=[(Xt_val, y_val)], verbose=False)
    return m


def tune_blend(p_nn, p_xgb, y):
    """Weight on the NN prediction that minimises validation RMSE."""
    grid = np.linspace(0, 1, 21)
    scores = [rmse(y, w * p_nn + (1 - w) * p_xgb) for w in grid]
    return float(grid[int(np.argmin(scores))])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    raw = add_rul(pd.read_csv(TRAIN_CSV))
    sensors = get_sensor_cols(raw)
    print(f"Using {len(sensors)} sensors: {sensors}")

    # Engine-wise split
    engines = np.array(sorted(raw[ENGINE].unique()))
    rng = np.random.default_rng(SEED)
    rng.shuffle(engines)
    n_val = int(len(engines) * VAL_FRACTION)
    val_eng, tr_eng = set(engines[:n_val]), set(engines[n_val:])
    tr_raw = raw[raw[ENGINE].isin(tr_eng)]
    val_raw = raw[raw[ENGINE].isin(val_eng)]
    print(f"Train engines: {len(tr_eng)}, validation engines: {len(val_eng)}")

    # Preprocessor fitted on train engines only
    pre = fit_preprocessor(tr_raw, sensors)
    joblib.dump(pre, OUT_DIR / "preproc.joblib")
    tr_df, val_df = apply_preprocessor(tr_raw, pre), apply_preprocessor(val_raw, pre)

    # Windows
    X_tr, y_tr, _, c_tr = make_train_windows(tr_df, sensors)
    X_val, y_val, _, c_val = make_train_windows(val_df, sensors)
    Xt_tr, Xt_val = window_features(X_tr, c_tr), window_features(X_val, c_val)
    print("Windows:", X_tr.shape, X_val.shape, "| tabular:", Xt_tr.shape)

    # Models
    xgb_model = train_xgb(Xt_tr, y_tr, Xt_val, y_val)
    xgb_model.save_model(OUT_DIR / "xgb.json")
    nets = train_nn_ensemble(X_tr, y_tr, X_val, y_val, OUT_DIR)

    # Blend weight on validation
    p_nn_val, p_xgb_val = predict_nn(nets, X_val), xgb_model.predict(Xt_val)
    w = tune_blend(p_nn_val, p_xgb_val, y_val)
    print(f"\nValidation RMSE (all windows of held-out engines)")
    print(f"  XGBoost : {rmse(y_val, p_xgb_val):.3f}")
    print(f"  CNN-GRU : {rmse(y_val, p_nn_val):.3f}")
    print(f"  Blend   : {rmse(y_val, w * p_nn_val + (1 - w) * p_xgb_val):.3f}  (NN weight {w:.2f})")
    print(f"  Error correlation NN vs XGB: "
          f"{np.corrcoef(p_nn_val - y_val, p_xgb_val - y_val)[0, 1]:.3f}")

    json.dump({"nn_weight": w, "window": WINDOW, "rul_cap": RUL_CAP,
               "sensors": sensors, "n_nn_models": N_NN_SEEDS},
              open(OUT_DIR / "config.json", "w"), indent=2)

    # Benchmark evaluation: last window per test engine
    if TEST_CSV.exists() and RUL_TXT.exists():
        test_raw = pd.read_csv(TEST_CSV)
        y_true = np.minimum(np.loadtxt(RUL_TXT), RUL_CAP)   # capped, like the training target
        X_te, c_te, _ = make_last_windows(apply_preprocessor(test_raw, pre), sensors)
        p_nn = predict_nn(nets, X_te)
        p_xgb = xgb_model.predict(window_features(X_te, c_te))
        p_bl = w * p_nn + (1 - w) * p_xgb
        print(f"\nBenchmark test (last window per engine, RUL capped at {RUL_CAP})")
        for name, p in [("XGBoost", p_xgb), ("CNN-GRU", p_nn), ("Blend", p_bl)]:
            print(f"  {name:8s} RMSE {rmse(y_true, p):7.3f}   NASA score {nasa_score(y_true, p):12.1f}")
    else:
        print("\nTest files not found; skipped benchmark evaluation.")


# --------------------------------------------------------------------------- #
# Serving helper (use this in the FastAPI app)
# --------------------------------------------------------------------------- #
def load_artifacts(art_dir=OUT_DIR):
    art_dir = Path(art_dir)
    cfg = json.load(open(art_dir / "config.json"))
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(art_dir / "xgb.json")
    nets = [tf.keras.models.load_model(art_dir / f"cnn_gru_seed{k}.keras")
            for k in range(cfg["n_nn_models"])]
    return {"cfg": cfg, "pre": joblib.load(art_dir / "preproc.joblib"),
            "xgb": xgb_model, "nets": nets}


def predict_rul(df, art):
    """df: raw history of ONE engine with columns Engine, Cycles, OP_Setting 1-3 and
    all Sensor_* columns, in the same format as the CSV (>= 1 row; padded if < 30)."""
    sensors = art["cfg"]["sensors"]
    X, cyc, _ = make_last_windows(apply_preprocessor(df, art["pre"]), sensors)
    p_nn = predict_nn(art["nets"], X)
    p_xgb = art["xgb"].predict(window_features(X, cyc))
    w = art["cfg"]["nn_weight"]
    return float((w * p_nn + (1 - w) * p_xgb)[0])


if __name__ == "__main__":
    main()