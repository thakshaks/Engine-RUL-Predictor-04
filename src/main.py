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
import time
import mlflow
import mlflow.pytorch
import mlflow.keras
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from preprocessing import get_sensor_cols, fit_preprocessor, apply_preprocessor

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
EPOCHS = 20
BATCH_SIZE = 128
SEED = 42

# def get_sensor_cols(df):
#     drop = {f"Sensor_{i}" for i in DROP_SENSOR_IDS}
#     return [c for c in df.columns if c.startswith("Sensor_") and c not in drop]

def add_rul(df, cap=RUL_CAP):  
    """Run-to-failure data only (train_FD004). NOT for the benchmark test file."""
    df = df.sort_values([ENGINE, CYCLE]).copy()
    max_c = df.groupby(ENGINE)[CYCLE].transform(max)
    df["RUL_Capped"] = (max_c - df[CYCLE]).clip(upper=cap)
    return df

# def fit_preprocessor(train_df, sensors):
#     """Fit on TRAIN engines only. Returns a dict that can be joblib-dumped."""
#     km = KMeans(n_clusters=N_CONDITIONS, n_init=10, random_state=SEED)
#     km.fit(train_df[OP_COLS])
#     cond = km.predict(train_df[OP_COLS])
#     grp = train_df[sensors].groupby(cond)
#     mean = grp.mean()
#     std = grp.std().replace(0, 1.0).fillna(1.0)
#     return { "km": km, "sensors": sensors, "mean": mean, "std": std}


# def apply_preprocessor(df, pre):
#     df = df.copy()
#     cond = pre["km"].predict(df[OP_COLS])
#     mu = pre["mean"].loc[cond].values
#     sd = pre["std"].loc[cond].values
#     df[pre["sensors"]] = (df[pre["sensors"]].values - mu) / sd
#     return df 

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

def evaluate_predictions(y_true, y_pred):
    rmse2 = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    score = nasa_score(y_true, y_pred)
    return rmse2, mae, r2, score


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
    net = build_cnn_gru(X_tr.shape[1], X_tr.shape[2])
    net.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                loss=tf.keras.losses.Huber(delta=10.0))
    cbs = [tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)]
    net.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=EPOCHS,
            batch_size=BATCH_SIZE, callbacks=cbs, verbose=2)
    #net.save(out_dir / f"cnn_gru_seed{k}.keras")
        
    return net


def predict_nn(nets, X):
    return nets.predict(X, batch_size=512, verbose=0).ravel()


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
# Log into mlflow
# --------------------------------------------------------------------------- #

def log_metrics(rmse2, mae, r2, nasa):
     mlflow.log_metrics({
                "val_rmse": rmse2,
                "val_mae": mae,
                "val_r2": r2,
                "val_nasa_score": nasa,
            })

def main():
    # OUT_DIR.mkdir(parents=True, exist_ok=True)
    # set_seed(SEED)

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

    pre = fit_preprocessor(tr_raw, sensors)
    #print(pre)
    joblib.dump(pre, OUT_DIR / "preproc.joblib")
    tr_df, val_df = apply_preprocessor(tr_raw, pre), apply_preprocessor(val_raw, pre)
    print(tr_df,val_df)

    X_tr, y_tr, _, c_tr = make_train_windows(tr_df, sensors)
    X_val, y_val, _, c_val = make_train_windows(val_df, sensors)
    Xt_tr, Xt_val = window_features(X_tr, c_tr), window_features(X_val, c_val)
    print(X_tr[0])
    print("Windows:", X_tr.shape, X_val.shape, "| tabular:", Xt_tr.shape)

#     db_path = Path("E:/P2/Python/TurbofanEngine/git_FD004/mlflow_artifacts/mlflow.db").resolve().as_uri()
#     # This converts the path to: sqlite:///E:/P2/Python/TurbofanEngine/mlflow.db under the hood correctly or file:/// equivalent for MLflow backend config
#     for i in range (1,6):
#         mlflow.set_tracking_uri(f"sqlite:///{db_path.split(':///')[1]}")
#         mlflow.create_experiment(f"cmapss_turbofan_experiment_{i}", artifact_location="E:/P2/Python/TurbofanEngine/git_FD004/mlruns2")
#         mlflow.set_experiment(f"cmapss_turbofan_experiment_{i}")

#         with mlflow.start_run(run_name="xgb"):
#             xgb_model = train_xgb(Xt_tr, y_tr, Xt_val, y_val)
#             mlflow.sklearn.log_model(
#                 xgb_model,
#                 name="XGB_Model",
#                 skops_trusted_types=[
#                     "sklearn.tree._tree.Tree",
#                     "xgboost.core.Booster",
#                     "xgboost.sklearn.XGBRegressor",
#                 ],
#             )
#             p_xgb_val = xgb_model.predict(Xt_val)
#             rmse2, mae, r2, nasa = evaluate_predictions(y_val, p_xgb_val)
#             log_metrics(rmse2, mae, r2, nasa)

#         # --- NN ---
#         with mlflow.start_run(run_name="cnn_gru"):
#             nets = train_nn_ensemble(X_tr, y_tr, X_val, y_val, OUT_DIR)
#             mlflow.keras.log_model(nets, name="CNN_GRU")
#             p_nn_val = predict_nn(nets, X_val)
#             rmse2, mae, r2, nasa = evaluate_predictions(y_val, p_nn_val)
#             log_metrics(rmse2, mae, r2, nasa)

#         with mlflow.start_run(run_name="blended"):
#             w = tune_blend(p_nn_val, p_xgb_val, y_val)
#             rmse2, mae, r2, nasa = evaluate_predictions(y_val, w * p_nn_val + (1 - w) * p_xgb_val)
#             mlflow.log_metrics({"Optimised Weight": w,})
#             log_metrics(rmse2, mae, r2, nasa)

#     # Blend weight on validation
#     #p_nn_val, p_xgb_val = predict_nn(nets, X_val), xgb_model.predict(Xt_val)
#     #w = tune_blend(p_nn_val, p_xgb_val, y_val)
#     print(f"\nValidation RMSE (all windows of held-out engines)")
#     print(f"  XGBoost : {rmse(y_val, p_xgb_val):.3f}")
#     print(f"  CNN-GRU : {rmse(y_val, p_nn_val):.3f}")
#     print(f"  Blend   : {rmse(y_val, w * p_nn_val + (1 - w) * p_xgb_val):.3f}  (NN weight {w:.2f})")
#     print(f"  Error correlation NN vs XGB: "f"{np.corrcoef(p_nn_val - y_val, p_xgb_val - y_val)[0, 1]:.3f}")
    
#     # json.dump({"nn_weight": w, "window": WINDOW, "rul_cap": RUL_CAP,
#     #            "sensors": sensors, "n_nn_models": N_NN_SEEDS},
#     #             open(OUT_DIR / "config.json", "w"), indent=2)

if __name__ == "__main__":
    main()
