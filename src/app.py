from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import xgboost as xgb
import os
from pathlib import Path

WINDOW = 30
NN_WEIGHT = 0.45          # Must equal your blend weight
state = {}
USE_XGB = True 

BASE = Path("E:/P2/Python/TurbofanEngine/git_FD004")
NN_PATH = BASE / "models" / "model.keras"
XGB_PATH = BASE / "models" / "model.skops" 
PRE_PATH = BASE / "artifacts_hybrid" / "preproc.joblib"



@asynccontextmanager
async def lifespan(app):
    import tensorflow as tf
 
    state["nn"] = tf.keras.models.load_model(NN_PATH)
    state["pre"] = joblib.load(PRE_PATH)          # pickled with scikit-learn 1.7.2: use the training environment
    state["xgb"] = None
    if USE_XGB:
        if XGB_PATH.suffix == ".skops":
            import skops.io as sio
            state["xgb"] = sio.load(XGB_PATH, trusted=["sklearn.tree._tree.Tree",
                                                       "xgboost.core.Booster",
                                                       "xgboost.sklearn.XGBRegressor"])
        else:
            import xgboost as xgb
            state["xgb"] = xgb.XGBRegressor()
            state["xgb"].load_model(XGB_PATH)
 
    n_feat = len(state["pre"]["sensors"])
    if state["nn"].input_shape[-1] != n_feat:      # fail at startup, not on the first request
        raise RuntimeError(f"NN expects {state['nn'].input_shape[-1]} features, preprocessor gives {n_feat}")
    yield

app = FastAPI(title="NASA C-MAPSS RUL Predictor", version="1.0", lifespan=lifespan)

class Payload(BaseModel):
    sensor_data: list

# --- CUSTOM PREPROCESS FUNCTION ---
def expected_cols(pre):
    return list(pre["km"].feature_names_in_) + list(pre["sensors"]) + ["Cycles"]   # 3 + 14 + 1 = 18

def preprocess(raw, pre):
    km, sensors = pre["km"], pre["sensors"]
    df_raw = pd.DataFrame(raw, columns=expected_cols(pre))
    cond = km.predict(df_raw[list(km.feature_names_in_)])
    mu = pre["mean"].loc[cond, sensors].to_numpy()
    sd = pre["std"].loc[cond, sensors].to_numpy()
    z = (df_raw[sensors].to_numpy(dtype=float) - mu) / sd
    return z, float(df_raw["Cycles"].iloc[-1])

def window_features(X, cycles):
    """Tabular summary of each window for XGBoost."""
    last = X[:, -1, :]
    mean10 = X[:, -10:, :].mean(axis=1)
    std10 = X[:, -10:, :].std(axis=1)
    t = np.arange(X.shape[1], dtype=np.float32)
    tc = t - t.mean()
    slope = (tc[None, :, None] * (X - X.mean(axis=1, keepdims=True))).sum(axis=1) / (tc ** 2).sum()
    return np.hstack([last, mean10, std10, slope, cycles[:, None]])

def make_tabular(z, cycle):
    X = z[None, ...].astype(np.float32)
    return window_features(X, np.array([cycle], dtype=np.float32))


@app.post("/predict")
def predict(p: Payload):
    raw = np.asarray(p.sensor_data, dtype=float)
    n_cols = len(expected_cols(state["pre"]))
    if raw.ndim != 2 or raw.shape != (WINDOW, n_cols):
        raise HTTPException(400, f"Expected shape ({WINDOW}, {n_cols}), got {raw.shape}")
    try:
        z, cycle = preprocess(raw, state["pre"])
        p_nn = float(state["nn"](z[None, ...].astype(np.float32), training=False).numpy().ravel()[0])
        if USE_XGB:
            T = make_tabular(z, cycle)
            if T.shape[1] != state["xgb"].n_features_in_:
                raise ValueError(f"{T.shape[1]} features, XGB expects {state['xgb'].n_features_in_}")
            rul = NN_WEIGHT * p_nn + (1 - NN_WEIGHT) * float(state["xgb"].predict(T)[0])
        else:
            rul = p_nn
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")
    return {"predicted_rul": min(max(0.0, rul), 125.0)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)