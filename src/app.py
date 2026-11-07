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
USE_XGB = False  

BASE = Path("E:/P2/Python/TurbofanEngine/git_FD004")
NN_PATH = BASE / "models" / "model.keras"
XGB_PATH = BASE / "models" / "xgb.json" 
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
def preprocess(raw, pre) -> np.ndarray:
    """
    Takes the raw 30-step numpy array, converts to DataFrame using training column names,
    predicts the flight condition cluster via KMeans, and normalizes using cluster mean/std.
    """
    km = pre["km"]
    sensors = pre["sensors"]
    
    # Get expected feature names from the KMeans model
    feature_names = list(km.feature_names_in_)
    
    
    # Construct a DataFrame from the raw array matching expected feature columns + sensors
    # Ensure your client payload sends columns in the exact order expected here.
    total_cols = feature_names + [s for s in sensors if s not in feature_names]
    df_raw = pd.DataFrame(raw, columns=total_cols[:raw.shape[1]])
    
    
    # Predict operating condition cluster
    cond = km.predict(df_raw[feature_names])
    
    # Retrieve cluster-specific means and standard deviations
    mu = pre["mean"].loc[cond, sensors].to_numpy()
    sd = pre["std"].loc[cond, sensors].to_numpy()

    
    print("Preprocessing Successful")
    y=(df_raw[sensors].to_numpy(dtype=float) - mu) / sd
    print("Normalized sensor data shape:", y.shape)
    
    # Normalize sensor data
    return y

def make_tabular(z: np.ndarray) -> np.ndarray:
    """
    Flattens your (30, Features) window into a 2D array (1, 30 * Features)
    if your XGBoost model expects flattened tabular features.
    """
    return z.reshape(1, -1)

@asynccontextmanager
async def lifespan(app):
    # Load your models and pre-fitted joblib scaler artifact on startup
    state["nn"] = tf.keras.models.load_model("E:/P2/Python/TurbofanEngine/git_FD004/models/model.keras")
    state["xgb"] = xgb.XGBRegressor()
    state["xgb"].load_model("E:/P2/Python/TurbofanEngine/git_FD004/models/model.skops")
    state["pre"] = joblib.load("E:/P2/Python/TurbofanEngine/git_FD004/artifacts_hybrid/preproc.joblib")
    yield


@app.post("/predict")
def predict(p: Payload):
    raw = np.asarray(p.sensor_data, dtype=float)
    print("Received raw sensor data shape:", raw.shape)
    if raw.shape[0] != WINDOW:
        raise HTTPException(400, f"Need exactly {WINDOW} time steps.")
    
    try:
        # 1. Preprocess the raw data using the joblib scaler artifact
        z = preprocess(raw, state["pre"])                 # Shape: (30, F)
        print(z)

        # 2. Neural Network prediction (expects 3D tensor: batch, window, features)
        p_nn = float(state["nn"](z[None, ...], training=False).numpy().ravel()[0])
        
        
        # 3. Optional XGBoost prediction
        if USE_XGB and state["xgb"] is not None:
            p_xgb = float(state["xgb"].predict(make_tabular(z))[0])
            # Corrected ensemble calculation syntax
            rul = NN_WEIGHT * p_nn + (1 - NN_WEIGHT) * p_xgb
        else:
            rul = p_nn
        
        print(rul)
         
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")
        
    return {"predicted_rul": max(0.0, rul)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)