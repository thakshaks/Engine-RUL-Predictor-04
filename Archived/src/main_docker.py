from fastapi import FastAPI, HTTPException
import numpy as np
import joblib
import tensorflow as tf
import os

app = FastAPI(title="NASA C-MAPSS RUL Predictor", version="1.0")

# Load model and scaler paths
MODEL_PATH = os.getenv("MODEL_PATH", "/app/model.keras")
SCALER_PATH = os.getenv("SCALER_PATH", "/app/scaler.save")

model = None
scaler = None

@app.on_event("startup")
def load_artifacts():
    global model, scaler
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)

@app.post("/predict")
def predict_rul(payload: dict):
    """
    Expects JSON payload with recent sensor data readings:
    {
      "sensor_data": [[val1, val2, ...], [val1, val2, ...], ...] (Must have window size of 30 steps)
    }
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or scaler not loaded.")
    
    try:
        raw_data = np.array(payload["sensor_data"])
        
        # Ensure correct window size (30 time steps)
        if raw_data.shape[0] != 30:
            raise HTTPException(status_code=400, detail="Sensor data must contain exactly 30 time steps.")
            
        # Scale incoming features using the fitted MinMaxScaler
        scaled_data = scaler.transform(raw_data)
        
        # Reshape to (1, WindowSize, Features) for the GRU network
        input_array = np.expand_dims(scaled_data, axis=0)
        
        # Make prediction
        prediction = model.predict(input_array)
        predicted_rul = float(prediction[0][0])
        
        return {"predicted_rul": predicted_rul}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)