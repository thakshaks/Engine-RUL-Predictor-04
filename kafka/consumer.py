import json
import requests
from collections import defaultdict, deque
from kafka import KafkaConsumer

# Maintain a rolling 30-cycle buffer for each engine ID
engine_windows = defaultdict(lambda: deque(maxlen=30))

consumer = KafkaConsumer(
    'engine-telemetry',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

print("Kafka Consumer started. Listening for engine telemetry...")

for message in consumer:
    data = message.value
    engine_id = data["engine_id"]
    feature_dict = data["features"]
    
    # Extract feature values in the exact order expected by your preprocessor
    # (Ensure this matches the feature columns array defined in your training pipeline)
    feature_values = list(feature_dict.values())
    
    # Append to the engine's rolling window queue
    engine_windows[engine_id].append(feature_values)
    
    # Once the window reaches 30 samples, trigger your FastAPI app
    if len(engine_windows[engine_id]) == 30:
        payload = {
            "sensor_data": list(engine_windows[engine_id])
        }
        

        try:
            # Call your running FastAPI application endpoint
            response = requests.post("http://localhost:8000/predict", json=payload)
            if response.status_code == 200:
                result = response.json()
                print(f"[RUL Prediction] Engine {engine_id} | Predicted Remaining Useful Life: {result['predicted_rul']:.2f} cycles")
            else:
                print(f"FastAPI Error: {response.text}")
        except Exception as e:
            print(f"Connection error to FastAPI: {e}")