import time
import json
import pandas as pd
from kafka import KafkaProducer
import sys
from pathlib import Path


target_folder = str(Path(__file__).resolve().parent.parent / "src")

if target_folder not in sys.path:
    sys.path.append(target_folder)
from preprocessing import get_sensor_cols

#Initialize Kafka Producer
producer = KafkaProducer(
    bootstrap_servers=['127.0.0.1:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    api_version=(2, 5, 0)
)

# Load your test dataset (e.g., FD004 test data)
test_df = pd.read_excel("E:/P2/Python/TurbofanEngine/git_FD004/data/test_FD004.xlsx ")
sensor_cols = get_sensor_cols(test_df)
feature_columns = ["OP_Setting 1", "OP_Setting 2", "OP_Setting 3"]+ sensor_cols + ["Cycles"]

#Stream each row chronologically
for _, row in test_df.iterrows():
    message = {
        "engine_id": int(row["Engine"]),
        "cycle": int(row["Cycles"]),
        # Extract the sensor and operational settings matching your feature list
        "features": row[feature_columns].to_dict()
    }


    producer.send("engine-telemetry", value=message)
    print(f"Produced -> Engine: {message['engine_id']} | Cycle: {message['cycle']}")
    
    # Simulate time delay between real-world sensor telemetry ticks
    time.sleep(0.05)

producer.flush()