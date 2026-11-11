import pandas as pd
import requests
import openpyxl

excel_path = "E:/P2/Python/TurbofanEngine/git/data/test_FD001.xlsx"
df = pd.read_excel(excel_path)

workbook = openpyxl.load_workbook("E:/P2/Python/TurbofanEngine/git/data/test_FD001.xlsx")
sheet = workbook["Sheet4"] 

# if len(df)<30:
#     print("Insufficient data. Please provide at least 30 time steps of sensor data.")


feature_columns = ["OP_Setting 1", "OP_Setting 2", "OP_Setting 3", "Sensor_2", "Sensor_3", "Sensor_4", "Sensor_7", "Sensor_8", "Sensor_9", "Sensor_11", "Sensor_12", "Sensor_13", "Sensor_14", "Sensor_15", "Sensor_17"]
# window_data = df[feature_columns].tail(30)
l2=[]
for eng_id,group in df.groupby("Engine"):
    l1=[]
    if len(group)>=30:
        window_data = group[feature_columns].tail(30)
        for index, row in window_data.iterrows():
            l1.append(row.tolist())
    s1=dict()
    s1["sensor_data"]=l1
    fastapi_url = "http://localhost:8000/predict"
    response = requests.post(fastapi_url, json=s1)
    if response.status_code == 200:
        print(f"Prediction Successful for!{eng_id}")
        f=response.json()
        z=f["predicted_rul"]
        l2.append(z)
        print("Response from FastAPI:", response.json())
    else:
        print(f"Error {response.status_code}:", response.text)

target_column = 2  
start_row = 2
for index, value in enumerate(l2):
    current_row = start_row + index
    sheet.cell(row=current_row, column=target_column, value=value)   

workbook.save(excel_path)   

# l1=[]
# for index, row in window_data.iterrows():
#     l1.append(row.tolist())
# s1=dict()
# s1["sensor_data"]=l1
    
    
    
    
    
    #print(s1)

#     fastapi_url = "http://localhost:8000/predict"

#     response = requests.post(fastapi_url, json=s1)

# # 7. Print out the prediction results
#     if response.status_code == 200:
#         print("Prediction Successful!")
#         print("Response from FastAPI:", response.json())
#     else:
#         print(f"Error {response.status_code}:", response.text)