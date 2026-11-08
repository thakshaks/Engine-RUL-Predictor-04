import pandas as pd
import requests
import openpyxl
from preprocessing import get_sensor_cols, fit_preprocessor, apply_preprocessor

excel_path = "E:/P2/Python/TurbofanEngine/git_FD004/data/test_FD004.xlsx"
df = pd.read_excel(excel_path)

workbook = openpyxl.load_workbook("E:/P2/Python/TurbofanEngine/git_FD004/data/test_FD004.xlsx")
sheet = workbook["Sheet3"] 

sensors= get_sensor_cols(df)
# pre = fit_preprocessor(df, sensors)
# tr_df = apply_preprocessor(df, pre)

# print(tr_df.head())



feature_columns = ["OP_Setting 1", "OP_Setting 2", "OP_Setting 3"]+ sensors + ["Cycles"]
window_data = df[feature_columns].tail(30)
l2=[]
for eng_id,group in df.groupby("Engine"):
    l1=[]
    if len(group)>=30:
        window_data = group[feature_columns].tail(30)
        for index, row in window_data.iterrows():
            l1.append(row.tolist())
    else:
        n=30-len(group)
        rows = group[feature_columns].values.tolist()          # the real cycles
        l1 = rows + [list(rows[-1]) for _ in range(n)]         # copy of the last row repeated n times
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
        l2.append("NULL")
        print(f"Error {response.status_code}:", len(group))

target_column = 2  
start_row = 2
for index, value in enumerate(l2):
    current_row = start_row + index
    sheet.cell(row=current_row, column=target_column, value=value)   

workbook.save(excel_path)   

