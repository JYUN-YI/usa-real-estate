# app.py

import pandas as pd
import numpy as np
import gradio as gr
from src.data_cleaning_pipeline import clean_data
from src.train import train_model
from src.predict import predict

# ----------------------------------------
# 1️⃣ 訓練模型（或你可以直接載入已訓練模型）
# ----------------------------------------
# 假設 df_clean 是你已經清理好的訓練資料
# from somewhere import df_clean  # 如果有 pickle 或 csv 可直接載入

# 這裡範例用空 DataFrame，請換成你的 df_clean
df_clean = pd.read_csv("data/clean_realtor.csv")  # 例：你已經存好清理後資料
model = train_model(df_clean)

# ----------------------------------------
# 2️⃣ Gradio 預測函數
# ----------------------------------------
def gradio_predict(input_dict):
    """
    input_dict: dict, key 是欄位名稱，value 是輸入值
    """
    # 1️⃣ 轉成 DataFrame
    df_input = pd.DataFrame([input_dict])

    # 2️⃣ 缺失值填補
    for col in ['season', 'metromicro']:
        if col not in df_input.columns or df_input[col].isnull().all():
            df_input[col] = 0
    numeric_cols = ['bed', 'bath', 'acre_lot', 'zip_code']
    for col in numeric_cols:
        if col not in df_input.columns:
            df_input[col] = 0
        df_input[col].fillna(df_input[col].median(), inplace=True)

    # 3️⃣ 清理資料
    df_input_clean = clean_data(df_input)

    # 4️⃣ 預測
    y_pred = predict(df_input_clean, model=model)

    return float(y_pred[0])

# ----------------------------------------
# 3️⃣ Gradio UI
# ----------------------------------------
input_fields = {
    'bed': gr.Number(label="Bedrooms"),
    'bath': gr.Number(label="Bathrooms"),
    'acre_lot': gr.Number(label="Lot Size (acres)"),
    'zip_code': gr.Number(label="ZIP Code"),
    'season': gr.Number(label="Season (0=Unknown, 1-4)"),
    'metromicro': gr.Number(label="Metro/Micro (0=Unknown)")
}

iface = gr.Interface(
    fn=gradio_predict,
    inputs=[gr.Number(label=k) for k in input_fields.keys()],
    outputs=gr.Textbox(label="Predicted House Price"),
    title="USA Real Estate Price Predictor",
    description="輸入房屋資訊預測價格"
)

if __name__ == "__main__":
    iface.launch()
