# Databricks notebook source
# MAGIC %pip install gradio
# MAGIC %pip install xgboost

# COMMAND ----------

import gradio as gr

import sys

project_root = '/Workspace/Users/kimjylin@gmail.com/USA-Real-Estate-Analysis'
if project_root not in sys.path:
    sys.path.append(project_root)

from src.predict import predict
from src.train import train_model
from src.data_cleaning_pipeline import clean_data
import pandas as pd

model = train_model(df_clean)


# 假設 df_clean 已經處理好，model 已經訓練好
def gradio_predict(bed, bath, acre_lot, zip_code, season, metromicro):
    import pandas as pd
    df_input = pd.DataFrame({
        'bed': [bed],
        'bath': [bath],
        'acre_lot': [acre_lot],
        'zip_code': [zip_code],
        'season': [season],
        'metromicro': [metromicro]
    })
    return predict(df_input, model=model)[0]

iface = gr.Interface(
    fn=gradio_predict,
    inputs=[
        gr.Number(label="Bed"),
        gr.Number(label="Bath"),
        gr.Number(label="Acre Lot"),
        gr.Number(label="ZIP Code"),
        gr.Number(label="Season (0=Unknown,1=Winter...)"),
        gr.Number(label="Metromicro (0=Unknown,1,2...)")
    ],
    outputs=gr.Number(label="Predicted Price")
)

iface.launch()
