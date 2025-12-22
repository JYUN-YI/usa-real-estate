# src/data_cleaning_pipeline.py
import pandas as pd
import numpy as np

def clean_data(df):
    df = df.copy()

    # --- prev_sold_date → season ---
    df['prev_sold_date'] = pd.to_datetime(df['prev_sold_date'], errors='coerce')
    df['season'] = df['prev_sold_date'].dt.month % 12 // 3 + 1
    df['season'] = df['season'].fillna(0).astype(int)  # 0 = Unknown

    # --- metromicro ---
    if 'metromicro' in df.columns:
        # 將缺失值設為 0 (Unknown)
        df['metromicro'] = df['metromicro'].fillna(0)
    else:
        # 如果根本沒這欄
        df['metromicro'] = 0

    # --- 其他欄位缺失值可視需求填補 ---
    # 例如 price、bed、bath、house_size、acre_lot 等
    numeric_cols = ['price', 'bed', 'bath', 'house_size', 'acre_lot']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    return df
