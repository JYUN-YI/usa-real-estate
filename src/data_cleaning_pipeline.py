# src/data_cleaning_pipeline.py
import pandas as pd
import numpy as np

def clean_data(df):
    df = df.copy()

    # --- Make sure the necessary fields exist ---
    required_cols = ['bed','bath','acre_lot','zip_code','season','metromicro','prev_sold_date','price']
    for col in required_cols:
        if col not in df.columns:
            if col == 'prev_sold_date':
                df[col] = pd.NaT
            else:
                df[col] = 0

    # --- prev_sold_date → season ---
    df['prev_sold_date'] = pd.to_datetime(df['prev_sold_date'], errors='coerce')
    df['season'] = df['prev_sold_date'].dt.month % 12 // 3 + 1
    df['season'] = df['season'].fillna(0).astype(int)  # 0 = Unknown

    # --- metromicro ---
    if 'metromicro' in df.columns:
        # Set up missing valuse as 0 (Unknown)
        df['metromicro'] = df['metromicro'].fillna(0)
    else:
        # If nothing for the column
        df['metromicro'] = 0

    # --- Missing values ​​in other fields can be filled in as needed. ---
    # e.g. price、bed、bath、house_size、acre_lot 等
    numeric_cols = ['price', 'bed', 'bath', 'house_size', 'acre_lot']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    return df
