# src/features.py
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

label_encoders = {}

def create_features(df, training=True):
    df = df.copy()
    
    # Build up year_month
    if 'prev_sold_date' in df.columns:
        df['year_month'] = df['prev_sold_date'].apply(
            lambda x: x.to_period('M').strftime("%Y-%m") if pd.notnull(x) else 'unknown'
        )
    else:
        df['year_month'] = 'unknown'
    
    categorical_cols = ['zip_code','season','metromicro','year_month']
    features = ['bed','bath','acre_lot','zip_code','season','metromicro']

    if training:
        # Fit LabelEncoder when training
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = df[col].astype(str)
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        # Perfrom log1p when getting target
        X = df[features]
        y = np.log1p(df['price'])
        return X, y
    else:
        # Only transform when predicting
        for col in categorical_cols:
            le = label_encoders[col]
            df[col] = df[col].astype(str).apply(lambda x: x if x in le.classes_ else 'unknown')
            df[col] = le.transform(df[col])
        X = df[features]
        return X, None
