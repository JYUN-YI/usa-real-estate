# src/predict.py
import pandas as pd
import numpy as np
from src.features import create_features

def predict(df_input, model):
    """
    Predict house prices using a trained model.
    Returns np.ndarray of predicted prices.
    """
    if model is None:
        raise ValueError("Please provide a trained model via the `model` argument.")

    # 1️⃣ Create features
    X, _ = create_features(df_input, training=False)

    # 2️⃣ Handle extreme values
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # 3️⃣ Predict log(price)
    y_pred_log = model.predict(X)

    # 4️⃣ Clip avoid overflow
    y_pred_log = np.clip(y_pred_log, a_min=None, a_max=20)  # log(price) 最大約 20 → ~5e8 美元

    # 5️⃣ Reverse log-transform
    y_pred = np.expm1(y_pred_log)

    return y_pred
