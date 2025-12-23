# src/train.py
from xgboost import XGBRegressor
from src.features import create_features

def train_model(df):
    X, y = create_features(df, training=True)
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )
    model.fit(X, y)

    return model
