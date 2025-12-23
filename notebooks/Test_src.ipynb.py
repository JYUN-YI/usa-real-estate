# ==============================
# Test_src.ipynb - Free Edition / UC Compatible (Safe Version)
# ==============================

# 1️⃣ Install required packages (first-time execution)
%pip install xgboost pandas numpy scikit-learn matplotlib seaborn gradio

# 2️⃣ Set project root path so Python can find src modules
import sys
project_root = '/Workspace/Users/kimjylin@gmail.com/USA-Real-Estate-Analysis'
if project_root not in sys.path:
    sys.path.append(project_root)

# 3️⃣ Import src modules

import importlib

from src import data_cleaning_pipeline
importlib.reload(data_cleaning_pipeline)

from src.data_cleaning_pipeline import clean_data
from src.features import create_features


import src.train
import src.predict

importlib.reload(src.train)
importlib.reload(src.predict)

from src.train import train_model
from src.predict import predict

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

# 4️⃣ Load Databricks tables
df_realtor = spark.table("workspace.default.realtor_data_zip")
df_cbsa = spark.table("workspace.default.OMB_cbsa_2015")

# 5️⃣ Convert Spark DataFrame to pandas
df_realtor_pd = df_realtor.toPandas()
df_cbsa_pd = df_cbsa.toPandas()

# 6️⃣ Auto-detect and rename CBSA columns
cbsa_code_col = [c for c in df_cbsa_pd.columns if 'CBSA' in c]
metromicro_col = [c for c in df_cbsa_pd.columns if 'Metropolitan' in c or 'Micropolitan' in c]

if cbsa_code_col and metromicro_col:
    df_cbsa_pd.rename(columns={
        cbsa_code_col[0]: 'cbsa_code',
        metromicro_col[0]: 'metromicro'
    }, inplace=True)
else:
    print("⚠️ Warning: CBSA columns not detected. 'metromicro' will be set to 'Unknown'.")

# 7️⃣ Build derived columns before clean_data()
# prev_sold_date to datetime
df_realtor_pd['prev_sold_date'] = pd.to_datetime(df_realtor_pd['prev_sold_date'], errors='coerce')

# season
df_realtor_pd['season'] = df_realtor_pd['prev_sold_date'].dt.month % 12 // 3 + 1
season_map = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
df_realtor_pd['season'] = df_realtor_pd['season'].map(season_map)
df_realtor_pd['season'].fillna('Unknown', inplace=True)

# metromicro
if 'cbsa_code' in df_realtor_pd.columns and 'cbsa_code' in df_cbsa_pd.columns:
    df_realtor_pd = df_realtor_pd.merge(
        df_cbsa_pd[['cbsa_code', 'metromicro']],
        on='cbsa_code',
        how='left'
    )
else:
    df_realtor_pd['metromicro'] = 'Unknown'

df_realtor_pd['metromicro'].fillna('Unknown', inplace=True)

# 8️⃣ Fill numeric missing values with reasonable defaults
numeric_cols = ['bed', 'bath', 'acre_lot', 'house_size', 'zip_code', 'price']
for col in numeric_cols:
    if col in df_realtor_pd.columns:
        df_realtor_pd[col].fillna(df_realtor_pd[col].median(), inplace=True)

# 9️⃣ Data cleaning
df_clean = clean_data(df_realtor_pd)
print("✅ Data cleaning done. Sample:")
print(df_clean.head())

# 10️⃣ Feature creation
X, y = create_features(df_clean)

# Encode categorical 'Unknown' as 0
for col in ['season', 'metromicro']:
    if col in X.columns:
        X[col] = X[col].replace({'Unknown': 0}).astype(int)

print("✅ Feature creation done. Sample features:")
print(X.head())

# 11️⃣ Train the model
model = train_model(df_clean)
print("✅ Model trained successfully.")

# 12️⃣ Make predictions safely (avoid overflow)
X_eval, y_true_log = create_features(df_clean)
for col in ['season', 'metromicro']:
    if col in X_eval.columns:
        X_eval[col] = X_eval[col].replace({'Unknown': 0}).astype(int)

y_pred_log = model.predict(X_eval)
y_pred_log = np.clip(y_pred_log, a_min=None, a_max=20)  # 避免 overflow
y_pred = np.expm1(y_pred_log)

print("✅ Prediction done. Sample predictions:")
print(y_pred[:5])

# 13️⃣ Evaluate model performance
# Clip predictions
y_pred_log = model.predict(X_eval)
y_pred_log = np.clip(y_pred_log, a_min=None, a_max=20)  # 避免 overflow

# Clip true values (log1p)
y_true_log = np.log1p(df_clean['price'].values)
y_true_log = np.clip(y_true_log, a_min=None, a_max=20)  # 同樣限制上限

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
y_pred = np.expm1(y_pred_log)  # 最後轉回 dollar scale
print(f"🌟 Test RMSE (log-scale): {rmse:.4f}")
