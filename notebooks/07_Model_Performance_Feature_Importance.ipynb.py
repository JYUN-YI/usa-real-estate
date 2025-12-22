# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # 3. One-year house price prediction
# MAGIC
# MAGIC ### 3-1. Model Performance Overview (RMSE Comparison)
# MAGIC
# MAGIC **Interpretation:**
# MAGIC
# MAGIC The RMSE values indicate that XGBoost achieves significantly higher predictive accuracy than Random Forest on the test set.  
# MAGIC Specifically, XGBoost's RMSE is approximately $46,684, whereas Random Forest's RMSE is around $70,683.  
# MAGIC
# MAGIC This difference suggests that XGBoost is better at capturing non-linear relationships and interactions between features such as zip code, lot size, and the number of bedrooms and bathrooms.  
# MAGIC In contrast, Random Forest, while robust, is less sensitive to high-cardinality categorical features and the log-transformed target variable, resulting in higher prediction errors.
# MAGIC
# MAGIC Model Performance Overview
# MAGIC
# MAGIC | Model               | Test RMSE (in dollars) |
# MAGIC |--------------------|----------------------|
# MAGIC | XGBoost (F-score)  | $46,683.78           |
# MAGIC | Random Forest (MDI) | $70,682.97           |
# MAGIC
# MAGIC Note: Lower RMSE indicates better predictive performance.
# MAGIC <br>
# MAGIC <br>
# MAGIC ### 3-2. Comparing Feature Importance: XGBoost F-score vs Random Forest MDI
# MAGIC
# MAGIC 3-2-1. F-Score
# MAGIC
# MAGIC **F-Score** (also known as frequency) is the number of times a feature is used to split the data across all trees in the XGBoost model.
# MAGIC
# MAGIC Interpretation:
# MAGIC - **zip_code: 15372.0** → Used **15372 times**.
# MAGIC - **acre_lot: 14592.0** → This feature was used **14592 times** to split nodes across all decision trees.
# MAGIC - **bed: 10510.0** → Used **10510 times**.
# MAGIC - **bath: 9762.0** → Used **9762 times**.
# MAGIC - **season: 6448.0** → Used **6448 times**.
# MAGIC - **metromicro: 2151.0** → Used **2151 times**.
# MAGIC
# MAGIC Features with higher F scores contribute more significantly to the model’s decision-making.
# MAGIC
# MAGIC 3-2-2. MDI 
# MAGIC
# MAGIC **MDI (Mean Decrease in Impurity)** measures **how much a feature contributes to reducing the prediction** error across all trees in a Random Forest model.
# MAGIC
# MAGIC Higher values indicate that the feature **has a stronger impact on improving the model’s accuracy.**
# MAGIC
# MAGIC Interpretation:
# MAGIC - **bath: 0.610** → Contributes **61%** of the total reduction in impurity; most influential feature.
# MAGIC - **zip_code: 0.206** → Contributes **20.6%** of the total impurity reduction.
# MAGIC - **acre_lot: 0.135** → Contributes **13.5%.**
# MAGIC - **bed: 0.040** → Contributes **4.0%.**
# MAGIC - **season: 0.006** → Contributes **0.6%.**
# MAGIC - **metromicro: 0.004** → Contributes **0.4%**; least influential.
# MAGIC
# MAGIC **Key Point:**
# MAGIC - Unlike F-score, which counts how often a feature is used to split, MDI reflects **the actual contribution to improving predictions.**
# MAGIC - Features with higher MDI values **have more effect on reducing model error**, even if they are used less frequently in tree splits.

# COMMAND ----------

# MAGIC %md
# MAGIC ⚠️ Due to Databricks Free Edition runtime limitations, early stopping
# MAGIC parameters are not supported in XGBoost's sklearn API. Model complexity
# MAGIC is controlled via n_estimators and validation-based evaluation instead.

# COMMAND ----------

# Step 1: Create year-month feature
df_merged['year_month'] = df_merged['prev_sold_date'].dt.to_period('M').astype(str)

# Step 2: Select features and target
features = ['bed', 'bath', 'acre_lot', 'zip_code', 'season', 'metromicro']
target = 'price'

# Step 3: Drop missing values
df_model = df_merged.dropna(subset=features + [target]).copy()

# Step 4: Label encode categorical features
for col in ['zip_code', 'season', 'metromicro']:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col].astype(str))

# Step 5: Time-based split（train / test / val）
train_data = df_model[df_model['prev_sold_date'] < '2024-01-01']
val_data   = df_model[
    (df_model['prev_sold_date'] >= '2024-01-01') &
    (df_model['prev_sold_date'] < '2025-01-01')
]
test_data  = df_model[df_model['prev_sold_date'] >= '2025-01-01']

X_train = train_data[features]
X_val   = val_data[features]
X_test  = test_data[features]

y_train = np.log1p(train_data[target])
y_val   = np.log1p(val_data[target])
y_test  = np.log1p(test_data[target])

# 🔁 Log-transform target
y_train = np.log1p(train_data[target])
y_test = np.log1p(test_data[target])

# Step 6: Train XGBoost model
model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="rmse",
    random_state=42
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=True
)

# Step 7: Predict and evaluate
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)  # Reverse log1p
y_test_actual = np.expm1(y_test)

rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
print(f"✅ Test RMSE (in dollars): ${rmse:,.2f}")

# Step 8: Plot feature importance
plot_importance(model,importance_type='weight', max_num_features=10)
plt.title("XGBoost Feature Importance (F Score)")
plt.show()

# COMMAND ----------

#  Step 9：Train Random Forest
rf_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# Step 10：
rf_pred_log = rf_model.predict(X_test)

rf_pred = np.expm1(rf_pred_log)
rf_rmse = np.sqrt(mean_squared_error(y_test_actual, rf_pred))

print(f"🌲 RF Test RMSE (in dollars): ${rf_rmse:,.2f}")

# Step 11：RF Feature Importance
rf_importance = pd.Series(
    rf_model.feature_importances_,
    index=features
).sort_values(ascending=False)

print(rf_importance)

# Step 12：RF Feature Importance Visualization
plt.figure(figsize=(6, 4))
rf_importance.head(10).plot(kind='barh', color='#2ca02c')
plt.title("Random Forest Feature Importance (MDI)")
plt.xlabel("Importance")
plt.gca().invert_yaxis()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC XGBoost substantially outperforms Random Forest in terms of predictive accuracy, achieving a test RMSE of $46,684 compared to $70,683 for Random Forest.
# MAGIC
# MAGIC Feature importance analysis reveals a clear methodological contrast: XGBoost’s F-score emphasizes frequently used split variables such as zip code, whereas Random Forest’s MDI highlights features that most effectively reduce prediction error, with the number of bathrooms dominating the importance ranking.
# MAGIC
# MAGIC Despite these differences, both models consistently identify structural housing attributes—acreage, bedrooms, and bathrooms—as the primary drivers of housing prices, while temporal and regional classification variables play a secondary role.