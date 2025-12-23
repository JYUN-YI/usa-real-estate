import pandas as pd
import gradio as gr
from datasets import load_dataset
from src.data_cleaning_pipeline import clean_data
from src.train import train_model
from src.predict import predict
from sklearn.preprocessing import LabelEncoder

# ----------------------------------------
# 1️⃣ Load HF Dataset and clean
# ----------------------------------------
dataset = load_dataset("jyunyilin/usa-real-estate")
df = dataset["train"].to_pandas()

# Fill necessary columns
if 'prev_sold_date' not in df.columns:
    df['prev_sold_date'] = pd.NaT
if 'price' not in df.columns:
    df['price'] = 0

# Clean data
df_clean = clean_data(df)

# ----------------------------------------
# 2️⃣ LabelEncoder for categorical features
# ----------------------------------------
categorical_cols = ['zip_code', 'season', 'metromicro', 'year_month']

# Create year_month column
df_clean['year_month'] = df_clean['prev_sold_date'].apply(
    lambda x: x.to_period('M').strftime("%Y-%m") if pd.notnull(x) else 'unknown'
)

# Fit LabelEncoders
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col].astype(str))
    label_encoders[col] = le

# ----------------------------------------
# 3️⃣ Train model
# ----------------------------------------
features = ['bed', 'bath', 'acre_lot', 'zip_code', 'season', 'metromicro']
target = 'price'
X_train = df_clean[features]
y_train = df_clean[target]
model = train_model(df_clean)

# ----------------------------------------
# 4️⃣ Gradio prediction function
# ----------------------------------------
def gradio_predict(bed, bath, acre_lot, zip_code, season, metromicro):
    # Build input DataFrame
    df_input = pd.DataFrame([{
        'bed': bed,
        'bath': bath,
        'acre_lot': acre_lot,
        'zip_code': zip_code,
        'season': season,
        'metromicro': metromicro,
        'prev_sold_date': pd.NaT  # auto fill
    }])

    # Clean data
    df_input_clean = clean_data(df_input)

    # Create year_month
    df_input_clean['year_month'] = df_input_clean['prev_sold_date'].apply(
        lambda x: x.to_period('M').strftime("%Y-%m") if pd.notnull(x) else 'unknown'
    )

    # Transform categorical features using fitted LabelEncoders
    for col in categorical_cols:
        le = label_encoders[col]
        # If new category not seen in training, map to -1
        df_input_clean[col] = df_input_clean[col].apply(
            lambda x: x if x in le.classes_ else 'unknown'
        )
        df_input_clean[col] = le.transform(df_input_clean[col].astype(str))

    # Select features
    X_input = df_input_clean[features]

    # Predict
    y_pred = predict(X_input, model=model)
    return float(y_pred[0])

# ----------------------------------------
# 5️⃣ Gradio UI
# ----------------------------------------
iface = gr.Interface(
    fn=gradio_predict,
    inputs=[
        gr.Number(label="Bedrooms"),
        gr.Number(label="Bathrooms"),
        gr.Number(label="Lot Size (acres)"),
        gr.Number(label="ZIP Code"),
        gr.Number(label="Season (0=Unknown, 1-4)"),
        gr.Number(label="Metro/Micro (0=Unknown)")
    ],
    outputs=gr.Textbox(label="Predicted House Price"),
    title="USA Real Estate Price Predictor",
    description="Enter housing information to predict prices"
)

if __name__ == "__main__":
    iface.launch()
