# 🏡 USA Real Estate Analysis
An end-to-end machine learning project for analyzing and predicting U.S. housing prices.
This project covers the full pipeline from raw data processing and exploratory data analysis (EDA) to model training, evaluation, and deployment as an interactive web application.

## 📌 Project Objective
**End-to-End USA Real Estate Price Prediction System**

From raw data → feature engineering → model training → experiment analysis → deployed interactive app.

The goal of this project is to:
- Analyze U.S. real estate market trends across time, geography, and seasonality
- Compare housing prices between metropolitan and micropolitan areas
- Identify key drivers of housing prices using machine learning models
- Build a one-year house price prediction model
- Deploy an interactive prediction interface using Gradio and Hugging Face Spaces

## 🏗️ Project Structure
```bash
📦 usa-real-estate
├── notebooks/
│   ├── 01_Data_Processing.ipynb
│   ├── 02_EDA_Temporal.ipynb
│   ├── 03_EDA_Geospatial.ipynb
│   ├── 04_EDA_Seasonal.ipynb
│   ├── 05_EDA_Housing_Comparison.ipynb
│   ├── 06_EDA_House_Type_Area.ipynb
│   ├── 07_Model_Performance_Feature_Importance.ipynb
│   ├── Test_src.ipynb
│   └── USA_Real_Estate_Analysis_Full.ipynb
│
├── src/
│   ├── data_cleaning_pipeline.py   # Data cleaning & preprocessing
│   ├── features.py                 # Feature engineering
│   ├── train.py                    # Model training
│   └── predict.py                  # Inference logic (Gradio)
│
├── models/
│   └── xgb_model.pkl
│
├── data/
│   ├── train.csv
│   └── README.md
│
├── app.py                          # Gradio app (Hugging Face Space)
├── requirements.txt
├── README.md
├── .gitattributes
└── .gitignore
```
## 🔍 Exploratory Data Analysis (EDA)
Key analyses include:
- Time Series Analysis: Long-term housing price trends
- Geospatial Analysis: Regional price differences across the U.S.
- Seasonal Analysis: Price fluctuations by season
- Metro vs. Micro Comparison:
  - Violin plot visualization
  - Welch’s t-test for statistical significance
  - Ranking high-demand areas by median housing price
- House Type & Area Analysis: Impact of property characteristics on price

## 🤖 Machine Learning Models
- Models Used
  - XGBoost Regressor
  - Random Forest Regressor (baseline comparison)  
- Evaluation Metric
  - RMSE (Root Mean Squared Error)

📊 Model Performance
| Model | RMSE (↓ better) |
|------|------|
| XGBoost | ⭐ Best |
| Random Forest | Baseline |

## 🔑 Feature Importance Analysis
Feature importance was analyzed using:
- XGBoost F-score
- Random Forest Mean Decrease in Impurity (MDI
Both methods consistently highlighted:
- Location-related features (ZIP code, metro/micro classification)
- Property characteristics (bedrooms, bathrooms, lot size)

## 🚀 Interactive Application
An interactive house price prediction app is deployed using **Gradio** on **Hugging Face Spaces**.

**User Inputs:**
- Bedrooms
- Bathrooms
- Lot size (acres)
- ZIP code
- Season (0 = unknown, 1–4)
- Metro / Micro classification (0 = unknown)
The app returns an estimated house price based on the trained model.

## ⚙️ Technologies Used
- Programming: Python
- Data Processing: Pandas, NumPy
- Visualization: Matplotlib, Seaborn
- Machine Learning: XGBoost, scikit-learn, scipy
- Deployment: Gradio, Hugging Face Spaces
- Version Control: Git, GitHub
- Development Environment: Databricks Free Edition

## 📎 Dataset

### Raw Data
- Kaggle USA Real Estate Dataset
- Zip Code to CBSA Crosswalk (2015), U.S. Census Bureau
- CBSA Delineation Reference (2015), Office of Management and Budget (OMB)

### Processed Data
- Cleaned, standardized, and feature-engineered dataset hosted on Hugging Face Datasets

## 🎨 Portfolio Showcase
Please view and interact with the live application here: 
[Kimberly Lin | Portfolio | USA Real Estate Analysis](https://kimberlylin.webflow.io/resources/usa-real-estate-analysis)
[Gradio App on Hugging Face Spaces](https://huggingface.co/spaces/jyunyilin/usa_real_estate)

## ✨ Future Improvements
- Add more property-level features (e.g., year built, house type)
- Improve spatial modeling using regional embeddings
- Extend prediction horizon beyond one year
- Add model monitoring and retraining pipeline
- Improve the Gradio application by aligning the inference pipeline with the training pipeline, ensuring consistent feature engineering and preprocessing for more accurate and stable real-time predictions.
