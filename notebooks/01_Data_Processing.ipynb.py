# Databricks notebook source
# MAGIC %md
# MAGIC # 1. Data Processing¶
# MAGIC
# MAGIC - Install and import libriaries
# MAGIC - Load the dataset
# MAGIC - Preview the data
# MAGIC - Format and standardize the data.

# COMMAND ----------

dbutils.library.restartPython()
%pip install xgboost
%pip install us

# COMMAND ----------

# Importing pandas for data analysis and manipulation (handling data frames)
import pandas as pd

# Used for array manipulation and matrix operations
import numpy as np

# Importing matplotlib for creating visualizations
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.cm as cm

# Import seaborn for statistical data visualization
import seaborn as sns

# Use choropleth maps 
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import MinMaxScaler to normalize feature to 0-1 range
from sklearn.preprocessing import MinMaxScaler

from xgboost import XGBClassifier
from xgboost import XGBRegressor
from xgboost import plot_importance

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

from scipy.stats import ttest_ind

# Import the 'us' library to access U.S. state information (names, abbreviations, FIPS codes, etc.)
import us

# To suppress warnings
import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

# Read dataset from Databricks Catalog
realtor_data_spark = spark.read.table("workspace.default.realtor_data_zip")
df = realtor_data_spark.toPandas()

print("Raw data:")
display(df.head())
display(df.info())

print("\n")

# Convert it to a pandas DataFrame for downstream EDA and modeling
def load_data():
    return spark.read.table("workspace.default.realtor_data_zip").toPandas()
    
# Load the dataset
df = load_data()

# Convert numeric columns to integers, even if it contains messy or non-numeric values
cols_to_int = ['brokered_by', 'price', 'bed', 'bath', 'street', 'zip_code', 'house_size']
for col in cols_to_int:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
# Fill the string-type columns with "Unknown" where values are NaN 
cols_to_fill = ['status', 'city', 'state']
df[cols_to_fill] = df[cols_to_fill].fillna("Unknown")

# Convert to datetime
df['prev_sold_date'] = pd.to_datetime(df['prev_sold_date'], errors='coerce')

# Fill missing values with placeholder
placeholder_date = pd.to_datetime("1900-01-01")
plot_df = df[df['prev_sold_date'] != placeholder_date]

# Fill NaN values in the float-type column with 0.0
df['acre_lot'] = df['acre_lot'].fillna(0.0)

print("Data standardization:")
display(df.head())
display(df.iloc[163364:163388])
display(df.iloc[[530530]])

# COMMAND ----------

# MAGIC %md
# MAGIC For further Metropolitan Statistical Area (MSA) analysis, related datasets were merged with consistent data types to ensure smooth processing.
# MAGIC
# MAGIC zipcode_FIPS_cbsa_crosswalk_2015.csv includes the following columns:
# MAGIC - zipcode
# MAGIC - FIPS
# MAGIC - CountyName
# MAGIC - cbsacode
# MAGIC - cbsatitle
# MAGIC - metromicro
# MAGIC
# MAGIC OMB_cbsa_2015.csv provides supplementary information and includes:
# MAGIC - CBSA Code
# MAGIC - CBSA Title
# MAGIC - Metropolitan/Micropolitan Statistical Area
# MAGIC
# MAGIC To fill in the missing cbsatitle and metromicro fields in zipcode_FIPS_cbsa_crosswalk_2015.csv, a merge was performed using the cbsacode as the key.

# COMMAND ----------

# MAGIC %md
# MAGIC Challenge: Dataset Alignment and Cross-Framework Integration
# MAGIC
# MAGIC To successfully merge multiple datasets, it was critical to ensure proper alignment of shared dimensions (e.g., CBSA code) across sources.
# MAGIC
# MAGIC One of the main challenges was handling messy official data, where column headers were not standardized and appeared in non-default rows. This required programmatically realigning headers and cleaning key fields before performing joins.
# MAGIC
# MAGIC Additionally, the workflow needed to seamlessly switch between Spark DataFrames and pandas DataFrames depending on the task.
# MAGIC
# MAGIC This hybrid approach ensured both scalability and analytical flexibility throughout the pipeline.

# COMMAND ----------

# Load data
crosswalk  = spark.read.table("workspace.default.zipcode_FIPS_cbsa_crosswalk_2015").toPandas()
cbsa_names  = spark.read.table("workspace.default.OMB_cbsa_2015").toPandas()

# Find the real header (The row 1, index=1)
cbsa_names.columns = cbsa_names.iloc[1]

# Drop the useless columns at the beginning.
cbsa_names = cbsa_names.iloc[3:].reset_index(drop=True)

print(cbsa_names.columns)

# COMMAND ----------

# Rename columns in cbsa_names dataframe to shorter, consistent lowercase names
cbsa_names = cbsa_names.rename(columns={
    'CBSA Code': 'cbsacode',
    'CBSA Title': 'cbsatitle',
    'Metropolitan/Micropolitan Statistical Area': 'metromicro'
})

# Ensure data type consistency
crosswalk['cbsacode'] = crosswalk['cbsacode'].astype(str).str.strip()
cbsa_names['cbsacode'] = cbsa_names['cbsacode'].astype(str).str.strip()

# To remedy the special value '99999', mark it as unknown
crosswalk.loc[crosswalk['cbsacode'] == '99999', ['cbsatitle', 'metromicro']] = ['No CBSA', 'Unknown']

# Merge crosswalk and cbsa_names, and do a left merge based on cbsacode (mainly crosswalk)
crosswalk_full = pd.merge(crosswalk, cbsa_names, on='cbsacode', how='left', suffixes=('_orig', ''))

# COMMAND ----------

# First make sure df['zip_code'] is a 5-code string format
df['zip_code'] = df['zip_code'].astype(str).str.zfill(5).str.strip()

# Make sure crosswalk_full['zipcode'] is in the same format
crosswalk_full['zipcode'] = crosswalk_full['zipcode'].astype(str).str.zfill(5).str.strip()

# Merge with df, and match zip_code to crosswalk_full’s zipcode
df_merged = df.merge(
    crosswalk_full[['zipcode', 'cbsatitle', 'metromicro']],
    left_on='zip_code',
    right_on='zipcode',
    how='left'
).drop(columns=['zipcode'])  # If no need the 'zipcode' field after merging, delete it

# Check the output
print(df_merged[['zip_code', 'cbsatitle', 'metromicro']].head(10))