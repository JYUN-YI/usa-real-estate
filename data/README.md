---
license: mit
task_categories:
- tabular-regression
language:
- en
pretty_name: USA Real Estate Clean Dataset
---

# USA Real Estate Clean Dataset

This dataset contains cleaned U.S. real estate listing data prepared for machine learning regression tasks,
such as house price prediction.

The data has been preprocessed from raw realtor listings, including:
- Missing value handling
- Feature normalization
- Categorical encoding (season, metro/micro area)

## Dataset Structure

The dataset provides a single split:

- **train**: cleaned real estate listings

## Columns

| Column Name   | Description |
|--------------|------------|
| bed          | Number of bedrooms |
| bath         | Number of bathrooms |
| acre_lot     | Lot size in acres |
| zip_code     | ZIP code |
| season       | Season encoded as integer (0 = Unknown) |
| metromicro  | Metro/Micro classification encoded as integer (0 = Unknown) |
| price        | House listing price (target variable) |

## Intended Use

- Regression modeling
- Feature engineering experiments
- ML demos (Gradio / Streamlit / HF Spaces)

## Usage Example

```python
from datasets import load_dataset

dataset = load_dataset("jyunyilin/usa-real-estate")
df = dataset["train"].to_pandas()
print(df.head())
