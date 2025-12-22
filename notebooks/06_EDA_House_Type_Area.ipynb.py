# Databricks notebook source
# MAGIC %md
# MAGIC ### 2-5. House Type and Area Analysis
# MAGIC
# MAGIC From the plot, we can see that property types in metropolitan areas are more diverse than in micropolitan areas, consistent with the findings in the price comparison. More price ranges are available, offering a wider variety of property types.

# COMMAND ----------

features = ['price', 'acre_lot', 'bed', 'bath']
fig, axes = plt.subplots(2, 2, figsize=(14, 10))  # 2 rows, 2 columns

label_dict = {
    'price': 'Price (USD)',
    'acre_lot': 'Lot Size (Acres)',
    'bed': 'Number of Bedrooms',
    'bath': 'Number of Bathrooms'
}

for i, col in enumerate(features):
    ax = axes[i // 2, i % 2]  # Get subplot position
    sns.boxplot(x='metromicro', y=col, data=df_merged, ax=ax)
    ax.set_title(f'{col} by Metro/Micro Area')
    ax.set_xlabel('')  # Optional: clean up x-label
    ax.set_ylabel(label_dict[col])

plt.tight_layout()
plt.show()