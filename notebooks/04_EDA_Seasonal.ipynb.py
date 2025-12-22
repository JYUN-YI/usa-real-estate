# Databricks notebook source
# MAGIC %md
# MAGIC ### 2-3. Seasonal Analysis (Spring/Summer vs. Fall/Winter) - Heatmaps

# COMMAND ----------

df_merged['prev_sold_date'] = pd.to_datetime(df_merged['prev_sold_date'])

def assign_season(month):
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'
    else:
        return 'Winter'

df_merged['season'] = df_merged['prev_sold_date'].dt.month.apply(assign_season)

season_price = df_merged.groupby(['season', 'metromicro'])['price'].mean().unstack()
season_count = df_merged.groupby(['season', 'metromicro']).size().unstack()

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # 1 row, 2 columns

# Heatmap 1: Average Price
sns.heatmap(season_price, annot=True, fmt=".0f", cmap="YlGnBu", ax=axes[0])
axes[0].set_title("Average Property Price by Season")
axes[0].set_xlabel("Metro Area Type")
axes[0].set_ylabel("Season")

# Heatmap 2: Number of Sales
sns.heatmap(season_count, annot=True, fmt="d", cmap="OrRd", ax=axes[1])
axes[1].set_title("Number of Property Sales by Season")
axes[1].set_xlabel("Metro Area Type")
axes[1].set_ylabel("")

plt.tight_layout()
plt.show()