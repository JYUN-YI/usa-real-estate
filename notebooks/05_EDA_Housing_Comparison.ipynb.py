# Databricks notebook source
# MAGIC %md
# MAGIC ### 2-4. The housing price between Metropolitan and Micropolitan Areas
# MAGIC
# MAGIC 2-4-1.Interpretation of Violin Plot
# MAGIC
# MAGIC The violin plot shows the distribution of housing prices in Metropolitan (Metro) vs. Micropolitan (Micro) areas. Here's what the shapes and lines represent:
# MAGIC
# MAGIC **Violin width**: Represents the density of data points at different price levels (how concentrated the prices are).
# MAGIC **Box inside the violin**: Shows the interquartile range (IQR), with the white dot indicating the median price.
# MAGIC **Whiskers (lines extending from the box)**: Indicate the range of prices (minimum to maximum or a percentile range).
# MAGIC
# MAGIC Observations:
# MAGIC | Area Type |	Violin Width |	Whisker Length |	Interpretation |
# MAGIC |-----------|--------------|-----------------|-----------------|
# MAGIC |**Micropolitan**|	Wide	| Short|	Prices are clustered tightly within a narrow range, indicating less variation and more consistency in property prices. |
# MAGIC |**Metropolitan**|	Narrow |	Long |	Prices are more spread out with greater variability, including higher priced properties, indicating a more diverse market. |
# MAGIC
# MAGIC Summary:
# MAGIC
# MAGIC **Micropolitan areas** have housing prices concentrated around similar values with fewer extreme highs or lows.
# MAGIC
# MAGIC **Metropolitan areas** show a wider range of prices, reflecting a diverse housing market with both affordable and luxury homes.
# MAGIC
# MAGIC This explains why the Micropolitan violin looks wider and shorter, while the Metropolitan violin is narrower with longer whiskers.
# MAGIC

# COMMAND ----------

sns.violinplot(data=df_merged, x='metromicro', y='price')
plt.title("Price Comparison: Metropolitan vs. Micropolitan")

# COMMAND ----------

# MAGIC %md
# MAGIC 2-4-2. Welch’s t-test
# MAGIC
# MAGIC While Welch’s t-test shows a statistically significant difference in mean
# MAGIC housing prices between Metropolitan and Micropolitan areas (p < 0.001),
# MAGIC the effect size is small (Cohen’s d = 0.13). This suggests that although
# MAGIC the difference is consistent across a large sample, the magnitude of the
# MAGIC average price gap is modest relative to the overall price variability.
# MAGIC

# COMMAND ----------

metro_prices = df_merged[df_merged['metromicro'] == 'Metropolitan Statistical Area']['price']
micro_prices = df_merged[df_merged['metromicro'] == 'Micropolitan Statistical Area']['price']

t_stat, p_val = ttest_ind(
    metro_prices,
    micro_prices,
    equal_var=False,
    nan_policy='omit'
)

print(f"T-test result: t={t_stat:.2f}, p={p_val:.2e}")

# COMMAND ----------

metro_log = np.log1p(metro_prices)
micro_log = np.log1p(micro_prices)

mean_diff = metro_log.mean() - micro_log.mean()
pooled_std = np.sqrt(
    (metro_log.var() + micro_log.var()) / 2
)

cohens_d_log = mean_diff / pooled_std
print(f"Cohen's d (log price) = {cohens_d_log:.2f}")

sns.boxplot(
    x='metromicro',
    y='price',
    data=df_merged
)
plt.yscale('log')
plt.title('Housing Prices: Metro vs Micro (Log Scale)')
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC 2-4-3. Ranking of High-Demand Areas by Median Housing Price¶ (Bar Chart)
# MAGIC
# MAGIC Using CBSA titles to identify the most popular (i.e., expensive) housing markets based on median home prices.
# MAGIC
# MAGIC The top 10 CBSAs are ranked by their median property price.
# MAGIC

# COMMAND ----------

top_cbsa = df_merged.groupby('cbsatitle')['price'].median() \
    .sort_values(ascending=False).head(10)

norm = (top_cbsa - top_cbsa.min()) / (top_cbsa.max() - top_cbsa.min())
colors = [cm.Reds(x) for x in norm]

ax = top_cbsa.plot(kind='barh', figsize=(8, 5), color=colors)
for i, v in enumerate(top_cbsa):
    ax.text(v + 1000, i, f"${int(v):,}", va='center')
    
plt.xlabel("Median Price")
plt.title("Top 10 CBSAs by Median Housing Price")
plt.gca().invert_yaxis()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC The analysis of average housing prices by CBSA shows that price differences
# MAGIC are driven more by specific metropolitan areas than by broad metro vs micro
# MAGIC classification. While the overall metro–micro effect is modest, certain
# MAGIC CBSAs exhibit substantially higher average prices.
# MAGIC