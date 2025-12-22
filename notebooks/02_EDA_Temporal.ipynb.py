# Databricks notebook source
# MAGIC %md
# MAGIC # 2. EDA and Visualization
# MAGIC
# MAGIC ### 2-1. Time Series Subplot
# MAGIC
# MAGIC 2-1-1. Sales count over the past five years: Displays historical housing sales data with the year on the x-axis and the number of properties sold on the y-axis.
# MAGIC
# MAGIC 2-1-2. Average price over the past five years: Displays the date on the x-axis and the average price (normalised) on the y-axis.

# COMMAND ----------

# MAGIC %md
# MAGIC Interpret the normalised price values
# MAGIC
# MAGIC 0.0 = cheapest price in the filtered set (i.e., $10,000)
# MAGIC
# MAGIC 0.6 = a price that is 60% of the way between 10,000 and 5,000,000 (roughly $3,000,000)
# MAGIC
# MAGIC 1.0 = most expensive price in the filtered set (i.e., $5,000,000)
# MAGIC
# MAGIC The historical average normalised price shows four distinct peaks above 0.5, indicating that on those day, the average property prices were higher than approximately 2.5 million. Given the filtered price range (10,000 to 5,000,000), a normalised value above 0.5 refects days with the average sale prices leaned toward the upper end of the market.

# COMMAND ----------

# Define today's date and calculate date 5 years ago
today = pd.Timestamp.today()
five_years_ago = today - pd.DateOffset(years=5)

# Filter plot_df for the past five years
plot_recent_df = plot_df[(plot_df['prev_sold_date'] >= five_years_ago) & (df['prev_sold_date'] <= today)]

# Count number of sales per day
sales_trend = plot_recent_df.groupby('prev_sold_date').size().reset_index(name='sales_count')

# Filter for dates in the past 5 years
recent_df = df[(df['prev_sold_date'] >= five_years_ago) & (df['prev_sold_date'] <= today)]

# Remove unrealistic prices (e.g., under $10,000 or over $5 million)
filtered_recent_df = recent_df[(recent_df['price'] >= 10000) & (recent_df['price'] <= 5000000)]

# Normalize the price column using MinMaxScaler
scaler = MinMaxScaler()
filtered_recent_df['normalized_price'] = scaler.fit_transform(filtered_recent_df[['price']]).ravel()

# Group by date and calculate mean price per day
daily_avg = filtered_recent_df.groupby('prev_sold_date')['price'].mean().reset_index()
daily_avg_norm = filtered_recent_df.groupby('prev_sold_date')['normalized_price'].mean().reset_index()

# To inverse the normalization
original_prices = scaler.inverse_transform(daily_avg_norm[['normalized_price']])

# Create subplots
fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=False)

# Plot 1: Sales count over time
axs[0].plot(sales_trend['prev_sold_date'], sales_trend['sales_count'], color='blue')
axs[0].set_title("Number of Properties Sold Over the Past 5 Years")
axs[0].set_xlabel("Date")
axs[0].set_ylabel("Number of Properties Sold")
axs[0].grid(True)

# Plot 2: Average price over the past 5 years
axs[1].plot(daily_avg['prev_sold_date'], daily_avg_norm['normalized_price'], color='green')
axs[1].set_title("Price Trend Over the Past 5 Years")
axs[1].set_xlabel("Previous Sold Date")
axs[1].set_ylabel("Average Price (USD)")
axs[1].grid(True)

# Rotate x-axis labels for the second plot
axs[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Brief Report
# MAGIC
# MAGIC Over the past five years, the report highlights the days with highest and lowest number of properties sold, as well as the higheset and lowest average sold price.

# COMMAND ----------

# Count number of properties sold per day
daily_sales_count = plot_recent_df.groupby('prev_sold_date').size().reset_index(name='sales_count')

# Find the date with the highest number of sales
highest_sales_day = daily_sales_count.loc[daily_sales_count['sales_count'].idxmax()]

# Find the date with the lowest number of sales (greater than 0)
lowest_sales_day = daily_sales_count[daily_sales_count['sales_count'] > 0].loc[daily_sales_count['sales_count'].idxmin()]

# Display results
print("📈 Day with the highest number of property sales:")
print("Date:", highest_sales_day['prev_sold_date'].date())
print("Number of Sales:", highest_sales_day['sales_count'])

print("\n📉 Day with the lowest number of property sales (excluding 0):")
print("Date:", lowest_sales_day['prev_sold_date'].date())
print("Number of Sales:", lowest_sales_day['sales_count'])

# Group by date and calculate average sold price
daily_avg_price = recent_df.groupby('prev_sold_date')['price'].mean().reset_index()

# Drop any NaNs if present
daily_avg_price.dropna(subset=['price'], inplace=True)

# Find the date with the highest average price
highest_avg = daily_avg_price.loc[daily_avg_price['price'].idxmax()]

# Find the date with the lowest average price
lowest_avg = daily_avg_price.loc[daily_avg_price['price'].idxmin()]

# Display results
print("\n📈 Highest average sold price:")
print("Date:", highest_avg['prev_sold_date'].date())
print("Average Price: ${:,.2f}".format(highest_avg['price']))

print("\n📉 Lowest average sold price:")
print("Date:", lowest_avg['prev_sold_date'].date())
print("Average Price: ${:,.2f}".format(lowest_avg['price']))