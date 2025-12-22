# Databricks notebook source
# MAGIC %md
# MAGIC ### 2-2. Geospatial Analysis
# MAGIC
# MAGIC 2-2-1. Properties For Sale by State: The map highlights Florida (FL) as having the highest number of properties for sale, identified by a yellow shade. Several states in the eastern U.S., such as Illinois (IL), Georgia (GA), and North Carolina (NC), show relatively high numbers. Among them, New York (NY) stands out with a brighter violet shade, indicating a higher concentration—second only to Florida.
# MAGIC
# MAGIC 2-2-2. Properties Ready to Build by State: The map highlights Florida as having the highest number of properties ready to build, indicated by a yellow shade. Notably, Texas (TX), with its wide area, shows a relatively high number, represented by a orange shade—similar to Michigan (MI). Additionally, Pennsylvania (PA) displays a brighter orange hue, suggesting a higher concentration compared to the other two.

# COMMAND ----------

# Convert full state names to 2-letter abbreviations
def name_to_abbr(state_name):
    state = us.states.lookup(state_name)
    return state.abbr if state else None
    
# Group by state and status, pivot table
status_counts = df.groupby(['state', 'status']).size().unstack(fill_value=0).reset_index()

# Convert 'state' column from full names to abbreviations
status_counts['state'] = status_counts['state'].apply(name_to_abbr)

# Drop rows with invalid or unrecognized states
status_counts.dropna(subset=['state'], inplace=True)

# Create subplots with 1 row, 2 columns
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Properties For Sale by State", "Properties Ready to Build by State"),
    specs=[[{"type": "choropleth"}, {"type": "choropleth"}]],
    horizontal_spacing=0.2
    
)

# Choropleth for 'for_sale'
fig.add_trace(
    go.Choropleth(
        locations=status_counts['state'],
        z=status_counts['for_sale'],
        locationmode='USA-states',
        colorbar=dict(title='For Sale', x=0.45), # Move to the left
        showscale=True
    ),
    row=1, col=1
)

# Choropleth for 'ready_to_build'
fig.add_trace(
    go.Choropleth(
        locations=status_counts['state'],
        z=status_counts['ready_to_build'],
        locationmode='USA-states',
        colorbar=dict(title='Ready to Build', x=1.05), # Move to the far right
        showscale=True
    ),
    row=1, col=2
)

# Layout settings
fig.update_layout(
    title_text="Real Estate Status by State",
    geo=dict(scope='usa'), # First map (left)
    geo2=dict(scope='usa'), # Second map (right)
    height=600,
    width=1000,
    #margin=dict(l=50, r=50, t=50, b=50)
)

fig.show()