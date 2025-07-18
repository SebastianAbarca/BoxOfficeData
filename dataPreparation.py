import pandas as pd
import inflation_data

# Load box office data
box_office_df = pd.read_csv('data/BoxOfficeData.csv')
box_office_df.rename(columns={'Release Group': 'Movie'}, inplace=True)

# Clean monetary columns
money_columns = ['$Worldwide', '$Domestic', '$Foreign']
for col in money_columns:
    if col in box_office_df.columns:
        box_office_df[col] = (
            box_office_df[col]
            .astype(str)
            .str.replace('$', '', regex=False)
            .str.replace(',', '', regex=False)
        )
        box_office_df[col] = pd.to_numeric(box_office_df[col], errors='coerce')

# Clean year
box_office_df['Year'] = pd.to_numeric(box_office_df['Year'], errors='coerce')
box_office_df.dropna(subset=['Year'], inplace=True)
box_office_df['Year'] = box_office_df['Year'].astype(int)

# Rename for consistency
box_office_df.rename(columns={
    '$Worldwide': 'Worldwide_Original',
    '$Domestic': 'Domestic_Original',
    '$Foreign': 'Foreign_Original'
}, inplace=True)

# Filter top 10 per year
df_top_ten = box_office_df[box_office_df['Rank'] <= 10].copy()

# (Removed genre exploding — keep as-is)
df_top_ten['Genres'] = df_top_ten['Genres'].astype(str)
genre_bridge = df_top_ten[['Movie', 'Year', 'Genres']].drop_duplicates()
genre_bridge['Genres'] = genre_bridge['Genres'].astype(str).str.split(',')
genre_bridge = genre_bridge.explode('Genres')
genre_bridge['Genres'] = genre_bridge['Genres'].str.strip()
genre_bridge.to_csv('genre_bridge.csv', index=False)
print(df_top_ten['Production_Countries'].unique())
df_production_bridge = df_top_ten[['Movie', 'Year', 'Production_Countries']].copy()
df_production_bridge['Production_Countries'] = df_production_bridge['Production_Countries'].astype(str)
df_production_bridge['Production_Countries'] = df_production_bridge['Production_Countries'].str.split(',')
df_production_bridge = df_production_bridge.explode('Production_Countries')
df_production_bridge['Production_Countries'] = df_production_bridge['Production_Countries'].str.strip()
df_production_bridge = df_production_bridge[df_production_bridge['Production_Countries'].notna()]
df_production_bridge = df_production_bridge[df_production_bridge['Production_Countries'] != '']
df_production_bridge = df_production_bridge.drop_duplicates()
print(df_production_bridge['Production_Countries'].unique())

df_production_bridge.to_csv('production_bridge.csv', index=False)
# Reshape to long format for adjustment
df_melted = df_top_ten.melt(
    id_vars=['Movie', 'Year', 'Rank', 'Production_Countries', 'Genres'],
    value_vars=['Domestic_Original', 'Worldwide_Original'],
    var_name='BoxOfficeType',
    value_name='Amount'
)

# Load inflation data
start_year = df_melted['Year'].min()
end_year = df_melted['Year'].max()
df_inflation = inflation_data.get_inflation_data(start_year, end_year)

# Calculate inflation factor
base_year = df_inflation['Year'].max()
base_cpi = df_inflation.loc[df_inflation['Year'] == base_year, 'Inflation rate'].values[0]
df_inflation['InflationFactor'] = base_cpi / df_inflation['Inflation rate']

# Merge and adjust
df_adjusted = df_melted.merge(
    df_inflation[['Year', 'InflationFactor']],
    on='Year',
    how='left'
)
df_adjusted['AdjustedAmount'] = df_adjusted.apply(
    lambda row: row['Amount'] * row['InflationFactor']
    if row['BoxOfficeType'] == 'Domestic_Original' else row['Amount'],
    axis=1
)

# Pivot back to wide
df_wide = df_adjusted.pivot_table(
    index=['Movie', 'Year', 'Rank', 'Production_Countries', 'Genres'],
    columns='BoxOfficeType',
    values=['Amount', 'AdjustedAmount'],
    aggfunc='first'
).reset_index()

df_wide.columns = [
    'Movie', 'Year', 'Rank', 'Production_Countries', 'Genres',
    'Domestic_Original', 'Worldwide_Original',
    'Domestic_Adjusted', 'Worldwide_Unused'
]
df_wide.drop(columns=['Worldwide_Unused'], inplace=True)

# Merge foreign revenue
df_wide = df_wide.merge(
    df_top_ten[['Movie', 'Year', 'Foreign_Original']],
    on=['Movie', 'Year'],
    how='left'
)

# Compute adjusted worldwide
df_wide['Worldwide_Adjusted'] = df_wide['Domestic_Adjusted'] + df_wide['Foreign_Original']

# Load income data
df_household_income = pd.read_csv('data/household_income.csv')

# Merge with income
df_final = df_wide.merge(
    df_household_income[['Year', 'MedianIncome']],
    on='Year',
    how='left'
)

# Add key ratios
df_final['Domestic_Adjusted_to_Income_Ratio'] = df_final['Domestic_Adjusted'] / df_final['MedianIncome']
df_final['Domestic_to_Foreign_Ratio'] = df_final['Domestic_Original'] / df_final['Foreign_Original']

# Merge inflation again for convenience
df_final = df_final.merge(
    df_inflation[['Year', 'InflationFactor']],
    on='Year',
    how='left'
)

# Sort and compute income growth
df_final = df_final.sort_values(['Movie', 'Year']).reset_index(drop=True)
df_final = df_final.sort_values('Year').reset_index(drop=True)
df_final['MedianIncome_YoY_Growth'] = df_final['MedianIncome'].pct_change(fill_method=None) * 100

# Fix bad Worldwide_Original by recomputing if it's less than Domestic
df_final['Worldwide_Corrected'] = df_final['Worldwide_Original']
mask_fix = df_final['Worldwide_Original'] < df_final['Domestic_Original']
df_final.loc[mask_fix, 'Worldwide_Corrected'] = (
    df_final.loc[mask_fix, 'Domestic_Original'] + df_final.loc[mask_fix, 'Foreign_Original']
)
print(f"Corrected {mask_fix.sum()} rows where Worldwide < Domestic by recomputing.")

# Save full data
df_final.to_csv('final_box_office.csv', index=False)

# Yearly aggregates for trend analysis
df_yearly_totals = df_final.groupby('Year').agg({
    'Domestic_Original': 'sum',
    'Domestic_Adjusted': 'sum',
    'Worldwide_Original': 'sum',
    'Worldwide_Adjusted': 'sum',
    'Foreign_Original': 'sum',
    'MedianIncome': 'first'
}).reset_index()

# Compute growth rates
df_yearly_totals['YoY_Growth_Domestic_Original'] = df_yearly_totals['Domestic_Original'].pct_change(fill_method=None) * 100
df_yearly_totals['YoY_Growth_Domestic_Adjusted'] = df_yearly_totals['Domestic_Adjusted'].pct_change(fill_method=None) * 100
df_yearly_totals['YoY_Growth_Worldwide_Original'] = df_yearly_totals['Worldwide_Original'].pct_change(fill_method=None) * 100
df_yearly_totals['YoY_Growth_Worldwide_Adjusted'] = df_yearly_totals['Worldwide_Adjusted'].pct_change(fill_method=None) * 100
df_yearly_totals['YoY_Growth_MedianIncome'] = df_yearly_totals['MedianIncome'].pct_change(fill_method=None) * 100
df_yearly_totals['YoY_Growth_Foreign'] = df_yearly_totals['Foreign_Original'].pct_change(fill_method=None) * 100

# Save yearly summary
df_yearly_totals.to_csv('yearly_top10_summary.csv', index=False)

# Optionally print genres (no explosion means genre counts will be multi-label strings)
print("Genre sample (no explosion):")
print(df_final['Genres'].head())
