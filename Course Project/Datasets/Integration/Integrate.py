import pandas as pd

# Load datasets
air_visits = pd.read_csv('air_visit_data.csv')
air_stores = pd.read_csv('air_store_info.csv')
date_info = pd.read_csv('date_info.csv')

# Parse date columns to datetime
air_visits['visit_date'] = pd.to_datetime(air_visits['visit_date'])
date_info['calendar_date'] = pd.to_datetime(date_info['calendar_date'])

# Merge air_visit_data with air_store_info on air_store_id
# Left join ensures we retain all historical visit records
merged_df = pd.merge(air_visits, air_stores, on='air_store_id', how='left')

# Merge with date_info on visit_date == calendar_date
merged_df = pd.merge(
    merged_df,
    date_info,
    left_on='visit_date',
    right_on='calendar_date',
    how='left',
)

# Drop redundant calendar_date column after merge
merged_df.drop(columns=['calendar_date'], inplace=True)

# Extract essential temporal features for modeling
merged_df['year'] = merged_df['visit_date'].dt.year
merged_df['month'] = merged_df['visit_date'].dt.month
merged_df['day'] = merged_df['visit_date'].dt.day
merged_df['day_of_week_num'] = merged_df['visit_date'].dt.dayofweek
merged_df['is_weekend'] = (
    merged_df['day_of_week_num'].isin([5, 6]).astype(int)
)

# Inspect merged dataset
print("Merged dataset shape:", merged_df.shape)
print("Missing values per column:\n", merged_df.isnull().sum())
print("\nFirst 5 rows:")
print(merged_df.head())

# Export ready-to-train dataset with UTF-8 BOM
merged_df.to_csv('restaurant_visits_integrated.csv', index=False, encoding='utf-8-sig')