import unicodedata
import pandas as pd

# 1. Load datasets
air_visits = pd.read_csv(r'..\data\integration\air_visit_data.csv')
air_stores = pd.read_csv(r'..\data\integration\air_store_info.csv')
date_info = pd.read_csv(r'..\data\integration\date_info.csv')


# 2. Helper function to normalize Japanese romaji and extract prefecture name
def clean_prefecture(series):
  normalized = series.apply(
      lambda x: unicodedata.normalize('NFKD', str(x))
      .encode('ascii', 'ignore')
      .decode('utf-8')
  )
  return normalized.str.split(' ').str[0].str.split('-').str[0].str.lower()


air_stores['air_area_name'] = clean_prefecture(air_stores['air_area_name'])

# 3. Parse date columns to datetime
air_visits['visit_date'] = pd.to_datetime(air_visits['visit_date'])
date_info['calendar_date'] = pd.to_datetime(date_info['calendar_date'])

# 4. Merge datasets
merged_df = pd.merge(air_visits, air_stores, on='air_store_id', how='left')
merged_df = pd.merge(
    merged_df,
    date_info,
    left_on='visit_date',
    right_on='calendar_date',
    how='left',
)

# 5. Stratified down-scaling by visit_date (~10,000 samples total)
# Calculate exact sampling fraction (~4% per day) to maintain daily volume ratios
target_total_samples = 10000
sampling_fraction = target_total_samples / len(merged_df)

sampled_df = merged_df.groupby('visit_date', group_keys=False).apply(
    lambda x: x.sample(frac=sampling_fraction, random_state=42)
)

# Sort chronologically by visit_date and air_store_id before splitting
sampled_df.sort_values(
    by=['visit_date', 'air_store_id'], ascending=[True, True], inplace=True
)
sampled_df.reset_index(drop=True, inplace=True)

# 6. Extract essential temporal features
sampled_df['month'] = sampled_df['visit_date'].dt.month
sampled_df['day_of_week_num'] = sampled_df['visit_date'].dt.dayofweek
sampled_df['is_weekend'] = (
    sampled_df['day_of_week_num'].isin([5, 6]).astype(int)
)

# 7. Drop unnecessary/redundant attributes
# Kept columns: air_genre_name, air_area_name, holiday_flg, month, day, day_of_week_num, is_weekend, visitors
columns_to_drop = [
    'air_store_id',
    'visit_date',
    'calendar_date',
    'latitude',
    'longitude',
    'day_of_week',
]
sampled_df.drop(columns=columns_to_drop, inplace=True)

# Reorder columns so target 'visitors' is at the very end (standard convention for Weka)
feature_cols = [col for col in sampled_df.columns if col != 'visitors']
sampled_df = sampled_df[feature_cols + ['visitors']]

# 8. Chronological Train/Test split (80% Train - 20% Test)
split_idx = int(len(sampled_df) * 0.8)
train_df = sampled_df.iloc[:split_idx].reset_index(drop=True)
test_df = sampled_df.iloc[split_idx:].reset_index(drop=True)

# 9. Inspect final results
print(f'Total sampled dataset shape: {sampled_df.shape}')
print(f'Train set shape (80%): {train_df.shape}')
print(f'Test set shape (20%): {test_df.shape}')
print('\nRetained columns:', list(sampled_df.columns))
print('\nFirst 5 rows of Train set:')
print(train_df.head())

# 10. Export datasets
sampled_df.to_csv(
    r'..\data\restaurant_visits.csv', index=False, encoding='utf-8-sig'
)
train_df.to_csv(r'..\data\restaurant_visits_train.csv', index=False, encoding='utf-8-sig')
test_df.to_csv(r'..\data\restaurant_visits_test.csv', index=False, encoding='utf-8-sig')