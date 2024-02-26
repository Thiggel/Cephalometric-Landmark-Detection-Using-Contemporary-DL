import pandas as pd

df = pd.read_csv('all_images_same_points.csv')

print(df['image_recorded_at'][:5])

df['image_recorded_at'] = pd.to_datetime(df['image_recorded_at'])

print(df['image_recorded_at'][:5])

start_date = pd.to_datetime('2023-03-01')
end_date = pd.to_datetime('2023-05-01')

mask = (df['image_recorded_at'] > start_date) & (df['image_recorded_at'] <= end_date)

print(df[mask]['image_recorded_at'])
