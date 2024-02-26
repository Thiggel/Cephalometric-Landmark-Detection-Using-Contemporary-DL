import pandas as pd
import ast

df = pd.read_csv(
    'all_images_same_points.csv',
    converters={
        'points': ast.literal_eval
    }
)

df['image_recorded_at'] = pd.to_datetime(df['image_recorded_at'])

day_image_xray_change = pd.to_datetime('2023-03-29')

old_width_px = 1360
old_height_px = 1840

new_width_px = 1800
new_height_px = 2137

px_per_m = 7_756


def process_points(row):
    if row['image_recorded_at'] > day_image_xray_change:
        points_dict = row['points']
        new_points_dict = {}

        for key, nested_dict in points_dict.items():
            new_points_dict[key] = {
                'x': nested_dict['x'] / new_width_px * old_width_px,
                'y': nested_dict['y'] / new_height_px * old_height_px
            }

        row['points'] = new_points_dict

    return row


df = df.apply(process_points, axis=1)

df.to_csv('all_images_same_points_dimensions.csv', index=False)
