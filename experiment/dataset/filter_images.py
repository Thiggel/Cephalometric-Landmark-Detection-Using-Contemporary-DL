import pandas as pd

df = pd.read_csv('all_images.csv')

# Split the points into individual identifiers
df['points'] = df['points'].str.split('; ')

# Parse the points
df['points'] = df['points'].apply(lambda x: {
    entry.split(':')[0]: {
        'x': int(entry.split(':')[1].split(',')[0]),
        'y': int(entry.split(':')[1].split(',')[1])
    } for entry in x
})

all_identifiers = set()
for points_dict in df['points']:
    all_identifiers.update(points_dict.keys())


def fill_missing_identifiers(points):
    for identifier in all_identifiers:
        if identifier not in points:
            points[identifier] = {'x': -1, 'y': -1}
    return points

df['points'] = df['points'].apply(fill_missing_identifiers)

# Sort Points alphabetically
df['points'] = df['points'].apply(lambda x: {
    k: v for k, v in sorted(x.items(), key=lambda item: item[0])
})

df['num_points'] = df['points'].apply(len)

print(f'''
      Sanity Check: there are these lengths of
      points in the dataframe:
      {df["num_points"].unique()}

      there are {df.shape[0]} images in the dataframe
''')

df.to_csv('filtered_data.csv', index=False)
# maybe pickle the data? what is the most efficient way lateron?
