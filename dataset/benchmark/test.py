import pandas as pd
import matplotlib.pyplot as plt
import ast

df = pd.read_csv('../../../dataset2/cepha_train.csv', header=None)
print(df)

def aggr_points(row):
    num_points = 19
    points = {}

    landmarks = row.iloc[
        1 : num_points * 2 + 1
    ].values.astype("float")

    landmarks = landmarks.reshape(-1, 2)

    for i in range(num_points):
        points[i] = {
            'x': landmarks[i][0],
            'y': landmarks[i][1]
        }

    row['points'] = points
    row['document'] = row[0]

    return row

df = df.apply(aggr_points, axis=1)

# only keep the columns document and points
df = df[['document', 'points']]

df.to_csv('points.csv', index=False)

print(df)

def show_image(i):
    points = df.iloc[i]['points']
    document = df.iloc[i]['document']
    image = plt.imread(f'images/{document}')
    print(document)
    print(points)
    plt.imshow(image)
    # points have format {0: {x: 100, y: 200}, 1: {x: 300, y: 400}}
    plt.scatter(
        [p['x'] for p in points.values()], [p['y'] for p in points.values()],
        color='red'
    )
    plt.show()
    

