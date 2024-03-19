import pandas as pd
import matplotlib.pyplot as plt
import ast
import numpy as np

df = pd.read_csv('points.csv')

file = df.iloc[0]['document']
path = 'images/' + file
image = plt.imread(path)

points = ast.literal_eval(df.iloc[0]['points'])

points_vector = np.array([
    [point['x'], point['y']]
    for point in points.values()
])

plt.imshow(image)
plt.scatter(points_vector[:, 0], points_vector[:, 1], c='r', s=10)


plt.show()

