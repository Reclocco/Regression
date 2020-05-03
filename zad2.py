import pandas as pd
import numpy as np


df = pd.read_csv('ratings.csv')

df = df.to_numpy()
full = np.zeros((611, 9019))
for i in range(100836):
    if df[i, 1] <= 10000:
        full[int(df[i, 0]), int(df[i, 1])] = df[i, 2]

full = full/np.linalg.norm(full, axis=0)
full = np.nan_to_num(full)

my_ratings = np.zeros((9019, 1))
my_ratings[2571] = 5      # patrz movies.csv  2571 - Matrix
my_ratings[32] = 4        # 32 - Twelve Monkeys
my_ratings[260] = 5       # 260 - Star Wars IV
my_ratings[1097] = 4
my_ratings = my_ratings/np.linalg.norm(my_ratings)

cosinus = np.dot(full, my_ratings)
cosinus = cosinus/np.linalg.norm(cosinus)

results = np.dot(full.T, cosinus)
results = results.reshape(9019)

movies = pd.read_csv('movies.csv')
movies = movies.to_numpy()

blob = np.ones((9019, 2), dtype='<U32')
blob[:, 0] = results

for i in range(5401):
    idx = int(movies[i, 0])
    name = movies[i, 1]
    blob[idx, 1] = name

print("BLOB: ", blob[blob[:, 0].argsort()])
