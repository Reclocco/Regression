import numpy as np
import matplotlib.pyplot as plt


def calc_err(x, param, ideal):
    errors = []
    for i_ter in range(215):
        my_sum = 0
        for j_ter in range(len(param)):
            my_sum += x[i_ter][j_ter] * param[j_ter]
        errors.append(ideal[i_ter][0] - my_sum)
    return errors


ratings = np.genfromtxt('ratings.csv', delimiter=',', names=True)
people = []

X_10 = np.zeros((215, 10))
X_1000 = np.zeros((215, 1000))
X_10000 = np.zeros((215, 10000))
y = np.zeros((215, 1))

my_iter = 0
for row in ratings:
    if row['movieId'] == 1:
        people.append(row['userId'])
        y[my_iter][0] = row['rating']
        my_iter += 1

for row in ratings:
    if row['movieId'] <= 10001 and row['movieId'] != 1 and row['userId'] in people:
        X_10000[people.index(row['userId'])][int(row['movieId']) - 2] = row['rating']

X_10 = X_10000[::, :10]
X_1000 = X_10000[::, :1000]

parameters1 = np.linalg.lstsq(X_10, y, rcond=None)[0]
parameters2 = np.linalg.lstsq(X_1000, y, rcond=None)[0]
parameters3 = np.linalg.lstsq(X_10000, y, rcond=None)[0]

# WYPISANIE WARTOŚCI
# print(calc_err(X_10, parameters1, y))
# print(calc_err(X_1000, parameters2, y))
# print(calc_err(X_10000, parameters3, y))

# WYPISANIE WYKRESÓW
plt.plot(calc_err(X_10, parameters1, y))
plt.ylabel('error')
plt.show()

plt.plot(calc_err(X_1000, parameters2, y), )
plt.ylabel('error')
plt.show()

plt.plot(calc_err(X_10000, parameters3, y))
plt.ylabel('error')
plt.show()

training1 = X_10000[:200, :10]
training2 = X_10000[:200, :100]
training3 = X_10000[:200, :200]
training4 = X_10000[:200, :500]
training5 = X_10000[:200, :1000]
training6 = X_10000[:200, :10000]
params_trained1 = np.linalg.lstsq(training1, y[:200], rcond=None)[0]
params_trained2 = np.linalg.lstsq(training2, y[:200], rcond=None)[0]
params_trained3 = np.linalg.lstsq(training3, y[:200], rcond=None)[0]
params_trained4 = np.linalg.lstsq(training4, y[:200], rcond=None)[0]
params_trained5 = np.linalg.lstsq(training5, y[:200], rcond=None)[0]
params_trained6 = np.linalg.lstsq(training6, y[:200], rcond=None)[0]

print(X_10000.shape)

X_100 = X_10000[::, :100]
X_200 = X_10000[::, :200]
X_500 = X_10000[::, :500]

print("m = 10:")
for i in range(15):
    summed = 0
    for j in range(10):
        summed += X_10[200 + i][j] * params_trained1[j]
    print("prawidłowy: ", y[200+i][0], " przewidywany: ", summed)

print("m = 100:")
for i in range(15):
    summed = 0
    for j in range(100):
        summed += X_100[200 + i][j] * params_trained2[j]
    print("prawidłowy: ", y[200+i][0], " przewidywany: ", summed)

print("m = 200:")
for i in range(15):
    summed = 0
    for j in range(200):
        summed += X_200[200 + i][j] * params_trained3[j]
    print("prawidłowy: ", y[200 + i][0], " przewidywany: ", summed)

print("m = 500:")
for i in range(15):
    summed = 0
    for j in range(500):
        summed += X_500[200 + i][j] * params_trained4[j]
    print("prawidłowy: ", y[200 + i][0], " przewidywany: ", summed)

print("m = 1000:")
for i in range(15):
    summed = 0
    for j in range(1000):
        summed += X_1000[200 + i][j] * params_trained5[j]
    print("prawidłowy: ", y[200 + i][0], " przewidywany: ", summed)

print("m = 10000:")
for i in range(15):
    summed = 0
    for j in range(10000):
        summed += X_10000[200 + i][j] * params_trained6[j]
    print("prawidłowy: ", y[200 + i][0], " przewidywany: ", summed)

