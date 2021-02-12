import csv
import codecs
from random import random
from matplotlib import pyplot as plt
from sklearn import model_selection, datasets, linear_model, metrics
from numpy import insert, array, matmul, matrix

def sqr(x):
    return x * x

def get_axis_range(values, margin=0.02):
    return ((1-margin) * min(values), (1+margin) * max(values))

def starter_plot():
    plt.figure()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

filepath = "data/london-borough-profiles-jan2018.csv"

with codecs.open(filepath, "r", "iso-8859-1") as datafile:
    records = [rec for rec in csv.reader(datafile)]

x = [float(record[70]) for record in records[3:]]
y = [float(record[71]) for record in records[3:]]
del records

starter_plot()
plt.plot(x, y, "bo")
plt.xlabel("Age (men)", fontsize=14)
plt.ylabel("Age (women)", fontsize=14)
plt.title("P2, Q3.1 - Raw Data", fontsize=14)
# plt.savefig("plots/p2-q3-1")

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.10)

starter_plot()
plt.plot(x_train, y_train, "bo", label="Training set")
plt.plot(x_test, y_test, "ro", label="Test set")
plt.legend()
plt.xlim(get_axis_range(x))
plt.ylim(get_axis_range(y))
plt.xlabel("Age (men)", fontsize=14)
plt.ylabel("Age (women)", fontsize=14)
plt.title("P2, Q3.2 - Partitoned Data", fontsize=14)
# plt.savefig("plots/p2-q3-2")


x_syn, y_syn, p = datasets.make_regression(n_samples=100, n_features=1, n_informative=1, noise=10, coef=True)
x_syn_train, x_syn_test, y_syn_train, y_syn_test = model_selection.train_test_split(x_syn, y_syn, test_size=0.10)

starter_plot()
plt.plot(x_syn_train, y_syn_train, "bo", label="Training set")
plt.plot(x_syn_test, y_syn_test, "ro", label="Test set")
plt.legend()
plt.xlim(get_axis_range(x_syn))
plt.ylim(get_axis_range(y_syn))
plt.xlabel("x", fontsize=14)
plt.ylabel("y", fontsize=14)
plt.title("P2, Q3.3 - Partitoned Data", fontsize=14)
# plt.savefig("plots/p2-q3-3")

del x_syn, y_syn, p, x_syn_train, x_syn_test, y_syn_train, y_syn_test


def calculate_forwards(instance, w):
    return w[0] + w[1] * instance

def compute_error(M, x, w, y):
    error = 0
    for j in range( M ):
        y_hat = calculate_forwards(x[j], w)
        error += sqr(y[j] - y_hat)
    return error / M

def gradient_descent_2(M, x, w, y, a):
    for instance, target in zip(x, y):
        y_hat = calculate_forwards(instance, w)
        error = target - y_hat
        w[0] += a * error * 1 * 1/M
        w[1] += a * error * instance * 1/M
    return w

def compute_r2(M, x, w, y):
    y_mean = sum(y) / M
    y_hat = [calculate_forwards(xj, w) for xj in x]
    u = sum([sqr(yj-y_hatj) for yj, y_hatj in zip(y, y_hat)])
    v = sum([sqr(yj-y_mean) for yj in y])
    return round(1 - (u/v), 3)

M, w, a = len(x_train), [random(), random()], 0.0001

for epoch in range(1, 101):
    w = gradient_descent_2(M, x_train, w, y_train, a)

    if epoch in set([2, 3, 4, 100]):
        y_hat = [calculate_forwards(xj, w) for xj in x_train]
        r2 = round(compute_r2(M, x_train, w, y_train), 3)
        starter_plot()
        plt.plot(x_train, y_train, 'bo', markersize=10)
        plt.plot(x_train, y_hat, 'k', linewidth=3)
        plt.xlim(get_axis_range(x_train, margin=0.1))
        plt.ylim(get_axis_range(y_train, margin=0.1))
        plt.title("Epoch %s, R-squared %s" % (epoch, r2), fontsize=14)
        # plt.savefig("plots/p2-q4-epoch" + str(epoch))

del M, w, a, y_hat, r2, epoch


lr = linear_model.LinearRegression()
x_train_array = array(x_train).reshape(-1, 1)
lr.fit(x_train_array, y_train)
y_hat = lr.predict(x_train_array)
r2 = round(metrics.r2_score(y_train, y_hat), 3)
mse = round(metrics.mean_squared_error(y_train, y_hat), 3)

starter_plot()
plt.plot(x_train, y_train, 'bo', markersize=10)
plt.plot(x_train, y_hat, 'k', linewidth=3)
plt.xlim(get_axis_range(x_train))
plt.ylim(get_axis_range(y_train))
plt.title("scikit-learn: r2 %s, mean squared error: %s" % (r2, mse), fontsize=14)
# plt.savefig("plots/p2-q4-scikit-learn")

del lr, x_train_array, y_hat, r2, mse, x_train, x_test, y_train, y_test, x, y