import csv
import codecs
from matplotlib import pyplot as plt
from sklearn import model_selection, datasets
from numpy import insert, array, matmul

def get_axis_range(values):
    return (0.98 * min(values), 1.02 * max(values))

def starter_plot():
    plt.figure()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

filepath = "data/london-borough-profiles-jan2018.csv"

with codecs.open(filepath, "r", "iso-8859-1") as datafile:
    records = [rec for rec in csv.reader(datafile)]

x = [float(record[70]) for record in records[3:]]
y = [float(record[71]) for record in records[3:]]

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

del x_syn, y_syn, p, x_train, x_syn_test, y_syn_train, y_syn_test