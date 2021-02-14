from os import system
from matplotlib import pyplot
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree, metrics, linear_model, preprocessing
from numpy import arange, meshgrid, array

print("Section 2", "\n")

iris = load_iris()
classes = iris.target_names
attributes = iris.feature_names
M = len(iris.data)

print("Classes: %s" % iris.target_names)
print("Attributes: %s" % iris.feature_names)
print("Number of instances: %s" % M)
print("\n")


print("Section 3", "\n")

def get_counts(c):
    return "setosa: %s, versicolor: %s, virginica: %s" % (round(c[0], 2), round(c[1], 2), round(c[2], 2))

def count_on_attribute(X, y, no_classes, features_extractor):
    class_counts = {}
    for feature_vector, classification in zip(X, y):
        key = features_extractor(feature_vector)
        if key not in class_counts:
            class_counts[key] = [0] * no_classes
        class_counts[key][classification] += 1
    return class_counts

def most_fequent_class(c):
    greatest_count, most_frequent_class = 0, []
    for i, count in enumerate(c):
        if count > greatest_count:
            greatest_count = count
            most_frequent_class = [i]
        elif count == greatest_count:
            most_frequent_class.append(i)
    return most_frequent_class


print("Sepal length")
sepal_lengths = count_on_attribute(iris.data, iris.target, 3, lambda x: x[0])
for sepal_length, class_counts in sepal_lengths.items():
    print("> %scm - %s" % (sepal_length, get_counts(class_counts)))
print("")

print("Sepal width")
sepal_widths = count_on_attribute(iris.data, iris.target, 3, lambda x: x[1])
for sepal_width, class_counts in sepal_widths.items():
    print("> %scm - %s" % (sepal_width, get_counts(class_counts)))
print("")

slw_extractor = lambda x: (x[0], x[1])

print("Sepal lengths and widths")
sepal_lengths_widths = count_on_attribute(iris.data, iris.target, 3, slw_extractor)
for (length, width), class_counts in sepal_lengths_widths.items():
    most_freq = most_fequent_class(class_counts)
    print("Sepal length: %scm, sepal width: %scm => %s" % (length, width, most_freq))
print("")

clf = {k: most_fequent_class(v)[0] for k, v in sepal_lengths_widths.items()}
y_hat = [clf[slw_extractor(fv)] for fv in iris.data]

def get_count(y, y_hat) -> int:
    count = 0
    for target, calculated in zip(y, y_hat):
        if target == calculated:
            count += 1
    return count

def get_accuracy(y, y_hat, M) -> float:
    return get_count(y, y_hat) / M

print("1R classifier accuracy: %s" % round(get_accuracy(iris.target, y_hat, M), 2))

del M, sepal_lengths, sepal_length, class_counts, sepal_width, sepal_widths
del slw_extractor, sepal_lengths_widths, length, width, clf, y_hat
print("\n")


print("Section 4", "\n")

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)
clf = tree.DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)
y_hat = clf.predict(X_test)

M_test = len(y_test)
count = get_count(y_test, y_hat)
accuracy = (count / M_test) * 100

print("Number of correct predictions: %d out of %d = %f%%" % (count, M_test, accuracy))
del count, M_test, accuracy

def get_training_score():
    training_score = clf.score(X_train, y_train) * 100
    print("Training score: %d%%" % round(training_score, 2))

def get_test_score():
    test_score = clf.score(X_test, y_test) * 100
    print("Test score: %d%%" % round(test_score, 2))

def get_accuracy():
    sklearn_accuracy = metrics.accuracy_score(y_test, y_hat)
    print("Accuracy (sk-learn): %f%%" % round(sklearn_accuracy, 2))

def get_precision():
    precision = metrics.precision_score(y_test, y_hat, average=None)
    print("Precision: %s" % get_counts(precision))

def get_recall():
    recall = metrics.recall_score(y_test, y_hat, average=None)
    print("Recall: %s" % get_counts(recall))

def get_f1():
    f1 = metrics.f1_score(y_test, y_hat, average=None)
    print("F1: %s" % get_counts(f1))

get_training_score()
get_test_score()
get_accuracy()
get_precision()
get_recall()
get_f1()

# print(clf.decision_path(iris.data))
# tree.export_graphviz(clf, out_file="plots/p3-iris-decision-tree.dot", class_names=classes, impurity=True)
# system("sudo apt install graphviz")
# system("dot plots/p3-iris-decision-tree.dot -Tpng > plots/p3-iris-decision-tree.png")

del X_test, X_train, clf, y_hat, y_test, y_train
print("\n")


print("Section 5", "\n")

X, y = iris.data[:, :2], iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
clf = linear_model.LogisticRegression(solver="lbfgs", multi_class="multinomial")
clf.fit(X_train, y_train)
y_hat = clf.predict(X_test)

get_training_score()
get_test_score()
get_accuracy()
get_precision()
get_recall()
get_f1()

pyplot.figure()
pyplot.xticks(fontsize=14)
pyplot.yticks(fontsize=14)
pyplot.set_cmap("Blues")
pyplot.xlabel(attributes[0], fontsize=14)
pyplot.ylabel(attributes[1], fontsize=14)

interval = 0.1
x0_range = arange(min(X[:,0]) - interval, max(X[:,0]) + interval, interval)
x1_range = arange(min(X[:,1]) - interval, max(X[:,1]) + interval, interval)
X_pairs = array([[x0, x1] for x1 in x1_range for x0 in x0_range])
y_hat_pairs = clf.predict(X_pairs)
x0_mesh, x1_mesh = meshgrid(x0_range, x1_range)
y_hat_mesh = y_hat_pairs.reshape(x0_mesh.shape)
pyplot.pcolormesh(x0_mesh, x1_mesh, y_hat_mesh, shading="auto")

markers = ['o','<','s']
colours = [(0.957, 0.263, 0.212, 1), (0.545, 0.765, 0.029, 0.7), (0.118, 0.533, 0.898, 0.5)]

for Xi, yi in zip(X, y):
    pyplot.plot(Xi[0], Xi[1], marker=markers[yi], markerfacecolor=colours[yi], markersize=9, markeredgecolor='w')

pyplot.savefig("plots/p3-q4-logistic-regression-decision-bounaries")

pyplot.figure()
pyplot.xticks(fontsize=14)
pyplot.yticks(fontsize=14)
pyplot.xlabel("False postive rate", fontsize=14)
pyplot.ylabel("True postive rate", fontsize=14)

conf_scores = clf.decision_function(X_pairs)
y_binary = preprocessing.label_binarize(y_hat_pairs, classes=sorted(set(y)))
for c in range(3):
    fpr, tpr, _ = metrics.roc_curve(y_binary[:, c], conf_scores[:, c])
    pyplot.plot(fpr, tpr, label=classes[c], color=colours[c])

pyplot.legend()
pyplot.savefig("plots/p3-q4-logistic-regression-roc-curve")

del X, y, X_train, X_test, y_train, y_test, clf, y_hat
del interval, x0_range, x1_range, X_pairs, y_hat_pairs, x0_mesh, x1_mesh, y_hat_mesh
del markers, colours, Xi, yi, conf_scores, y_binary, c, fpr, tpr
print("\n")
