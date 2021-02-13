from os import system
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree, metrics

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
    return "setosa: %s, versicolor: %s, virginica: %s" % (c[0], c[1], c[2])

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
print("")


print("Section 4")

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)
clf = tree.DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)
y_hat = clf.predict(X_test)

M_test = len(y_test)
count = get_count(y_test, y_hat)
accuracy = (count / M_test) * 100

print("Number of correct predictions: %d out of %d = %f%%" % (count, M_test, accuracy))
del count, M_test, accuracy

training_score = clf.score(X_train, y_train) * 100
test_score = clf.score(X_test, y_test) * 100

print("Training score: %f%%" % training_score)
print("Test score: %f%%" % test_score)
del training_score, test_score

sklearn_accuracy = metrics.accuracy_score(y_test, y_hat)
print("Accuracy (sk-learn): %f%%" % sklearn_accuracy)
del sklearn_accuracy

precision = metrics.precision_score(y_test, y_hat, average=None)
print("Precision: %s" % get_counts(precision))
del precision

recall = metrics.recall_score(y_test, y_hat, average=None)
print("Recall: %s" % get_counts(recall))
del recall

f1 = metrics.f1_score(y_test, y_hat, average=None)
print("F1: %s" % get_counts(f1))
del f1

# print(clf.decision_path(iris.data))
# tree.export_graphviz(clf, out_file="plots/p3-iris-decision-tree.dot", class_names=classes, impurity=True)
# system("sudo apt install graphviz")
# system("dot plots/p3-iris-decision-tree.dot -Tpng > plots/p3-iris-decision-tree.png")

del X_test, X_train, clf, y_hat, y_test, y_train
