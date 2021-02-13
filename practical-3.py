from sklearn.datasets import load_iris

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

classifier = {k: most_fequent_class(v)[0] for k, v in sepal_lengths_widths.items()}
y_hat = [classifier[slw_extractor(fv)] for fv in iris.data]

def get_accuracy(y, y_hat, M) -> float:
    count = 0
    for target, calculated in zip(y, y_hat):
        if target == calculated:
            count += 1
    return count / M

print("1R classifier accuracy: %s" % round(get_accuracy(iris.target, y_hat, M), 2))
print("")