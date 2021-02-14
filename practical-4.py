from sklearn.datasets import _samples_generator
from matplotlib import pyplot as plt
from numpy.random import rand
from numpy import array, square, sum, mean, zeros, array_equal

colours = [(0.957, 0.263, 0.212, 1), (0.545, 0.765, 0.029, 0.7), (0.118, 0.533, 0.898, 0.5)]


print("Section 2 - Getting Started")

plt.figure(figsize=[6.4 * 2.5, 4.8])
for i, std in enumerate([1.0, 2.0, 5.0], 1):
    X, _ = _samples_generator.make_blobs(n_samples=1000, n_features=2, cluster_std=std, random_state=1)
    plt.subplot(1, 3, i)
    plt.plot(X[:,0], X[:,1], 'g.')
    plt.title("cluster_std: %s" % std, fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
plt.savefig("plots/practical-4/std-clusters")
plt.close()

del i, std, X
print("\n")


print("Section 3 - Write K-Means Yourself")

def squared_euclidean_distance(a, b) -> float:
    return sum(square(array(a) - array(b)))
    
def kmeans(points, k):
    feature_dimensionality = len(points[0])
    cluster_centroids = None
    new_cluster_centroids = [rand(feature_dimensionality) for _ in range(k)]

    while not array_equal(cluster_centroids, new_cluster_centroids):
        allocation = [[] for _ in range(k)]
        cluster_centroids = new_cluster_centroids

        for point in points:
            closest_cluster = None
            closest_cluster_distance = float("inf")
            for i, centroid in enumerate(cluster_centroids):
                distance = squared_euclidean_distance(point, centroid)
                if distance < closest_cluster_distance:
                    closest_cluster_distance = distance
                    closest_cluster = i
            allocation[closest_cluster].append(point)

        new_cluster_centroids = [None] * k
        for i, cluster_points in enumerate(allocation):
            new_cluster_centroids[i] = mean(cluster_points, axis=0)

    return allocation

X, _ = _samples_generator.make_blobs(n_samples=1000, n_features=2, cluster_std=1.0, random_state=1)
kmeans_clusters = kmeans(X, 2)

plt.figure()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
for i, cluster_points in enumerate(kmeans_clusters):
    cluster_points = array(cluster_points)
    plt.plot(cluster_points[:,0], cluster_points[:,1], "bo", color=colours[i])
del i, cluster_points
plt.savefig("plots/practical-4/kmeans-by-hand")
plt.close()

del X, kmeans_clusters
print("\n")