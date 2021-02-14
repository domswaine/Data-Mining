from sklearn.datasets import _samples_generator
from matplotlib import pyplot as plt

print("Section 2 - Getting Started")

plt.figure(figsize=[6.4 * 2.5, 4.8])

for i, std in enumerate([1.0, 2.0, 5.0], 1):
    X, clusters = _samples_generator.make_blobs(n_samples=1000, n_features=2, cluster_std=std, random_state=1)
    plt.subplot(1, 3, i)
    plt.plot(X[:,0], X[:,1], 'g.')
    plt.title("cluster_std: %s" % std, fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

plt.savefig("plots/practical-4/std-clusters")
plt.close()

del i, std, X, clusters
print("\n")