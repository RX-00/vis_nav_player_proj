import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def tune_kmeans_elbow(X, n_clusters_range):
    """
    Tune the number of clusters for KMeans using the Elbow method.
    
    Parameters:
    - X: array-like, shape (n_samples, n_features)
        The input data to cluster.
    - n_clusters_range: list or range
        A list or range of cluster numbers to evaluate.
    
    Returns:
    - wcss_values: list
        The WCSS values for each number of clusters.
    - best_n_clusters: int
        The number of clusters that might be optimal based on the Elbow method.
    """
    wcss_values = []

    # Loop over different values of n_clusters
    for n_clusters in n_clusters_range:
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
        kmeans.fit(X)
        wcss_values.append(kmeans.inertia_)  # Inertia is the WCSS value.

        print(f'n_clusters: {n_clusters}, WCSS: {kmeans.inertia_:.2f}')

    # Plot the elbow graph
    plt.figure(figsize=(8, 5))
    plt.plot(n_clusters_range, wcss_values, marker='o', linestyle='-')
    plt.xlabel('Number of Clusters (n_clusters)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.title('Elbow Method for Optimal Number of Clusters')
    plt.grid(True)
    plt.show()
    plt.savefig("kmeans-tune.png")

    # Find the elbow point by looking for the "knee" in the WCSS curve
    best_n_clusters = find_elbow_point(n_clusters_range, wcss_values)
    
    print(f"\nOptimal number of clusters (based on the Elbow method): {best_n_clusters}")
    return wcss_values, best_n_clusters



def find_elbow_point(n_clusters_range, wcss_values):
    """
    A simple heuristic to find the 'elbow' point.
    This uses the point with the maximum second derivative (rate of change).
    """
    # Calculate the first and second derivatives (approximations)
    first_derivative = np.diff(wcss_values)
    second_derivative = np.diff(first_derivative)

    # Find the index of the maximum second derivative
    elbow_index = np.argmax(second_derivative) + 2  # +2 to align with n_clusters

    return n_clusters_range[elbow_index]


X = np.load("sift_descriptors.npy")


n_clusters_range = [2**n for n in range(1, 10)]  # This will give [2, 4, 8]

wcss_values, best_n_clusters = tune_kmeans_elbow(X, n_clusters_range)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate synthetic data
X, _ = make_blobs(n_samples=300, centers=5, cluster_std=0.60, random_state=0)

# Elbow Method
inertia = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Plotting the Elbow Method
plt.figure(figsize=(10, 6))
plt.plot(k_values, inertia, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.xticks(k_values)
plt.grid()
plt.show()
