import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Assuming X is your dataset
X = np.load("sift_descriptors.npy")

wcss = []
k_values = [2**x for x in range(1, 9)]

for k in k_values:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=5,random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(k_values, wcss)
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.show()