import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# Function to determine the best k using Silhouette Coefficient
def optimal_k_using_silhouette(data, max_k=10):
    silhouette_scores = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
        labels = kmeans.labels_
        score = silhouette_score(data, labels)
        silhouette_scores.append(score)
        print(f'k={k}, Silhouette Score={score:.4f}')

    best_k = np.argmax(silhouette_scores) + 2  # Adding 2 because range starts at 2
    print(f'The optimal number of clusters is {best_k}')
    return best_k, silhouette_scores


def kmeans_clustering(points):
    # K-Means Clustering
    num_clusters, silhouette_scores = optimal_k_using_silhouette(points, max_k=10)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(points)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Visualize the clustered points
    # plt.figure(figsize=(8, 6))
    # colors = ['red', 'green', 'blue', 'purple', 'orange']
    # for i in range(num_clusters):
    #     cluster_points = points[labels == i]
    #     plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i], marker='o', label=f'Cluster {i + 1}')
    #     plt.scatter(centroids[i, 0], centroids[i, 1], color='black', marker='x', s=100, linewidths=3)
    #
    # plt.title('K-Means Clustering of Random Points')
    # plt.xlabel('X Coordinate')
    # plt.ylabel('Y Coordinate')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # Print points and their assigned clusters
    #for i, (point, label) in enumerate(zip(points, labels)):
    #   print(f'Point {i}: ({point[0]:.2f}, {point[1]:.2f}) - Cluster {label + 1}')
