import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# Function to determine the best k using Silhouette Coefficient
# def optimal_k_using_silhouette(data, max_k=10):
#     silhouette_scores = []
#     for k in range(2, max_k + 1):
#         kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
#         labels = kmeans.labels_
#         score = silhouette_score(data, labels)
#         silhouette_scores.append(score)
#         # print(f'k={k}, Silhouette Score={score:.4f}')
#
#     best_k = np.argmax(silhouette_scores) + 2  # Adding 2 because range starts at 2
#     print(f'The optimal number of clusters is {best_k}')
#     return best_k, silhouette_scores


def kmeans_clustering(points, num_clusters):
    # K-Means Clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(points)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    return labels, centroids, num_clusters
