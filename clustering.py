import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering

def kmeans_cluster(frontier_points, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(frontier_points)
    centroids = kmeans.cluster_centers_
    print(f"[KMeans] Found {len(centroids)} observation points for {len(frontier_points)} frontier points.")
    return labels, centroids


def hierarchical_cluster(frontier_points, n_clusters):
    agg_cluster = AgglomerativeClustering(n_clusters=n_clusters)
    labels = agg_cluster.fit_predict(frontier_points)
    centroids = np.array([
        np.mean(frontier_points[labels == i], axis=0)
        for i in np.unique(labels)
    ])
    print(f"[Hierarchical] Found {len(centroids)} observation points for {len(frontier_points)} frontier points.")
    return labels, centroids


def cluster_points(points: np.ndarray, method="kmeans"):
    """
    Apply clustering and return centers.
    """
    if len(points) == 0:
        return np.array([])

    if method == "kmeans":
        # Your KMeans clustering code goes here
        raise ValueError("Implement kmeans frontier clustering.")

    elif method == "hierarchical":
        # Your hierarchical clustering code goes here
        raise ValueError("Implement hierarchical frontier clustering.")