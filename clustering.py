import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering

def cluster_points(points: np.ndarray, method="kmeans"):
    """
    Apply clustering and return centers.
    """
    if len(points) == 0:
        return np.array([])
    k = 10

    if method == "kmeans":
        # Your KMeans clustering code goes here
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(points)
        return kmeans.cluster_centers_

    elif method == "hierarchical":
        # Your hierarchical clustering code goes here
        hc = AgglomerativeClustering(n_clusters=k)
        labels = hc.fit_predict(points)
        centers = []
        for i in range(k):
            points_in_cluster = points[labels == i]


            if len(points_in_cluster) > 0:
                center = np.mean(points_in_cluster, axis=0)
                centers.append(center)
        return np.array(centers)

    else:
        raise ValueError(f"Unknown method: {method}")