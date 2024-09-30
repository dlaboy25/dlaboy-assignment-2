import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, init_method='random', max_iter=100, initial_centroids=None):
        self.n_clusters = n_clusters
        self.init_method = init_method
        self.max_iter = max_iter
        self.initial_centroids = initial_centroids
        self.centroids = None
        self.labels = None
        self.history = []

    def fit(self, X):
        self.history = []
        if self.init_method == 'manual':
            if self.initial_centroids is None:
                raise ValueError("Initial centroids must be provided for manual initialization.")
            self.centroids = self.initial_centroids.copy()
        elif self.init_method == 'random':
            self.centroids = self._init_random(X)
        elif self.init_method == 'farthest':
            self.centroids = self._init_farthest(X)
        elif self.init_method == 'kmeans++':
            self.centroids = self._init_kmeans_plus_plus(X)
        else:
            raise ValueError("Invalid initialization method.")

        for _ in range(self.max_iter):
            labels = self._assign_clusters(X)
            new_centroids = self._compute_centroids(X, labels)
            self.history.append((self.centroids.copy(), labels.copy()))
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids
        self.labels = labels

    def _assign_clusters(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _compute_centroids(self, X, labels):
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for idx in range(self.n_clusters):
            points = X[labels == idx]
            if len(points) > 0:
                centroids[idx] = points.mean(axis=0)
            else:
                centroids[idx] = self.centroids[idx]
        return centroids

    def _init_random(self, X):
        indices = np.random.choice(len(X), self.n_clusters, replace=False)
        return X[indices]

    def _init_farthest(self, X):
        centroids = []
        centroids.append(X[np.random.randint(len(X))])
        for _ in range(1, self.n_clusters):
            dist_sq = np.array([min([np.linalg.norm(x - c)**2 for c in centroids]) for x in X])
            next_centroid = X[np.argmax(dist_sq)]
            centroids.append(next_centroid)
        return np.array(centroids)

    def _init_kmeans_plus_plus(self, X):
        centroids = []
        centroids.append(X[np.random.randint(len(X))])
        for _ in range(1, self.n_clusters):
            dist_sq = np.array([min([np.linalg.norm(x - c)**2 for c in centroids]) for x in X])
            probabilities = dist_sq / dist_sq.sum()
            cumulative_probabilities = probabilities.cumsum()
            r = np.random.rand()
            for idx, cumulative_probability in enumerate(cumulative_probabilities):
                if r < cumulative_probability:
                    centroids.append(X[idx])
                    break
        return np.array(centroids)
