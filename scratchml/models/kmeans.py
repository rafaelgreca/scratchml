from ..utils import convert_array_numpy
from ..distances import euclidean
import numpy as np


class KMeans:
    """
    Creates a class for the KMeans model.
    """

    def __init__(
        self,
        n_init: int,
        n_clusters: int = 8,
        max_iter: int = 300,
        tol: float = 0.0001,
        verbose: int = 0,
        n_jobs: int = None,
    ) -> None:
        """
        Creates a KMeans instance.

        Args:
            n_init (int): how many times the KMeans algorithm is run with
                different centroid initializations.
            n_clusters (int, optional): how many clusters that will be formed.
                Defaults to 8.
            tol (float): the tolerance of the difference between two sequential
                lossed, which is used as a stopping criteria.
            max_iter (int, optional): the maximum number of iterations
                during the model training. -1 means that no maximum
                iterations is used. Defaults to -1.
            verbose (int, optional): how much information should be printed.
                Should be 0, 1, or 2. Defaults to 2.
            n_jobs (int, optional): the number of jobs to be used.
                -1 means that all CPUs are used to train the model. Defaults to None.
        """
        self.n_init = n_init
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.labels_ = None
        self.cluster_centers_ = None
        self.inertia_ = 0.0
        self.n_features_in_ = None
        self.n_jobs = n_jobs

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> None:
        """
        Function responsible for training the KMeans model.

        Args:
            X (np.ndarray): the features.
        """
        self._validate_parameters()

        X = convert_array_numpy(X)
        y = convert_array_numpy(y)

        self.n_features_in_ = X.shape[1]
        best_inertia = np.inf

        for _ in range(self.n_init):
            # initializing the clusters centers randomly by choosing random data samples
            random_samples = np.random.choice(
                X.shape[0], self.n_clusters, replace=False
            )
            current_centroids = X[random_samples]
            epoch = 1
            last_centroids = None
            current_labels = None
            current_inertia = 0.0

            while True:
                # stopping criterias
                if last_centroids is not None:
                    # applying the Frobenius norm as a stop criteria to see if the clusters
                    # centers didn't change significativally
                    if (
                        np.linalg.norm(last_centroids - current_centroids, "fro")
                        < self.tol
                    ):
                        break

                if self.max_iter != -1:
                    if epoch > self.max_iter:
                        break

                # assigning the label of the closest cluster for each sample
                current_labels = self._assign_cluster(X, current_centroids)

                # calculating inertia
                current_inertia = np.sum(
                    [
                        np.square(np.linalg.norm(x - current_centroids[y]))
                        for x, y in zip(X, current_labels)
                    ]
                )

                # updating clusters centroid
                last_centroids = current_centroids.copy()
                current_centroids = self._update_centroids(
                    X, current_labels, current_centroids
                )

                epoch += 1

            # getting only the results with best (lowest) inertia
            if current_inertia < best_inertia:
                self.labels_ = current_labels
                self.cluster_centers_ = current_centroids
                self.inertia_ = current_inertia
                best_inertia = current_inertia

    def _assign_cluster(
        self, X: np.ndarray, clusters_centroid: np.ndarray
    ) -> np.ndarray:
        """
        Assign the closest cluster centroid for each data sample.

        Args:
            X (np.ndarray): the features array.
            clusters_centroid (np.ndarray): the clusters' centroid/center.

        Returns:
            labels (np.ndarray): the clusters labels for each sample.
        """
        labels = []

        # detecting the closest cluster center for each sample
        clusters_distance = euclidean(X, clusters_centroid)

        # getting the index of the cluster with the lowest distance
        labels = np.argmin(clusters_distance, axis=1)

        return labels

    def _update_centroids(
        self, X: np.ndarray, labels: np.ndarray, clusters_centroid: np.ndarray
    ) -> np.ndarray:
        """
        Update the clusters' centroid by calculating the mean of the samples
        within each cluster.

        Args:
            X (np.ndarray): the features array.
            labels (np.ndarray): the classes associated for each sample.
            clusters_centroid (np.ndarray): the clusters' centroid/center.

        Returns:
            np.ndarray: the updated clusters' centroid.
        """
        new_clusters_centroid = np.zeros((self.n_clusters, self.n_features_in_))

        for i, _ in enumerate(clusters_centroid):
            _samples_indexes = np.argwhere(labels == i).reshape(-1)

            if X[_samples_indexes].shape[0] > 0:
                new_clusters_centroid[i] = np.mean(X[_samples_indexes], axis=0)
            else:
                new_clusters_centroid[i] = clusters_centroid[i].copy()

        return new_clusters_centroid

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Uses the trained model to predict the classes of a given
        data points (also called features).

        Args:
            X (np.ndarray): the features.

        Returns:
            np.ndarray: the predicted classes.
        """
        X = convert_array_numpy(X)

        # detecting the closest cluster center for each sample
        clusters_distance = euclidean(X, self.cluster_centers_)

        # getting the index of the cluster with the lowest distance
        return np.argmin(clusters_distance, axis=1).reshape(-1)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Trains the model and then uses it to predict the classes
        of a given data points (also called features).

        Args:
            X (np.ndarray): the features.

        Returns:
            np.ndarray: the predicted classes.
        """
        self.fit(X)
        return self.predict(X)

    def score(self) -> np.float64:
        """
        Opposite of the value of X on the K-means objective.

        Returns:
            np.float32: the score achieved by the model.
        """
        return -1 * self.inertia_

    def _validate_parameters(self) -> None:
        """
        Auxiliary function used to validate the values of the parameters
        passed during the initialization.
        """
        # validating the n_init value
        try:
            assert self.n_init is not None
        except AssertionError as error:
            raise ValueError("The 'n_init' value must be defined.\n") from error

        try:
            assert self.n_init > 0
        except AssertionError as error:
            raise ValueError("The 'n_init' must be bigger than zero.\n") from error

        # validating the tol value
        try:
            assert self.tol > 0
        except AssertionError as error:
            raise ValueError("The 'tol' must be bigger than zero.\n") from error

        # validating the n_clusters value
        try:
            assert self.n_clusters > 0
        except AssertionError as error:
            raise ValueError("The 'n_clusters' must be bigger than zero.\n") from error

        # validating the max_iters value
        if self.max_iter < -1 or self.max_iter == 0:
            raise ValueError("Invalid value for 'max_iter'. Must be -1 or >= 1.\n")

        # validating the n_jobs value
        if self.n_jobs is not None:
            try:
                if self.n_jobs < 0:
                    assert self.n_jobs == -1
                else:
                    assert self.n_jobs > 0
            except AssertionError as error:
                raise ValueError(
                    "If not None, 'n_jobs' must be equal to -1 or higher than 0.\n"
                ) from error

        # validating the verbose value
        try:
            assert self.verbose in [0, 1, 2]
        except AssertionError as error:
            raise ValueError(
                "Indalid value for 'verbose'. Must be 0, 1, or 2.\n"
            ) from error
