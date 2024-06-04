import numpy as np
from abc import ABC
from scratchml.utils import convert_array_numpy
from scratchml.distances import (
    euclidean,
    minkowski,
    chebyshev,
    manhattan
)
from scratchml.metrics import (
    accuracy,
    recall,
    precision,
    f1_score,
    confusion_matrix,
    mean_squared_error,
    root_mean_squared_error,
    r_squared,
    mean_absolute_error,
    median_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_logarithmic_error,
    max_error
)
from typing import Tuple, Union, List

class BaseKNN(ABC):
    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "uniform",
        p: float = 2,
        metric: str = "minkowski",
        n_jobs: int = None
    ) -> None:
        """
        Creats a K-Nearest Neighbors (KNN) base.

        Args:
            n_neighbors (int, optional): the value for k. Defaults to 5.
            weights (str, optional): how the weights for each
                data point should be initialized. Defaults to "uniform".
            p (float, optional): the power parameter of the Minkowski metric.
                It's only used when the chosen metric is "minkowski". Defaults to 2.
            metric (str, optional): which metric distance should
                be used. Defaults to "minkowski".
            n_jobs (int, optional): the number of jobs to be used.
                -1 means that all CPUs are used to train the model. Defaults to None.
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p
        self.effective_metric_ = metric
        self.n_jobs = n_jobs
        self.classes_ = None
        self.n_features_in_ = None
        self.n_samples_fit_ = None
        self.X_ = None
        self.y_ = None
        self._valid_metrics = [
            "euclidean",
            "chebyshev",
            "manhattan",
            "minkowski"
        ]
        self._valid_weights = [
            "uniform",
            "distance"
        ]
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> None:
        """
        Function responsible for fitting the KNN model. Since this model
        does not requires any fitting, the major calculation will be done
        during the prediction.

        Args:
            X (np.ndarray): the features array.
            y (np.ndarray): the classes array.
        """
        self._validate_parameters()

        X = convert_array_numpy(X)
        y = convert_array_numpy(y)

        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.n_samples_fit_ = X.shape[0]
        self.X_ = X.copy()
        self.y_ = y.copy()
    
    def kneighbors(
        self,
        X: np.ndarray = None,
        n_neighbors: int = None,
        return_distance: bool = True
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        # TODO: Optimize and improve this function
        """
        Gets the K-neighbors of a data point.

        Args:
            X (np.ndarray, optional): the features array. 
                If None, utilizes the data used to fit the model. Defaults to None.
            n_neighbors (int, optional): the number of neighbors.
                If None, utilizes the number of neighbors used in the
                model initialization. Defaults to None.
            return_distance (bool, optional): whether to return the
                distances or not. Defaults to True.

        Returns:
            Tuple[np.ndarray, np.ndarray]: the distances of each neighbor (if
                return_distance = True) and the neighbors indexes, respectively.
        """
        # validating the X value
        if not X is None:
            X = convert_array_numpy(X)
        else:
            X = self.X_
        
        # validating the n_neighbors value
        if n_neighbors != None:
            try:
                assert n_neighbors > 0
            except AssertionError:
                raise ValueError("The number of neighbors must be bigger than zero.\n")
        else:
            n_neighbors = self.n_neighbors
        
        distances = []
        indexes = []

        # calculating the distances between the data points
        if self.effective_metric_ == "euclidean":
            dist = euclidean(X, self.X_)
        elif self.effective_metric_ == "chebyshev":
            dist = chebyshev(X, self.X_)
        elif self.effective_metric_ == "manhattan":
            dist = manhattan(X, self.X_)
        elif self.effective_metric_ == "minkowski":
            dist = minkowski(X, self.X_, self.p)

        # getting the n neighbors for each calculated distance
        for i in range(len(dist)):
            n_indexes = dist[i].argsort()[:n_neighbors]
            indexes.append(n_indexes)
            distances.append(dist[i][n_indexes])
        
        indexes = convert_array_numpy(indexes)
        distances = convert_array_numpy(distances)

        if return_distance:
            return distances, indexes
        else:
            return indexes

    def predict(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Uses the trained model to predict the classes of a given
        data points (also called features).

        Args:
            X (np.ndarray): the features.

        Returns:
            np.ndarray: the predicted classes.
        """
        pass

    def score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metric: str = "accuracy"
    ) -> np.ndarray:
        """
        Uses the trained model to predict the classes of a given
        data points (also called features).

        Args:
            X (np.ndarray): the features array.
            y (np.ndarray): the labels array.
            metric (string): which metric should be used to assess
                the model's performance.
            np.float32: the score achieved by the model.

        Returns:
            np.ndarray: the predicted classes.
        """
        pass

    def _validate_parameters(self) -> None:
        """
        Auxiliary function used to validate the values of the parameters
        passed during the initialization.
        """
        # validating the n_neighbors value
        try:
            assert self.n_neighbors > 0
        except AssertionError:
            raise ValueError("The 'n_neighbors' must be bigger than zero.\n")
        
        # validating the n_jobs value
        if self.n_jobs != None:
            try:
                if self.n_jobs < 0:
                    assert self.n_jobs == -1
                else:
                    assert self.n_jobs > 0
            except AssertionError:
                raise ValueError("If not None, 'n_jobs' must be equal to -1 or higher than 0.\n")
        
        # validating the p value
        try:
            assert self.p > 0
        except AssertionError:
            raise ValueError("The value for 'p' must be a positive number.\n")
        
        # validating the metric value
        try:
            assert self.effective_metric_ in self._valid_metrics
        except AssertionError:
            raise ValueError(f"'Metric' should be {self._valid_metrics}, got {self.effective_metric_} instead.\n")
        
        # validating the weights value
        try:
            assert self.weights in self._valid_weights
        except AssertionError:
            raise ValueError(f"'Weights' should be {self._valid_weights}, got {self.weights} instead.\n")

        if self.p == 2 and self.effective_metric_ == "minkowski":
            self.effective_metric_ = "euclidean"
        
        if self.p == 1 and self.effective_metric_ == "minkowski":
            self.effective_metric_ = "manhattan"

class KNNClassifier(BaseKNN):
    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "uniform",
        p: float = 2,
        metric: str = "minkowski",
        n_jobs: int = None
    ) -> None:
        """
        Creats a K-Nearest Neighbors (KNN) Classifier instance.

        Args:
            n_neighbors (int, optional): the value for k. Defaults to 5.
            weights (str, optional): how the weights for each
                data point should be initialized. Defaults to "uniform".
            p (float, optional): the power parameter of the Minkowski metric.
                It's only used when the chosen metric is "minkowski". Defaults to 2.
            metric (str, optional): which metric distance should
                be used. Defaults to "minkowski".
            n_jobs (int, optional): the number of jobs to be used.
                -1 means that all CPUs are used to train the model. Defaults to None.
        """
        super().__init__(n_neighbors, weights, p, metric, n_jobs)
        self._valid_score_metrics = [
            "accuracy",
            "recall",
            "precision",
            "f1_score",
            "confusion_matrix"
        ]
    
    def predict(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Uses the trained model to predict the classes of a given
        data points (also called features).

        Args:
            X (np.ndarray): the features.

        Returns:
            np.ndarray: the predicted classes.
        """
        X = convert_array_numpy(X)
        prediction = []

        # getting the k closest neighbors
        distances, indexes = self.kneighbors(
            X=X,
            n_neighbors=self.n_neighbors,
            return_distance=True
        )

        indexes = indexes.astype(np.int32)

        for index, dist in zip(indexes, distances):
            _y = self.y_[index]

            # getting the most common classes between the k neighbors
            # for each data point
            if self.weights == "uniform":
                classes_count = np.bincount(_y)
                most_frequent_class = np.argmax(classes_count)
                prediction.append(most_frequent_class)
            elif self.weights == "distance":
                # we devide the distances by the sum of distances for the k
                # neighbors, then we subtract it from 1 so the smallest distances
                # will have a value closer to 1 (higher weights)
                dist /= sum(dist)
                _dist = 1 - dist
                _classes = np.zeros(len(self.classes_))

                # suming the weights for each classes and then getting the class
                # with the highest sum
                for label, weight in zip(_y, _dist):
                    _classes[label] += weight
                
                prediction.append(np.argmax(_classes))
        
        prediction = convert_array_numpy(prediction)
        prediction = prediction.astype(int)

        return prediction

    def score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metric: str = "accuracy",
        labels_cm: List = None,
        normalize_cm: bool = False
    ) -> Union[np.float64, np.ndarray]:
        """
        Uses the trained model to predict the classes of a given
        data points (also called features).

        Args:
            X (np.ndarray): the features array.
            y (np.ndarray): the labels array.
            metric (string): which metric should be used to assess
                the model's performance. Defaults to Accuracy.
            labels_cm (str, optional): which labels should be used to calculate
                the confusion matrix. If other metric is selected, then this
                parameter will be ignore. Defaults to None.
            normalize_cm (bool, optional): whether the confusion matrix should be
                normalized ('all', 'pred', 'true') or not. If other metric is selected,
                then this parameter will be ignore. Defaults to False.

        Returns:
            (np.float32, np.ndarray): the score achieved by the model
                or its confusion matrix.
        """
        try:
            assert metric in self._valid_score_metrics
        except AssertionError:
            raise ValueError(
                f"Invalid value for 'metric'. Must be {self._valid_score_metrics}.\n"
            )
        
        y_hat = self.predict(X)

        if metric == "accuracy":
            return accuracy(y, y_hat)
        elif metric == "precision":
            return precision(y, y_hat)
        elif metric == "recall":
            return recall(y, y_hat)
        elif metric == "f1_score":
            return f1_score(y, y_hat)
        elif metric == "confusion_matrix":
            return confusion_matrix(
                y,
                y_hat,
                labels_cm,
                normalize_cm
            )

class KNNRegressor(BaseKNN):
    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "uniform",
        p: float = 2,
        metric: str = "minkowski",
        n_jobs: int = None
    ) -> None:
        """
        Creats a K-Nearest Neighbors (KNN) Regressor instance.

        Args:
            n_neighbors (int, optional): the value for k. Defaults to 5.
            weights (str, optional): how the weights for each
                data point should be initialized. Defaults to "uniform".
            p (float, optional): the power parameter of the Minkowski metric.
                It's only used when the chosen metric is "minkowski". Defaults to 2.
            metric (str, optional): which metric distance should
                be used. Defaults to "minkowski".
            n_jobs (int, optional): the number of jobs to be used.
                -1 means that all CPUs are used to train the model. Defaults to None.
        """
        super().__init__(n_neighbors, weights, p, metric, n_jobs)
        self._valid_score_metrics = [
            "r_squared",
            "mse",
            "mae",
            "rmse",
            "medae",
            "mape",
            "msle",
            "max_error"
        ]
    
    def predict(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Uses the trained model to predict the classes of a given
        data points (also called features).

        Args:
            X (np.ndarray): the features.

        Returns:
            np.ndarray: the predicted classes.
        """
        X = convert_array_numpy(X)
        prediction = []

        # getting the k closest neighbors
        indexes = self.kneighbors(
            X=X,
            n_neighbors=self.n_neighbors,
            return_distance=False
        )

        indexes = indexes.astype(np.int32)
        _weights = np.arange(1, self.n_neighbors + 1)[::-1]
        
        for index in indexes:
            _y = self.y_[index]

            # getting the mean target value for the k neighbors
            if self.weights == "uniform":
                _mean = np.mean(_y)
            elif self.weights == "distance":
                # assigning a weight (from a range between n_neighbors value
                # and 1), where the closest neighbor will get the highest weight
                # (n_neighbors) and the farthest will receive the lowest (1)
                # the prediction will be the mean of the product between the weights
                # and the targets
                _mean = np.dot(_weights, _y) / sum(_weights)
                
            prediction.append(_mean)
        
        prediction = convert_array_numpy(prediction)
        prediction = prediction.astype(np.float64)

        return prediction

    def score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metric: str = "r_squared"
    ) -> np.float64:
        """
        Uses the trained model to predict the classes of a given
        data points (also called features).

        Args:
            X (np.ndarray): the features array.
            y (np.ndarray): the labels array.
            metric (string): which metric should be used to assess
                the model's performance. Defaults to R Squared.

        Returns:
            np.float32: the score achieved by the model.
        """
        try:
            assert metric in self._valid_score_metrics
        except AssertionError:
            raise ValueError(
                f"Invalid value for 'metric'. Must be {self._valid_score_metrics}.\n"
            )
        
        y_hat = self.predict(X)

        if metric == "r_squared":
            return r_squared(y, y_hat)
        elif metric == "mse":
            return mean_squared_error(y, y_hat)
        elif metric == "mae":
            return mean_absolute_error(y, y_hat)
        elif metric == "rmse":
            return root_mean_squared_error(y, y_hat)
        elif metric == "medae":
            return median_absolute_error(y, y_hat)
        elif metric == "mape":
            return mean_absolute_percentage_error(y, y_hat)
        elif metric == "msle":
            return mean_squared_logarithmic_error(y, y_hat)
        elif metric == "max_error":
            return max_error(y, y_hat)