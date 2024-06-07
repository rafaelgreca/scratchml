import numpy as np
from scratchml.metrics import accuracy, recall, precision, f1_score, confusion_matrix
from scratchml.utils import convert_array_numpy
from typing import Union, List


class GaussianNB(object):
    def __init__(self, priors: np.ndarray = None, var_smoothing: float = 1e-09) -> None:
        """
        Creates an instance of the Gaussian Naive Bayes model.

        Args:
            priors (np.ndarray, optional): prior probabilities of the classes.
                If specified, the priors are not adjusted. Defaults to None.
            var_smoothing (float, optional): a very small value that will be
                added to variances for calculation stability. Defaults to 1e-09.
        """
        self.class_count_ = None
        self.class_prior_ = priors
        self.classes_ = None
        self.epsilon_ = var_smoothing
        self.n_features_in_ = None
        self.var_ = None
        self.theta_ = None
        self._valid_metrics = [
            "accuracy",
            "recall",
            "precision",
            "f1_score",
            "confusion_matrix",
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Function responsible for fitting the Gaussian Naive Bayes.

        Args:
            X (np.ndarray): the features array.
            y (np.ndarray): the labels array.
        """
        X = convert_array_numpy(X)
        y = convert_array_numpy(y)

        # validating the var smoothing value
        try:
            assert self.epsilon_ > 0
        except AssertionError:
            raise ValueError("The var smoothing value should be bigger than 0.\n")

        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.class_count_ = []
        n_classes = len(self.classes_)
        calculate_prior = False
        self.var_ = np.ndarray((n_classes, self.n_features_in_))
        self.theta_ = np.ndarray((n_classes, self.n_features_in_))

        if self.class_prior_ is not None:
            try:
                assert np.sum(self.class_prior_) == 1
            except AssertionError:
                raise ValueError("The sum of the priors should be 1.\n")

            self.class_prior_ = convert_array_numpy(self.class_prior_)
        else:
            self.class_prior_ = np.ndarray(n_classes)
            calculate_prior = True

        for c in range(n_classes):
            indexes = np.where(y == c)[0]
            _X = X[indexes].copy()

            if calculate_prior:
                self.class_prior_[c] = indexes.shape[0] / y.shape[0]

            self.theta_[c, :] = _X.mean(axis=0)
            self.var_[c, :] = _X.var(axis=0)
            self.class_count_.append(indexes.shape[0])

        self.class_count_ = convert_array_numpy(self.class_count_)

    def predict(self, X: np.ndarray, output_format: str = "class") -> np.ndarray:
        """
        Uses the model to predict the classes of a given set (also called features).

        Args:
            X (np.ndarray): the features array.
            output_format (str): how the output should be return ("class", "log",
                "proba"). "Class" will return the class with the highest likelihood,
                "log" will return the log of the likelihood, and "proba" will return
                the classes probabilities based on the likelihood. Defaults to "class".

        Returns:
            np.ndarray: the predicted classes.
        """
        X = convert_array_numpy(X)
        predictions = []

        # validating the output format
        try:
            assert output_format in ["class", "proba", "log"]
        except AssertionError:
            raise ValueError("Output format should be 'class', 'proba', or 'log'.\n")

        # TODO: Optimize this function (think how to do the same thing
        # using matrix multiplication)
        for i in range(X.shape[0]):
            _X = X[i, :].copy()
            _posteriors = []

            # calculates the likelihood (posterior) for each class
            for idx, _ in enumerate(self.classes_):
                _posterior = np.prod(self._gaussian(_X, idx)) * self.class_prior_[idx]
                _posteriors.append(_posterior)

            if output_format == "class":
                # gets the class with the highest likelihood
                predictions.append(self.classes_[np.argmax(_posteriors)])
            elif output_format == "log":
                # gets the logs of the likelihoods
                predictions.append(np.log(_posteriors / np.sum(_posteriors)))
            elif output_format == "proba":
                # gets the classes probabilities
                predictions.append(_posteriors / np.sum(_posteriors))

        predictions = convert_array_numpy(predictions)
        return predictions

    def predict_proba(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Uses the model to predict the classes probabilities of a
        given set (also called features).

        Args:
            X (np.ndarray): the features array.

        Returns:
            np.ndarray: the predicted classes.
        """
        X = convert_array_numpy(X)
        predictions = self.predict(X, "proba")
        return predictions

    def predict_log_proba(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Uses the model to predict the classes log probabilities of a
        given set (also called features).

        Args:
            X (np.ndarray): the features array.

        Returns:
            np.ndarray: the predicted classes.
        """
        X = convert_array_numpy(X)
        predictions = self.predict(X, "log")
        return predictions

    def score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metric: str = "accuracy",
        labels_cm: List = None,
        normalize_cm: bool = False,
    ) -> Union[np.float32, np.ndarray]:
        """
        Calculates the score of the model on a given set for a
        determined metric.

        Args:
            X (np.ndarray): the features.
            y (np.ndarray): the targets of the features.
            threshold (float): the threshold of the prediction. Defaults to 0.5.
            metric (str, optional): which metric to use. Defaults to "r_squared".
            labels_cm (str, optional): which labels should be used to calculate
                the confusion matrix. If other metric is selected, then this
                parameter will be ignore. Defaults to None.
            normalize_cm (bool, optional): whether the confusion matrix should be
                normalized ('all', 'pred', 'true') or not. If other metric is selected,
                then this parameter will be ignore. Defaults to False.

        Returns:
            np.float32: the score achieved by the model.
        """
        try:
            assert metric in self._valid_metrics
        except AssertionError:
            raise ValueError(
                f"Invalid value for 'metric'. Must be {self._valid_metrics}.\n"
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
            return confusion_matrix(y, y_hat, labels_cm, normalize_cm)

    def _gaussian(self, X: np.ndarray, index_class: int) -> np.ndarray:
        """
        Applies the Guassian distribution formula (likelihood).

        Args:
            X (np.ndarray): the features array.
            index_class (int): the index of the class which is being used.

        Returns:
            np.ndarray: the likelihood for this feature for the given class.
        """
        _theta = self.theta_[index_class, :].copy()  # mean
        _var = self.var_[index_class, :].copy()  # variance

        # likelihood/gaussian distribution equation
        numerator = np.exp(-(np.square(X - _theta) / (2 * _var)) + self.epsilon_)
        denominator = np.sqrt(2 * np.pi * _var)
        return numerator / denominator
