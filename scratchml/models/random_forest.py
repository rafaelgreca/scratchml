from typing import Union, List, Tuple
from abc import ABC
from .decision_tree import DecisionTreeClassifier, DecisionTreeRegressor
from ..utils import convert_array_numpy
from ..metrics import (
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
    max_error,
)
import numpy as np


class RandomForestBase(ABC):
    """
    Creates a base class for the Random Forest model.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        bootstrap: bool = True,
        criterion: str = "gini",
        max_depth: int = None,
        max_samples: Union[int, float] = None,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        max_features: Union[int, float, str] = None,
        max_leaf_nodes: int = None,
        min_impurity_decrease: Union[int, float] = 0.0,
        verbose: int = 0,
    ) -> None:
        """
        Creates a Random Forest Base.

        Args:
            n_estimators (str, optional): The number of trees in the forest. Defaults to 100.
            bootstrap (bool, optional): Whether bootstrap samples are used when building trees.
                If False, the whole dataset is used to build each tree. Defaults to True.
            criterion (str, optional): The function to measure the quality of a split.
                Defaults to "gini".
            max_depth (int, optional): The maximum depth of the tree. Defaults to None.
            max_samples (Union[int, float, optional): If bootstrap is True, the number of samples
                to draw from X to train each base estimator. Defaults to None.
            min_samples_split (Union[int, float, optional): The minimum number of samples required
                to split an internal node. Defaults to 2.
            min_samples_leaf (Union[int, float], optional): The minimum number of samples required
                to be at a leaf node. Defaults to 1.
            max_features (Union[int, float, str], optional): The number of features to consider when
                looking for the best split. Defaults to None.
            max_leaf_nodes (int, optional): Grow a tree with max_leaf_nodes in best-first fashion.
                Defaults to None.
            min_impurity_decrease (Union[int, float], optional): A node will be split if this split
                induces a decrease of the impurity greater than or equal to this value.
                Defaults to 0.0.
            verbose (int, optional): how much information should be printed.
                Should be 0, 1, or 2. Defaults to 0.
        """
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.bootstrap = bootstrap
        self.max_depth = max_depth
        self.max_samples = max_samples
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.verbose = verbose
        self.max_features_ = None
        self.classes_ = None
        self.n_classes_ = None
        self.n_features_in_ = None
        self.n_samples_ = None
        self.estimator_ = None
        self.estimators_ = []
        self.estimators_samples_ = []

        # setting the valid criterions for the random forest
        # classifier and for the regressor
        if isinstance(self, RandomForestClassifier):
            self._valid_criterions = [
                "gini",
                "entropy",
                "log_loss",
            ]
            self.estimator_ = DecisionTreeClassifier(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                max_leaf_nodes=self.max_leaf_nodes,
                min_impurity_decrease=self.min_impurity_decrease,
            )
        elif isinstance(self, RandomForestRegressor):
            self._valid_criterions = ["squared_error", "poisson", "absolute_error"]
            self.estimator_ = DecisionTreeRegressor(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                max_leaf_nodes=self.max_leaf_nodes,
                min_impurity_decrease=self.min_impurity_decrease,
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Function responsible for fitting the Decision Tree model.

        Args:
            X (np.ndarray): the features array.
            y (np.ndarray): the classes array.
        """
        self._validate_parameters()

        X = convert_array_numpy(X)
        y = convert_array_numpy(y)

        self.n_samples_, self.n_features_in_ = X.shape
        self.classes_ = np.unique(y)
        self.n_classes_ = len(np.unique(y))

        # setting the max_samples in case bootstrap is enabled
        if self.bootstrap:
            if self.max_samples is None:
                self.max_samples = self.n_samples_
            else:
                if isinstance(self.max_samples, float):
                    self.max_samples = max(round(self.n_samples_ * self.max_samples), 1)

        for _ in range(self.n_estimators):
            # generating a new sub sample that will be used
            # to train the current decision tree model
            if self.bootstrap:
                _X, _y = self._generate_dataset(X=X, y=y)
            else:
                _X = X.copy()
                _y = y.copy()

            dt = None

            # creating a new decision tree and training it
            # with the sub samples (or the whole dataset if
            # bootstrap is used)
            if isinstance(self, RandomForestClassifier):
                dt = DecisionTreeClassifier(
                    criterion=self.criterion,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    max_features=self.max_features,
                    max_leaf_nodes=self.max_leaf_nodes,
                    min_impurity_decrease=self.min_impurity_decrease,
                )
            elif isinstance(self, RandomForestRegressor):
                dt = DecisionTreeRegressor(
                    criterion=self.criterion,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    max_features=self.max_features,
                    max_leaf_nodes=self.max_leaf_nodes,
                    min_impurity_decrease=self.min_impurity_decrease,
                )

            dt.fit(X=_X, y=_y)
            self.estimators_.append(dt)
            self.estimators_samples_.append(_X)

        if self.verbose != 0:
            if isinstance(self, RandomForestClassifier):
                metric_msg = f"Score (Accuracy): {self.score(X, y)}"
                print(f"{metric_msg}\n")
            elif isinstance(self, RandomForestRegressor):
                metric_msg = f"Score (R²): {self.score(X, y)}"
                print(f"{metric_msg}\n")

    def _generate_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Auxiliary function to generate a new bootstrap dataset.

        Args:
            X (np.ndarray): the features array.
            y (np.ndarray): the target array.

        Returns:
            Tuple[np.ndarray, np.ndarray]: the new bootstrap dataset
                features and targets, respectively.
        """
        # randomly selecting samples within the dataset
        new_dataset_indexes = np.random.choice(
            self.n_samples_, self.max_samples, replace=True
        )
        return X[new_dataset_indexes, :], y[new_dataset_indexes]

    def score(
        self, X: np.ndarray, y: np.ndarray, metric: str = "accuracy", **kwargs
    ) -> Union[np.float64, np.ndarray]:
        """
        Uses the trained model to predict the classes of a given
        data points (also called features).

        Args:
            X (np.ndarray): the features array.
            y (np.ndarray): the labels array.
            metric (string): which metric should be used to assess
                the model's performance. Defaults to Accuracy.

        Returns:
            (np.float32, np.ndarray): the score achieved by the model
                or its confusion matrix.
        """

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Uses the trained model to predict the classes of a given
        data points (also called features).

        Args:
            X (np.ndarray): the features.

        Returns:
            np.ndarray: the predicted classes.
        """

    def _validate_parameters(self) -> None:
        """
        Auxiliary function used to validate the values of the parameters
        passed during the initialization.
        """
        # validating the bootstrap value
        try:
            assert isinstance(self.bootstrap, bool)
        except AssertionError as error:
            raise ValueError(
                "The value for 'bootstrap' must be a boolean value.\n"
            ) from error

        # validating the max_samples value
        if self.max_samples is not None:
            if isinstance(self.max_samples, int):
                try:
                    assert (self.max_samples > 0) and (
                        self.max_samples <= self.n_samples_
                    )
                except AssertionError as error:
                    raise ValueError(
                        "The value for 'max_samples' must be a positive number "
                        + f"smaller than {self.n_samples_}, inclusive.\n"
                    ) from error
            elif isinstance(self.max_samples, float):
                try:
                    assert (self.max_samples > 0) and (self.max_samples <= 1)
                except AssertionError as error:
                    raise ValueError(
                        "The value for 'max_samples' must be between 0 and 1 (inclusive).\n"
                    ) from error

        # validating the n_estimators value
        try:
            assert self.n_estimators > 0 and isinstance(self.n_estimators, int)
        except AssertionError as error:
            raise ValueError(
                "The value for 'n_estimators' must be a positive number.\n"
            ) from error

        # validating the verbose value
        try:
            assert self.verbose in [0, 1, 2]
        except AssertionError as error:
            raise ValueError(
                "Indalid value for 'verbose'. Must be 0, 1, or 2.\n"
            ) from error


class RandomForestClassifier(RandomForestBase):
    """
    Creates a class for the Random Forest Classifier model.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        bootstrap: bool = True,
        criterion: str = "gini",
        max_depth: int = None,
        max_samples: Union[int, float] = None,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        max_features: Union[int, float, str] = None,
        max_leaf_nodes: int = None,
        min_impurity_decrease: Union[int, float] = 0.0,
        verbose: int = 0,
    ) -> None:
        """
        Creates a Random Forest Classifier instance.

        Args:
            n_estimators (str, optional): The number of trees in the forest. Defaults to 100.
            bootstrap (bool, optional): Whether bootstrap samples are used when building trees.
                If False, the whole dataset is used to build each tree. Defaults to True.
            criterion (str, optional): The function to measure the quality of a split.
                Defaults to "gini".
            max_depth (int, optional): The maximum depth of the tree. Defaults to None.
            max_samples (Union[int, float, optional): If bootstrap is True, the number of samples
                to draw from X to train each base estimator. Defaults to None.
            min_samples_split (Union[int, float, optional): The minimum number of samples required
                to split an internal node. Defaults to 2.
            min_samples_leaf (Union[int, float], optional): The minimum number of samples required
                to be at a leaf node. Defaults to 1.
            max_features (Union[int, float, str], optional): The number of features to consider when
                looking for the best split. Defaults to None.
            max_leaf_nodes (int, optional): Grow a tree with max_leaf_nodes in best-first fashion.
                Defaults to None.
            min_impurity_decrease (Union[int, float], optional): A node will be split if this split
                induces a decrease of the impurity greater than or equal to this value.
                Defaults to 0.0.
            verbose (int, optional): how much information should be printed.
                Should be 0, 1, or 2. Defaults to 0.
        """
        super().__init__(
            n_estimators,
            bootstrap,
            criterion,
            max_depth,
            max_samples,
            min_samples_split,
            min_samples_leaf,
            max_features,
            max_leaf_nodes,
            min_impurity_decrease,
            verbose,
        )
        self._valid_score_metrics = [
            "accuracy",
            "recall",
            "precision",
            "f1_score",
            "confusion_matrix",
        ]

        try:
            assert criterion in self._valid_criterions
        except AssertionError as error:
            raise ValueError(
                f"The value for 'criterion' must be {self._valid_criterions}.\n"
            ) from error

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Uses the trained model to predict the classes of a given
        data points (also called features).

        Args:
            X (np.ndarray): the features.

        Returns:
            np.ndarray: the predicted classes.
        """
        predictions = []

        # making a prediction with each estimator
        for estimator in self.estimators_:
            predictions.append(estimator.predict(X).tolist())

        # transforming features in a list format to numpy array
        predictions = np.asarray(predictions)

        # getting the most common value for each sample
        # considering all estimators
        predictions = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), axis=0, arr=predictions
        )

        return predictions

    def score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metric: str = "accuracy",
        labels_cm: List = None,
        normalize_cm: bool = False,
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
        except AssertionError as error:
            raise ValueError(
                f"Invalid value for 'metric'. Must be {self._valid_score_metrics}.\n"
            ) from error

        y_hat = self.predict(X)

        if metric == "accuracy":
            return accuracy(y, y_hat)

        if metric == "precision":
            return precision(y, y_hat)

        if metric == "recall":
            return recall(y, y_hat)

        if metric == "f1_score":
            return f1_score(y, y_hat)

        if metric == "confusion_matrix":
            return confusion_matrix(y, y_hat, labels_cm, normalize_cm)


class RandomForestRegressor(RandomForestBase):
    """
    Creates a class for the Random Forest Regressor model.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        bootstrap: bool = True,
        criterion: str = "squared_error",
        max_depth: int = None,
        max_samples: Union[int, float] = None,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        max_features: Union[int, float, str] = None,
        max_leaf_nodes: int = None,
        min_impurity_decrease: Union[int, float] = 0.0,
        verbose: int = 0,
    ) -> None:
        """
        Creates a Random Forest Regressor instance.

        Args:
            n_estimators (str, optional): The number of trees in the forest. Defaults to 100.
            bootstrap (bool, optional): Whether bootstrap samples are used when building trees.
                If False, the whole dataset is used to build each tree. Defaults to True.
            criterion (str, optional): The function to measure the quality of a split.
                Defaults to "squared_error".
            max_depth (int, optional): The maximum depth of the tree. Defaults to None.
            max_samples (Union[int, float, optional): If bootstrap is True, the number of samples
                to draw from X to train each base estimator. Defaults to None.
            min_samples_split (Union[int, float, optional): The minimum number of samples required
                to split an internal node. Defaults to 2.
            min_samples_leaf (Union[int, float], optional): The minimum number of samples required
                to be at a leaf node. Defaults to 1.
            max_features (Union[int, float, str], optional): The number of features to consider when
                looking for the best split. Defaults to None.
            max_leaf_nodes (int, optional): Grow a tree with max_leaf_nodes in best-first fashion.
                Defaults to None.
            min_impurity_decrease (Union[int, float], optional): A node will be split if this split
                induces a decrease of the impurity greater than or equal to this value.
                Defaults to 0.0.
            verbose (int, optional): how much information should be printed.
                Should be 0, 1, or 2. Defaults to 0.
        """
        super().__init__(
            n_estimators,
            bootstrap,
            criterion,
            max_depth,
            max_samples,
            min_samples_split,
            min_samples_leaf,
            max_features,
            max_leaf_nodes,
            min_impurity_decrease,
            verbose,
        )
        self._valid_score_metrics = [
            "r_squared",
            "mse",
            "mae",
            "rmse",
            "medae",
            "mape",
            "msle",
            "max_error",
        ]

        try:
            assert criterion in self._valid_criterions
        except AssertionError as error:
            raise ValueError(
                f"The value for 'criterion' must be {self._valid_criterions}.\n"
            ) from error

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Uses the trained model to predict the classes of a given
        data points (also called features).

        Args:
            X (np.ndarray): the features.

        Returns:
            np.ndarray: the predicted classes.
        """
        predictions = []

        # making a prediction with each estimator
        for estimator in self.estimators_:
            predictions.append(estimator.predict(X).tolist())

        # transforming features in a list format to numpy array
        predictions = np.asarray(predictions)

        # getting the mean of the target for each sample
        # considering all estimators
        predictions = np.mean(predictions, axis=0)

        return predictions

    def score(
        self, X: np.ndarray, y: np.ndarray, metric: str = "r_squared"
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
        except AssertionError as error:
            raise ValueError(
                f"Invalid value for 'metric'. Must be {self._valid_score_metrics}.\n"
            ) from error

        y_hat = self.predict(X)

        if metric == "r_squared":
            return r_squared(y, y_hat)

        if metric == "mse":
            return mean_squared_error(y, y_hat)

        if metric == "mae":
            return mean_absolute_error(y, y_hat)

        if metric == "rmse":
            return root_mean_squared_error(y, y_hat)

        if metric == "medae":
            return median_absolute_error(y, y_hat)

        if metric == "mape":
            return mean_absolute_percentage_error(y, y_hat)

        if metric == "msle":
            return mean_squared_logarithmic_error(y, y_hat)

        if metric == "max_error":
            return max_error(y, y_hat)
