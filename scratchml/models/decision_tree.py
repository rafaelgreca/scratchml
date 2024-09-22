from collections import Counter
from abc import ABC
from typing import Union, List, Tuple
from ..utils import convert_array_numpy
from ..criterions import entropy, gini, squared_error
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
import math


class Node:
    """
    Creates a node class.
    """

    def __init__(
        self,
        feature_index: int = None,
        threshold: Union[int, float] = None,
        left: "Node" = None,
        right: "Node" = None,
        value: Union[int, float] = None,
    ) -> None:
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTreeBase(ABC):
    """
    Creates a base class for the Decision Tree model.
    """

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: int = None,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        max_features: Union[int, float, str] = None,
        max_leaf_nodes: int = None,
        min_impurity_decrease: float = 0.0,
    ) -> None:
        """
        Creates a Decision Tree Base.

        Args:
            criterion (str, optional): The function to measure the quality of a split.
                Defaults to "gini".
            max_depth (int, optional): The maximum depth of the tree. Defaults to None.
            min_samples_split (Union[int, float, optional): The minimum number of samples required
                to split an internal node. Defaults to 2.
            min_samples_leaf (Union[int, float], optional): The minimum number of samples required
                to be at a leaf node. Defaults to 1.
            max_features (Union[int, float, str], optional): The number of features to consider when
                looking for the best split. Defaults to None.
            max_leaf_nodes (int, optional): Grow a tree with max_leaf_nodes in best-first fashion.
                Defaults to None.
            min_impurity_decrease (float, optional): A node will be split if this split induces a
                decrease of the impurity greater than or equal to this value. Defaults to 0.0.
        """
        self.criterion = criterion
        self.max_depth = np.inf if max_depth is None else max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_leaf_nodes = np.inf if max_leaf_nodes is None else max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features_ = None
        self.classes_ = None
        self.n_classes_ = None
        self.n_features_in_ = None
        self.n_outputs_ = None
        self.tree_ = None
        self.n_samples_ = None
        self.depth_ = None
        self.leaf_nodes_ = 0

        if isinstance(self, DecisionTreeClassifier):
            self._valid_criterions = [
                "gini",
                "entropy",
                "log_loss",
            ]
        elif isinstance(self, DecisionTreeRegressor):
            self._valid_criterions = ["squared_error"]

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Function responsible for fitting the Decision Tree model.

        Args:
            X (np.ndarray): the features array.
            y (np.ndarray): the classes array.
        """
        X = convert_array_numpy(X)
        y = convert_array_numpy(y)

        self.n_samples_, self.n_features_in_ = X.shape

        if isinstance(self, DecisionTreeClassifier):
            self.classes_ = np.unique(y)
            self.n_classes_ = len(np.unique(y))

        if self.max_features is None:
            self.max_features_ = self.n_features_in_
        elif isinstance(self.max_features, int):
            self.max_features_ = self.max_features
        elif isinstance(self.max_features, float):
            self.max_features_ = max(1, int(self.max_features * self.n_features_in_))
        elif isinstance(self.max_features, str):
            if self.max_features == "sqrt":
                self.max_features_ = math.floor(math.sqrt(self.n_features_in_))
            elif self.max_features == "log2":
                self.max_features_ = math.floor(math.log2(self.n_features_in_))

        self.tree_ = self._build_tree(X=X, y=y, depth=0)

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """
        Auxiliary function responsible for building the Decision Tree.

        Args:
            X (np.ndarray): the features array.
            y (np.ndarray): the classes array.
            depth (int, optional): the current depth of the tree. Defaults to 0.
        """
        num_samples, num_feats = X.shape
        self.depth_ = depth
        features_indexes = np.arange(0, num_feats, 1, dtype=int)

        if self.max_features is not None:
            features_indexes = np.random.choice(
                features_indexes, self.max_features_, replace=False
            )

        # creating an internal node
        if (self.depth_ <= self.max_depth) and (num_samples >= self.min_samples_split):
            # finding the best split
            best_gain, best_feature, best_threshold = self._get_best_split(
                X=X, y=y, feature_indexes=features_indexes
            )

            if best_gain > 0:
                # creating the children node
                left, right = self._create_split(
                    X=X[:, best_feature], threshold=best_threshold
                )

                left_node = self._build_tree(
                    X=X[left, :], y=y[left], depth=self.depth_ + 1
                )
                right_node = self._build_tree(
                    X=X[right, :], y=y[right], depth=self.depth_ + 1
                )

                return Node(
                    feature_index=best_feature,
                    threshold=best_threshold,
                    left=left_node,
                    right=right_node,
                    value=None,
                )

        return self._create_leaf_node(y)

        # creating a leaf node
        # if (len(y) > self.min_samples_leaf) and (self.leaf_nodes_ < self.max_leaf_nodes):
        #     self.leaf_nodes_ += 1
        #     return self._create_leaf_node(y)

    def _create_leaf_node(self, y: np.ndarray) -> Node:
        """
        Auxiliary function to create a leaf node.

        Args:
            y (np.ndarray): the classes array.

        Returns:
            Node: the leaf node.
        """
        if isinstance(self, DecisionTreeClassifier):
            counter = Counter(y)
            value = counter.most_common(1)[0][0]
            return Node(value=value)

        return Node(value=np.mean(y))

    def _get_best_split(
        self, X: np.ndarray, y: np.ndarray, feature_indexes: List
    ) -> Tuple[float, int, Union[int, float]]:
        """
        Auxiliary function responsible for getting the best split for the current node.

        Args:
            X (np.ndarray): the features array.
            y (np.ndarray): the classes array.
            feature_indexes (List): a list containing the index of the features that
                wasn't used yet.

        Returns:
            Tuple[int, Union[int, float]]: the information gain, best feature index
                and the best threshold, respectively.
        """
        best_gain = np.NINF
        split_index = None
        split_threshold = None

        for feature_index in feature_indexes:
            _X = X[:, feature_index].reshape(-1)
            thresholds = np.unique(_X)

            for threshold in thresholds:
                information_gain = self._calculate_information_gain(
                    X=_X, y=y, threshold=threshold
                )

                if information_gain > best_gain:
                    best_gain = information_gain
                    split_index = feature_index
                    split_threshold = threshold

        return best_gain, split_index, split_threshold

    def _calculate_information_gain(
        self, X: np.ndarray, y: np.ndarray, threshold: Union[int, float]
    ) -> float:
        """
        Auxiliary function responsible for calculating the information gain of a split.

        Args:
            X (np.ndarray): the features array.
            y (np.ndarray): the classes array.
            threshold (Union[int, float]): the threshold for the split.
        """
        # creating the left and right children
        left, right = self._create_split(X=X, threshold=threshold)

        # empty list, so the information gain should be zero
        if len(left) == 0 or len(right) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(left), len(right)
        c_parent, c_l, c_r = 0, 0, 0

        # calculating the weighted average entropy of children
        if self.criterion in ["entropy", "log_loss"]:
            c_parent = entropy(y)
            c_l = entropy(y[left])
            c_r = entropy(y[right])
        elif self.criterion == "gini":
            c_parent = gini(y)
            c_l = gini(y[left])
            c_r = gini(y[right])
        elif self.criterion == "squared_error":
            c_parent = squared_error(np.mean(y), y)
            c_l = squared_error(np.mean(y[left]), y[left])
            c_r = squared_error(np.mean(y[right]), y[right])

        c_child = (n_l / n) * c_l + (n_r / n) * c_r
        information_gain = c_parent - c_child
        impurity_decrease = np.NINF

        if n > 0:
            impurity_decrease = n / (
                self.n_samples_ * (c_parent - ((n_r / n) * c_r) - ((n_l / n)) * c_l)
            )

        if impurity_decrease >= self.min_impurity_decrease:
            return information_gain

        return 0

    def _create_split(
        self, X: np.ndarray, threshold: Union[int, float]
    ) -> Tuple[int, int]:
        """
        Auxiliary function responsible for splitting the dataset.

        Args:
            X (np.ndarray): the features array.
            threshold (Union[int, float]): the threshold for the split.

        Returns:
            Tuple[int, int]: the left and right features indexes, respectively.
        """
        left = np.argwhere(X <= threshold).reshape(-1)
        right = np.argwhere(X > threshold).reshape(-1)
        return left, right

    def _walk_on_tree(self, X: np.array, node: Node) -> Union[int, float]:
        """
        Auxiliary function responsible for walking on the builded tree
        in order to reach the leaf node and make a classification.

        Args:
            X (np.array): the features array.
            node (Node): the builded, trained decision tree.

        Returns:
            Union[int, float]: the target prediction.
        """
        # checking if it's a leaf node
        if node.value is not None:
            return node.value

        # checking if the value of the best feature for that node
        # it's lower or higher than the best threshold
        if X[node.feature_index] <= node.threshold:
            return self._walk_on_tree(X, node.left)

        return self._walk_on_tree(X, node.right)

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
        predictions = np.array([self._walk_on_tree(x, self.tree_) for x in X])
        return predictions

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

    def get_depth(self) -> int:
        """
        Get the depth of the tree.

        Returns:
            int: tree's depth.
        """
        return self.depth_

    def get_n_leaves(self) -> int:
        """
        Get the number of leaves of the tree.

        Returns:
            int: tree's number of leaves.
        """
        return self.leaf_nodes_

    def _validate_parameters(self) -> None:
        """
        Auxiliary function used to validate the values of the parameters
        passed during the initialization.
        """
        # validating the criterion value
        try:
            assert (self.criterion in self._valid_criterions) and (
                isinstance(self.criterion, str)
            )
        except AssertionError as error:
            raise ValueError(
                f"The 'criterion' must be {self._valid_criterions}.\n"
            ) from error

        # validating the max_depth value
        try:
            assert self.max_depth > 0 and isinstance(self.max_leaf_nodes, int)
        except AssertionError as error:
            raise ValueError(
                "The value for 'max_depth' must be a positive number.\n"
            ) from error

        # validating the min_samples_split value
        try:
            assert self.min_samples_split > 2 and isinstance(self.max_leaf_nodes, int)
        except AssertionError as error:
            raise ValueError(
                "The value for 'min_samples_split' must be a positive number "
                + "bigger than 2.\n"
            ) from error

        # validating the min_samples_leaf value
        try:
            assert self.min_samples_leaf > 1 and isinstance(self.max_leaf_nodes, int)
        except AssertionError as error:
            raise ValueError(
                "The value for 'min_samples_leaf' must be a positive number "
                + "bigger than 1.\n"
            ) from error

        # validating the max_leaf_nodes value
        try:
            assert self.max_leaf_nodes > 0 and isinstance(self.max_leaf_nodes, int)
        except AssertionError as error:
            raise ValueError(
                "The value for 'max_leaf_nodes' must be a positive number.\n"
            ) from error

        # validating the min_impurity_decrease value
        try:
            assert (self.min_impurity_decrease > 0) and (
                isinstance(self.min_impurity_decrease, float)
            )
        except AssertionError as error:
            raise ValueError(
                "The value for 'min_impurity_decrease' must be a positive number.\n"
            ) from error

        # validating the max_features value
        try:
            assert isinstance(self.max_features, (int, float, str))
        except AssertionError as error:
            raise TypeError(
                "The type for 'max_features' must be integer, float, or string.\n"
            ) from error

        if isinstance(self.max_features, int):
            try:
                assert (self.max_features > 0) and (
                    self.max_features < self.n_features_in_
                )
            except AssertionError as error:
                raise ValueError(
                    "The value for 'max_features' must be a positive number "
                    + f"smaller than {self.n_features_in_}.\n"
                ) from error
        elif isinstance(self.max_features, float):
            try:
                assert (self.max_features > 0) and (self.max_features < 1)
            except AssertionError as error:
                raise ValueError(
                    "The value for 'max_features' must be between 0 and 1.\n"
                ) from error
        elif isinstance(self.max_features, str):
            try:
                assert self.max_features in ["sqrt", "log2"]
            except AssertionError as error:
                raise ValueError(
                    "The value for 'max_features' must be 'sqrt' or 'log2'.\n"
                ) from error


class DecisionTreeClassifier(DecisionTreeBase):
    """
    Creates a class for the Decision Tree Classifier model.
    """

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: int = None,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        max_features: Union[int, float, str] = None,
        max_leaf_nodes: int = None,
        min_impurity_decrease: float = 0.0,
    ) -> None:
        """
        Creates a Decision Tree Base.

        Args:
            criterion (str, optional): The function to measure the quality of a split.
                Defaults to "gini".
            max_depth (int, optional): The maximum depth of the tree. Defaults to None.
            min_samples_split (Union[int, float, optional): The minimum number of samples required
                to split an internal node. Defaults to 2.
            min_samples_leaf (Union[int, float], optional): The minimum number of samples required
                to be at a leaf node. Defaults to 1.
            max_features (Union[int, float, str], optional): The number of features to consider when
                looking for the best split. Defaults to None.
            max_leaf_nodes (int, optional): Grow a tree with max_leaf_nodes in best-first fashion.
                Defaults to None.
            min_impurity_decrease (float, optional): A node will be split if this split induces a
                decrease of the impurity greater than or equal to this value. Defaults to 0.0.
        """
        super().__init__(
            criterion,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            max_features,
            max_leaf_nodes,
            min_impurity_decrease,
        )
        self._valid_score_metrics = [
            "accuracy",
            "recall",
            "precision",
            "f1_score",
            "confusion_matrix",
        ]

        try:
            assert criterion in ["gini", "entropy", "log_loss"]
        except AssertionError as error:
            raise ValueError(
                "The value for 'criterion' must be 'gini', 'entropy' or 'log_loss'.\n"
            ) from error

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


# class DecisionTreeRegressor(DecisionTreeBase):
#     """
#     Creates a class for the Decision Tree Regressor model.
#     """

#     def __init__(
#         self,
#         criterion: str = "squared_error",
#         max_depth: int = None,
#         min_samples_split: Union[int, float] = 2,
#         min_samples_leaf: Union[int, float] = 1,
#         max_features: Union[int, float, str] = None,
#         max_leaf_nodes: int = None,
#         min_impurity_decrease: float = 0.0,
#     ) -> None:
#         """
#         Creates a Decision Tree Base.

#         Args:
#             criterion (str, optional): The function to measure the quality of a split.
#                 Defaults to "squared_error".
#             max_depth (int, optional): The maximum depth of the tree. Defaults to None.
#             min_samples_split (Union[int, float, optional): The minimum number of samples required
#                 to split an internal node. Defaults to 2.
#             min_samples_leaf (Union[int, float], optional): The minimum number of samples required
#                 to be at a leaf node. Defaults to 1.
#             max_features (Union[int, float, str], optional): The number of features to consider when
#                 looking for the best split. Defaults to None.
#             max_leaf_nodes (int, optional): Grow a tree with max_leaf_nodes in best-first fashion.
#                 Defaults to None.
#             min_impurity_decrease (float, optional): A node will be split if this split induces a
#                 decrease of the impurity greater than or equal to this value. Defaults to 0.0.
#         """
#         super().__init__(
#             criterion,
#             max_depth,
#             min_samples_split,
#             min_samples_leaf,
#             max_features,
#             max_leaf_nodes,
#             min_impurity_decrease,
#         )
#         self._valid_score_metrics = [
#             "r_squared",
#             "mse",
#             "mae",
#             "rmse",
#             "medae",
#             "mape",
#             "msle",
#             "max_error",
#         ]

#         try:
#             assert criterion in ["squared_error"]
#         except AssertionError as error:
#             raise ValueError(
#                 "The value for 'criterion' must be 'squared_error'.\n"
#             ) from error

#     def score(
#         self, X: np.ndarray, y: np.ndarray, metric: str = "r_squared"
#     ) -> np.float64:
#         """
#         Uses the trained model to predict the classes of a given
#         data points (also called features).

#         Args:
#             X (np.ndarray): the features array.
#             y (np.ndarray): the labels array.
#             metric (string): which metric should be used to assess
#                 the model's performance. Defaults to R Squared.

#         Returns:
#             np.float32: the score achieved by the model.
#         """
#         try:
#             assert metric in self._valid_score_metrics
#         except AssertionError as error:
#             raise ValueError(
#                 f"Invalid value for 'metric'. Must be {self._valid_score_metrics}.\n"
#             ) from error

#         y_hat = self.predict(X)

#         if metric == "r_squared":
#             return r_squared(y, y_hat)

#         if metric == "mse":
#             return mean_squared_error(y, y_hat)

#         if metric == "mae":
#             return mean_absolute_error(y, y_hat)

#         if metric == "rmse":
#             return root_mean_squared_error(y, y_hat)

#         if metric == "medae":
#             return median_absolute_error(y, y_hat)

#         if metric == "mape":
#             return mean_absolute_percentage_error(y, y_hat)

#         if metric == "msle":
#             return mean_squared_logarithmic_error(y, y_hat)

#         if metric == "max_error":
#             return max_error(y, y_hat)
