from abc import ABC
from typing import Union, List, Tuple
from ..utils import convert_array_numpy, split_data_into_batches
from ..losses import binary_cross_entropy, cross_entropy
from ..activations import (
    relu,
    leaky_relu,
    linear,
    softmax,
    sigmoid,
    elu,
    tanh,
    softplus,
    selu,
)
from ..metrics import (
    accuracy,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_squared_logarithmic_error,
    median_absolute_error,
    r_squared,
    recall,
    precision,
    f1_score,
    confusion_matrix,
    root_mean_squared_error,
)
from ..regularizations import l2
import numpy as np


class Layer:
    """
    Creates a layer class that will be used in the Multilayer Perceptron (MLP) model.
    """

    def __init__(
        self,
        activation: str,
        n_units: int = None,
        input_size: int = None,
        alpha: float = None,
        momentum: float = None,
    ) -> None:
        """
        Creates a Layer instance.

        Args:
            activation (str): the activation function that is being used.
            n_units (int, optional): the number of units in the layer. Defaults to None.
            input_size (int, optional): the input size of the layer. Defaults to None.
            alpha (float, optional): the strength of the L2 regularization term.
                Defaults to None.
            momentum (float, optional): the momentum for gradient descent update.
                Defaults to None.
        """
        self.n_units = n_units
        self.activation = activation
        self.alpha = alpha
        self.input_size = input_size
        self.momentum = momentum
        self.weights = None
        self.bias = None
        self.input_ = None
        self.output_ = None

    def __call__(
        self,
        propagation: str,
        input_: np.ndarray = None,
        loss: np.ndarray = None,
        learning_rate: float = None,
    ) -> np.ndarray:
        """
        Applies the forward or backward propagation.

        Args:
            propagation (str): whether it's a 'forward' or 'backward' propagation.
            input_ (np.ndarray, optional): the input of the layer, only used when
                propagation is 'forward'. Defaults to None.
            loss (np.ndarray, optional): the loss value, only used when propagation is
                'backward'. Defaults to None.
            learning_rate (float, optional): the learning rate. Defaults to None.

        Returns:
            np.ndarray: the output of the forward or backward propagation.
        """
        if propagation == "forward":
            self.input_ = input_.copy()
            self.output_ = np.matmul(self.input_, self.weights) + self.bias
            activated_output = np.zeros_like(self.output_, dtype=np.float64)

            # Apply activation function
            if self.activation == "relu":
                activated_output = relu(self.output_)
            elif self.activation == "linear":
                activated_output = linear(self.output_)
            elif self.activation == "tanh":
                activated_output = tanh(self.output_)
            elif self.activation == "sigmoid":
                activated_output = sigmoid(self.output_)
            elif self.activation == "elu":
                activated_output = elu(self.output_)
            elif self.activation == "softmax":
                activated_output = softmax(self.output_)
            elif self.activation == "leaky_relu":
                activated_output = leaky_relu(self.output_)
            elif self.activation == "softplus":
                activated_output = softplus(self.output_)
            elif self.activation == "selu":
                activated_output = selu(self.output_)

            return activated_output

        if propagation == "backward":
            activation_derivative = np.zeros_like(self.output_, dtype=np.float64)

            # Apply derivative of the activation function
            if self.activation == "relu":
                activation_derivative = relu(self.output_, derivative=True)
            elif self.activation == "linear":
                activation_derivative = linear(self.output_, derivative=True)
            elif self.activation == "tanh":
                activation_derivative = tanh(self.output_, derivative=True)
            elif self.activation == "sigmoid":
                activation_derivative = sigmoid(self.output_, derivative=True)
            elif self.activation == "elu":
                activation_derivative = elu(self.output_, derivative=True)
            elif self.activation == "softmax":
                activation_derivative = softmax(self.output_, derivative=True)
            elif self.activation == "leaky_relu":
                activation_derivative = leaky_relu(self.output_, derivative=True)
            elif self.activation == "softplus":
                activation_derivative = softplus(self.output_, derivative=True)
            elif self.activation == "selu":
                activation_derivative = selu(self.output_, derivative=True)

            activation_derivative = activation_derivative * loss
            last_loss = np.matmul(activation_derivative, self.weights.T)

            # Gradient clipping to prevent overflow
            np.clip(activation_derivative, -1e3, 1e3, out=activation_derivative)

            # Update weights and biases
            self.weights -= learning_rate * np.matmul(
                self.input_.T, activation_derivative
            ) * self.momentum + l2(self.weights, reg_lambda=self.alpha, derivative=True)

            np.clip(self.weights, -1e3, 1e3, out=self.weights)

            self.bias -= learning_rate * np.sum(
                activation_derivative, axis=0, keepdims=False
            )
            np.clip(self.bias, -1e3, 1e3, out=self.bias)

            return last_loss

    def set_input_size(self, input_size: int) -> None:
        """
        Set the input size value.

        Args:
            input_size (int): the new input size.
        """
        self.input_size = input_size

    def set_activation(self, activation: str) -> None:
        """
        Set the activation function.

        Args:
            activation (str): the new activation function.
        """
        self.activation = activation

    def initialize_weights(self) -> None:
        """
        Initialize the weights and bias values of the current layer.
        """
        sigma = np.sqrt(2 / (self.n_units + self.input_size))
        self.weights = np.random.randn(self.input_size, self.n_units) * sigma
        self.bias = np.random.randn(self.n_units) * sigma


class BaseMLP(ABC):
    """
    Creates a base class for the Multilayer Perceptron (MLP) model.
    """

    def __init__(
        self,
        loss_function: str,
        hidden_layer_sizes: np.ndarray = (100,),
        activation: str = "relu",
        alpha: float = 0.0001,
        momentum: float = 0.9,
        batch_size: Union[np.int16, str] = "auto",
        learning_rate: str = "constant",
        learning_rate_init: float = 0.001,
        max_iter: int = 200,
        tol: float = 1e-4,
        verbose: int = 0,
    ) -> None:
        """
        Creates a MLP instance.

        Args:
            loss_function (str): the loss function that will be used.
            hidden_layer_sizes (np.ndarray, optional): _description_. Defaults to (100,).
            activation (str, optional): the activation function that will be used.
                Defaults to "relu".
            alpha (float, optional): the strength of the L2 regularization term.
                Defaults to 0.0001.
            momentum (float, optional): the momentum for gradient descent update.
                Defaults to None.
            batch_size (Union[np.int16, str], optional): the batch size. Defaults to "auto".
            learning_rate (str, optional): learning rate schedule for weight updates.
                Defaults to "constant".
            learning_rate_init (float, optional): the initial value for the learning rate.
                Defaults to 0.001.
            max_iter (int, optional): the number of max iterations. Defaults to 200.
            tol (float, optional): the tolerance for optimization. Defaults to 1e-4.
            verbose (int, optional): whether to print progress messages. Defaults to 0.
        """
        self.loss_function = loss_function
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.alpha = alpha
        self.momentum = momentum
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.classes_ = None
        self.best_loss_ = np.inf
        self.loss_curve_ = []
        self.coefs_ = None
        self.intercepts_ = None
        self.n_features_in_ = None
        self.n_layers_ = None
        self.n_iter_ = 0
        self.n_outputs_ = None
        self.out_activation_ = None
        self.layers_ = []
        self._valid_activations = [
            "relu",
            "leaky_relu",
            "linear",
            "softmax",
            "sigmoid",
            "elu",
            "tanh",
            "softplus",
            "selu",
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Function responsible for fitting the MLP model.

        Args:
            X (np.ndarray): the features array.
            y (np.ndarray): the classes array.
        """
        # validating the parameters
        self._validate_parameters()

        X = convert_array_numpy(X)
        y = convert_array_numpy(y).reshape(-1, 1)

        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        batch_size = 0

        # initializing the batch size
        if self.batch_size == "auto":
            batch_size = min(200, X.shape[0])
        else:
            batch_size = self.batch_size

        # creating and initializing hidden layers
        for i, _ in enumerate(self.hidden_layer_sizes):
            self.layers_.append(
                Layer(
                    n_units=self.hidden_layer_sizes[i],
                    activation=self.activation,
                    alpha=self.alpha,
                    momentum=self.momentum,
                )
            )

            # Set input size for the layer
            if i == 0:
                self.layers_[0].set_input_size(self.n_features_in_)
            else:
                self.layers_[i].set_input_size(self.hidden_layer_sizes[i - 1])

            # Initialize weights for the layer
            self.layers_[i].initialize_weights()

        # defining the parameters for the output layer
        if isinstance(self, MLPClassifier):
            if len(self.classes_) == 2:
                self.n_outputs_ = 1
                self.out_activation_ = "sigmoid"
            else:
                self.n_outputs_ = len(self.classes_)
                self.out_activation_ = "softmax"
        elif isinstance(self, MLPRegressor):
            self.n_outputs_ = 1
            self.out_activation_ = "linear"  # Typically used in regression

        # creating and initializing the output layer
        self.layers_.append(
            Layer(
                n_units=self.n_outputs_,
                activation=self.out_activation_,
                alpha=self.alpha,
                momentum=self.momentum,
            )
        )
        self.layers_[-1].set_input_size(self.hidden_layer_sizes[-1])
        self.layers_[-1].initialize_weights()

        self.n_layers_ = len(self.layers_) + 1  # adding the input layer
        last_loss = None

        # iterating over the epochs
        for i in range(self.max_iter):
            total_loss = 0.0
            loss = None

            # iterating over the batches
            for _, batch in enumerate(
                split_data_into_batches(X=X, y=y, batch_size=batch_size), start=1
            ):
                batch_X, batch_y = batch

                # forward propagation
                y_hat = self._forward(input_=batch_X.copy())

                # calculating the loss
                if isinstance(self, MLPClassifier):
                    if self.loss_function == "bce":
                        loss = binary_cross_entropy(batch_y, y_hat, derivative=True)
                    elif self.loss_function == "cross_entropy":
                        loss = cross_entropy(batch_y, y_hat, derivative=True)
                elif isinstance(self, MLPRegressor):
                    if self.loss_function == "mse":
                        loss = (
                            2
                            * mean_squared_error(batch_y, y_hat, derivative=True)
                            / len(batch_y)
                        )

                total_loss += np.sum(loss)

                # back propagation
                self._backward(loss=loss)

            total_loss = np.mean(total_loss)

            if self.verbose != 0:
                loss_msg = f"Loss ({self.loss_function}): {total_loss}"
                metric_msg = (
                    f"Metric (Accuracy): {self.score(X, y)}"
                    if isinstance(self, MLPClassifier)
                    else ""
                )
                epoch_msg = f"Epoch: {i}/{self.max_iter}"

                if self.verbose == 1:
                    if i % 20 == 0:
                        print(f"{epoch_msg}\t\t{loss_msg}\t\t{metric_msg}\n")
                elif self.verbose == 2:
                    print(f"{epoch_msg}\t\t{loss_msg}\t\t{metric_msg}\n")

            # checking the tolerance stop criteria
            if last_loss is not None:
                if np.max(np.abs(last_loss - loss)) < self.tol:
                    break

            last_loss = loss
            self.n_iter_ += 1
            self.loss_curve_.append(total_loss)
            self.best_loss_ = min(self.best_loss_, total_loss)

        self.coefs_ = [layer.weights for layer in self.layers_]
        self.intercepts_ = [layer.bias for layer in self.layers_]

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Uses the trained model to predict the classes of a given
        data points (also called features).

        Args:
            X (np.ndarray): the features.

        Returns:
            np.ndarray: the predicted classes.
        """

    def score(
        self, X: np.ndarray, y: np.ndarray, metric: str = "accuracy", **kwargs
    ) -> Union[np.float32, np.ndarray]:
        """
        Calculates the score of the model on a given set for a
        determined metric.

        Args:
            X (np.ndarray): the features.
            y (np.ndarray): the targets of the features.
            metric (str, optional): which metric to use. Defaults to "r_squared".

        Returns:
            np.float32: the score achieved by the model.
        """

    def _forward(self, input_: np.ndarray) -> np.ndarray:
        """
        Applies the forward propagation.

        Args:
            input_ (np.ndarray): the input data that will be passed
                through the model.

        Returns:
            np.ndarray: the raw output of the model.
        """
        for layer in self.layers_:
            input_ = layer(input_=input_, propagation="forward")

        return input_

    def _backward(self, loss: np.ndarray) -> None:
        """
        Applies the backward propagation.

        Args:
            loss (np.ndarray): the loss for the whole model.
        """
        for layer in reversed(self.layers_):
            loss = layer(
                propagation="backward", loss=loss, learning_rate=self.learning_rate_init
            )

    def _validate_parameters(self) -> None:
        """
        Auxiliary function used to validate the values of the parameters
        passed during the initialization.
        """
        # validating the hidden_layer_sizes
        try:
            assert isinstance(self.hidden_layer_sizes, Tuple) and all(
                h > 0 for h in self.hidden_layer_sizes
            )
        except AssertionError as error:
            raise ValueError(
                "The 'hidden_layer_sizes must be a Tuple containing '"
                + "a positive number representing the number of units "
                + f"in wich layer, got {self.hidden_layer_sizes}.\n"
            ) from error

        # validating the activation value
        try:
            assert (self.activation in self._valid_activations) and (
                isinstance(self.activation, str)
            )
        except AssertionError as error:
            raise ValueError(
                f"The 'activation' must be {self._valid_activations}.\n"
            ) from error

        # validating the alpha value
        try:
            assert (self.alpha > 0) and (isinstance(self.alpha, (int, float)))
        except AssertionError as error:
            raise ValueError(
                "The 'alpha' must be a positive number bigger than zero.\n"
            ) from error

        # validating the momentum value
        try:
            assert (self.momentum > 0 and self.momentum < 1) and (
                isinstance(self.momentum, (int, float))
            )
        except AssertionError as error:
            raise ValueError(
                "The 'momentum' must be a positive number between zero and 1.\n"
            ) from error

        # validating the batch_size value
        try:
            assert isinstance(self.batch_size, (int, str))
        except AssertionError as error:
            raise TypeError(
                "The 'batch_size' must be an integer or string.\n"
            ) from error

        if isinstance(self.batch_size, str):
            try:
                assert self.batch_size == "auto"
            except AssertionError as error:
                raise ValueError(
                    f"The 'batch_size' value must be 'auto', got {self.batch_size}.\n"
                ) from error
        elif isinstance(self.batch_size, int):
            try:
                assert self.batch_size > 0
            except AssertionError as error:
                raise ValueError(
                    "The 'batch_size' value must be a positive number bigger than zero.\n"
                ) from error

        # validating the learning_rate value
        try:
            assert (self.learning_rate == "constant") and (
                isinstance(self.learning_rate, str)
            )
        except AssertionError as error:
            raise ValueError(
                "The 'learning_rate' must be equal to 'constant', "
                + f"got {self.learning_rate}.\n"
            ) from error

        # validating the learning_rate_init value
        try:
            assert (self.learning_rate_init > 0) and (
                isinstance(self.learning_rate_init, float)
            )
        except AssertionError as error:
            raise ValueError(
                "The 'learning_rate_init' must be a positive number bigger than zero.\n"
            ) from error

        # validating the max_iter value
        try:
            assert (self.max_iter > 0) and (isinstance(self.max_iter, int))
        except AssertionError as error:
            raise ValueError(
                "The 'max_iter' must be a positive number bigger than zero.\n"
            ) from error

        # validating the tol value
        try:
            assert (self.tol > 0) and (isinstance(self.tol, float))
        except AssertionError as error:
            raise ValueError(
                "The 'tol' must be a positive number bigger than zero.\n"
            ) from error

        # validating the verbose value
        try:
            assert self.verbose in [0, 1, 2]
        except AssertionError as error:
            raise ValueError(
                "Indalid value for 'verbose'. Must be 0, 1, or 2.\n"
            ) from error


class MLPClassifier(BaseMLP):
    """
    Creates a class for the Multilayer Perceptron (MLP) classifier model.
    """

    def __init__(
        self,
        loss_function: str,
        hidden_layer_sizes: np.ndarray = (100,),
        activation: str = "relu",
        alpha: float = 0.0001,
        momentum: float = 0.9,
        batch_size: np.int16 | str = "auto",
        learning_rate: str = "constant",
        learning_rate_init: float = 0.001,
        max_iter: int = 200,
        tol: float = 0.0001,
        verbose: int = 0,
    ) -> None:
        """
        Creates a MLP Classifier instance.

        Args:
            loss_function (str): the loss function that will be used.
            hidden_layer_sizes (np.ndarray, optional): _description_. Defaults to (100,).
            activation (str, optional): the activation function that will be used.
                Defaults to "relu".
            alpha (float, optional): the strength of the L2 regularization term. Defaults to 0.0001.
            momentum (float, optional): the momentum for gradient descent update.
                Defaults to None.
            batch_size (Union[np.int16, str], optional): the batch size. Defaults to "auto".
            learning_rate (str, optional): learning rate schedule for weight updates.
                Defaults to "constant".
            learning_rate_init (float, optional): the initial value for the learning rate.
                Defaults to 0.001.
            max_iter (int, optional): the number of max iterations. Defaults to 200.
            tol (float, optional): the tolerance for optimization. Defaults to 1e-4.
            verbose (int, optional): whether to print progress messages. Defaults to 0.
        """
        super().__init__(
            loss_function,
            hidden_layer_sizes,
            activation,
            alpha,
            momentum,
            batch_size,
            learning_rate,
            learning_rate_init,
            max_iter,
            tol,
            verbose,
        )
        self._valid_metrics = [
            "accuracy",
            "recall",
            "precision",
            "f1_score",
            "confusion_matrix",
        ]
        self._valid_loss_functions = ["bce", "cross_entropy"]

        try:
            assert loss_function in self._valid_loss_functions
        except AssertionError as error:
            raise ValueError(
                f"The value for 'loss_function' must be {self._valid_loss_functions}.\n"
            ) from error

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Uses the trained model to predict the classes of a given
        set (also called features).

        Args:
            X (np.ndarray): the features.

        Returns:
            np.ndarray: the predicted classes.
        """
        # forward propagation
        y_hat = self._forward(input_=X.copy())

        if len(self.classes_) > 2:
            y_hat = np.argmax(y_hat, axis=-1)
        else:
            y_hat = (y_hat > 0.5).astype(int)

        return y_hat

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
        except AssertionError as error:
            raise ValueError(
                f"Invalid value for 'metric'. Must be {self._valid_metrics}.\n"
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


class MLPRegressor(BaseMLP):
    """
    Creates a class for the Multilayer Perceptron (MLP) Regressor model.
    """

    def __init__(
        self,
        loss_function: str = "mse",
        hidden_layer_sizes: Tuple[int, ...] = (100,),
        activation: str = "relu",
        alpha: float = 0.0001,
        momentum: float = 0.9,
        batch_size: Union[int, str] = "auto",
        learning_rate: str = "constant",
        learning_rate_init: float = 0.001,
        max_iter: int = 200,
        tol: float = 1e-4,
        verbose: int = 0,
    ) -> None:
        """
        Creates a MLP Regressor instance.

        Args:
            loss_function (str, optional): The loss function that will be used. Defaults to "mse".
            hidden_layer_sizes (Tuple[int, ...], optional): The sizes of the hidden layers. Defaults to (100,).
            activation (str, optional): The activation function that will be used. Defaults to "relu".
            alpha (float, optional): The strength of the L2 regularization term. Defaults to 0.0001.
            momentum (float, optional): The momentum for gradient descent update. Defaults to 0.9.
            batch_size (Union[int, str], optional): The batch size. Defaults to "auto".
            learning_rate (str, optional): Learning rate schedule for weight updates. Defaults to "constant".
            learning_rate_init (float, optional): The initial value for the learning rate. Defaults to 0.001.
            max_iter (int, optional): The number of max iterations. Defaults to 200.
            tol (float, optional): The tolerance for optimization. Defaults to 1e-4.
            verbose (int, optional): Whether to print progress messages. Defaults to 0.
        """
        super().__init__(
            loss_function,
            hidden_layer_sizes,
            activation,
            alpha,
            momentum,
            batch_size,
            learning_rate,
            learning_rate_init,
            max_iter,
            tol,
            verbose,
        )
        self._valid_loss_functions = ["mse"]
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

        # Validate the loss function
        if self.loss_function not in self._valid_loss_functions:
            raise ValueError(
                f"The value for 'loss_function' must be one of {self._valid_loss_functions}."
            )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Uses the trained model to predict the values of a given set of features.

        Args:
            X (np.ndarray): The features.

        Returns:
            np.ndarray: The predicted values.
        """
        X = convert_array_numpy(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected input features of size {self.n_features_in_}, but got {X.shape[1]}."
            )

        # Forward pass through the network
        y_hat = self._forward(X)

        # If the output is a 2D array with shape (n_samples, 1), flatten it to 1D array
        if y_hat.shape[1] == 1:
            y_hat = y_hat.flatten()

        return y_hat

    def score(
        self, X: np.ndarray, y: np.ndarray, metric: str = "r_squared"
    ) -> np.float64:
        """
        Calculates the score of the model on a given set for a determined metric.

        Args:
            X (np.ndarray): The features.
            y (np.ndarray): The targets of the features.
            metric (str, optional): Which metric to use. Defaults to "r_squared".

        Returns:
            np.float64: The score achieved by the model.
        """
        try:
            assert metric in self._valid_score_metrics
        except AssertionError as error:
            raise ValueError(
                f"Invalid value for 'metric'. Must be one of {self._valid_score_metrics}."
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

        raise ValueError(
            f"Invalid value for 'metric'. Must be one of {self._valid_score_metrics}."
        )
