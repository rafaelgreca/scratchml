<a id="readme-top"></a>

# Roadmap

## Implementations

- [x] Scalers
    - [x] [StandardScaler](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/scalers.py#L155)
    - [x] [MinMaxScaler](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/scalers.py#L37)
- [ ] Regularizations
    - [x] [L1](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/regularizations.py#L4)
    - [x] [L2](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/regularizations.py#L27)
    - [ ] Batch Normalization
- [x] Activation functions
    - [x] [Sigmoid](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/activations.py#L109)
    - [x] [ReLU](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/activations.py#L23)
    - [x] [Linear](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/activations.py#L4)
    - [x] [Softmax](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/activations.py#L132)
    - [x] [TanH](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/activations.py#L84)
    - [x] [ELU](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/activations.py#L42)
    - [x] [Leaky ReLU](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/activations.py#L65)
    - [x] [SoftPlus](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/activations.py#L157)
    - [x] [SELU](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/activations.py#L180)
- [x] Loss functions
    - [x] [Binary Cross Entropy](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/losses.py#L4)
    - [x] [Cross Entropy](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/losses.py#L33)
- [x] Metrics
    - [x] Regression Metrics
        - [x] [Mean Squared Error (MSE)](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/metrics.py#L7)
        - [x] [Root Mean Squared Error (RMSE)](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/metrics.py#L29)
        - [x] [Mean Absolute Error (MAE)](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/metrics.py#L51)
        - [x] [Median Absolute Error (MedAE)](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/metrics.py#L73)
        - [x] [Mean Absolute Percentage Error (MAPE)](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/metrics.py#L95)
        - [x] [Mean Squared Logarithmic Error (MSLE)](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/metrics.py#L128)
        - [x] [Max Error (ME)](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/metrics.py#L156)
        - [x] [R Squared (R2)](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/metrics.py#L180)
    - [x] Classification Metrics
        - [x] [Accuracy](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/metrics.py#L200)
        - [x] [Precision](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/metrics.py#L215)
        - [x] [Recall](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/metrics.py#L272)
        - [x] [F1-Score](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/metrics.py#L329)
        - [x] [Confusion Matrix](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/metrics.py#L373)
        - [x] [ROC AUC Score](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/metrics.py#L474)
        - [x] [False Positive Rate (FPR)](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/metrics.py#L458)
        - [x] [True Positive Rate (TPR)](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/metrics.py#L442)
- [x] Distances
    - [x] [Euclidean](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/distances.py#L6)
    - [x] [Manhattan](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/distances.py#L26)
    - [x] [Chebyshev](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/distances.py#L46)
    - [x] [Minkowski](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/distances.py#L66)
- [x] Encoders
    - [x] [One-hot Encoding](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/encoders.py#L133)
    - [x] [Label Encoding](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/encoders.py#L39)
- [x] Splitters
    - [x] [KFold](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/utils.py#L42)
    - [x] [Stratify KFold](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/utils.py#L42)
    - [x] [Train Test Split](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/utils.py#L187)
    - [x] [Split Into Batches](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/utils.py#L5)
- [ ] Models
    - [ ] Isolation Forest
    - [ ] AdaBoost
        - [ ] AdaBoost Regressor
        - [ ] AdaBoost Classifier
    - [x] [Linear Regression](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/models/linear_regression.py)
    - [x] [Logistic Regression](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/models/logistic_regression.py)
    - [x] SVM
        - [x] [SVC](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/models/svc.py)
        - [x] [SVR](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/models/svr.py)
    - [x] KNN
        - [x] [KNN Classifier](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/models/knn.py#L236)
        - [x] [KNN Regressor](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/models/knn.py#L375)
    - [x] [Naive Bayes](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/models/naive_bayes.py)
    - [x] Random Forest
        - [x] [Random Forest Classifier](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/models/random_forest.py#L291)
        - [x] [Random Forest Regressor](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/models/random_forest.py#L445)
    - [x] Decision Tree
        - [x] [Decision Tree Classifier](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/models/decision_tree.py#L525)
        - [x] [Decision Tree Regressor](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/models/decision_tree.py#L640)
    - [x] [Perceptron](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/models/perceptron.py)
    - [x] MLP
        - [x] [MLP Classifier](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/models/multilayer_perceptron.py#L569)
        - [x] [MLP Regressor](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/models/multilayer_perceptron.py#L710)
    - [x] [KMeans](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/models/kmeans.py)
    - [x] [PCA](https://github.com/rafaelgreca/scratchml/blob/main/scratchml/models/pca.py)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Known Issues

- [ ] Binary Labeling Issue in Perceptron Model
- [ ] CI Unit test workflow not working (Numpy version error)
- [ ] MLP not working properly when dealing with a multiclass classification problem (I think it's a vanishing gradient problem)
- [ ] Sometimes the Logistic Regression model gets stuck (I think it's a vanishing gradient problem)
- [ ] Recursion error when max depth is None in the Decision Tree model
- [ ] Zero Division Warning in the Decision Tree code

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Enhancement

- [ ] Create visualization plots for all models (including for the training step)
- [ ] Create a function to print the Decision Tree and Random Forest Tree
- [ ] Optimize KNN code (looping functions and distance metrics taking too long)
- [ ] Improve testings (exclude redudant tests and add tests for model parameters)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
