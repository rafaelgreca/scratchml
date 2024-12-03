
## Badges

<p align="center">
    <a href="https://github.com/rafaelgreca/scratchml/contributors" alt="Contributors">
        <img src="https://img.shields.io/github/contributors/rafaelgreca/scratchml?color=greeb&style=for-the-badge" /></a>
    <a href="https://github.com/rafaelgreca/scratchml/forks" alt="Forks">
        <img src="https://img.shields.io/github/forks/rafaelgreca/scratchml?color=greeb&style=for-the-badge" /></a>
    <a href="https://github.com/rafaelgreca/scratchml/stars" alt="Stars">
        <img src="https://img.shields.io/github/stars/rafaelgreca/scratchml?color=greeb&style=for-the-badge" /></a>
    <a href="https://github.com/rafaelgreca/scratchml/issues" alt="Issues">
        <img src="https://img.shields.io/github/issues/rafaelgreca/scratchml?color=greeb&style=for-the-badge" /></a>
    <a href="https://github.com/rafaelgreca/scratchml/pulse" alt="Activity">
        <img src="https://img.shields.io/github/commit-activity/m/rafaelgreca/scratchml?color=greeb&style=for-the-badge" /></a>
    <a alt="Downloads">
        <img src="https://img.shields.io/github/downloads-pre/rafaelgreca/scratchml/latest/total?color=greeb&style=for-the-badge">
    <a alt="Version">
        <img src="https://img.shields.io/badge/version-9.0.0-orange.svg?color=greeb&style=for-the-badge" /></a>
    <a href="https://github.com/rafaelgreca/scratchml/blob/main/LICENSE" alt="License">
        <img src="https://img.shields.io/badge/license-MIT-blue?color=greeb&style=for-the-badge" /></a>

</p>

# ScratchML

A Python library called ScratchML was created to build the most fundamental Machine Learning models from scratch (using only Numpy), emphasizing producing user-friendly, straightforward, easy-to-use, well-organized implementations for novices and enthusiasts.

Disclaimer: This library is not intended to surpass those that already exist and which are better, more optimized, and with more diversity of implemented algorithms (such as scikit-learn, PyTorch, Keras, and Tensorflow), but rather to provide code that is easier to understand, simple, and friendly for beginners and enthusiasts in the field of artificial intelligence who wish to gain a deeper understanding of how algorithms work or who want to contribute to an open-source repository.

## Installation

To install this package, first clone the repository to the directory of your choice using the following command:

```bash
git clone https://github.com/rafaelgreca/scratchml.git
```

### Using Virtual Environment

Create a virtual environment (ideally using conda) and install the requirements with the following command:

```bash
conda create --name scratchml python=3.11.9
conda activate scratchml
pip install -r requirements/requirements.txt
```

### Using Docker

Build the Docker image using the following command:

```bash
sudo docker build -f Dockerfile -t scratchml . --no-cache
```

Run the Docker container using the following command:

```bash
sudo docker run -d -p 8000:5000 --name scratchml scratchml
```

## Usage/Examples

See the `examples` folder to see some use cases.


## Running Tests

### Locally

Run the following command on the root folder:

```bash
python3 -m unittest discover -p 'test_*.py'
```

### Using Docker

Build the Docker image using the following command:

```bash
sudo docker build -f test.Dockerfile -t test_scratchml . --no-cache
```

Run the Docker container using the following command:

```bash
sudo docker run -d -p 8001:5000 --name test_scratchml test_scratchml
```

## Roadmap

Implementations:

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

## Feedback

If you have any feedback, please feel free to create an issue pointing out whatever you want or reach out to me at rgvieira97@gmail.com

## Contributing

Contributions are what makes the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

If you have a suggestion that would make this better, please read carefully the [Contributing Guide](https://github.com/rafaelgreca/scratchml/blob/main/docs/CONTRIBUTING.md) and the [Code of Conduct](https://github.com/rafaelgreca/scratchml/blob/main/docs/CODE_OF_CONDUCT.md) before contributing.

## Acknowledge

We would like to thank all these amazing materials and repositories for their amazing work, which indirectly contributed in some sort or that inspired us to create this project.

- [REPOSITORY] [SKADI by Douglas Oliveira](https://github.com/Dellonath/SKADI/)
- [REPOSITORY] [ML From Scratch by Erik Linder-Norén](https://github.com/eriklindernoren/ML-From-Scratch)
- [REPOSITORY] [Machine Learning from Scratch by AssemblyAI](https://github.com/AssemblyAI-Community/Machine-Learning-From-Scratch)
- [COURSE] [Machine Learning Specialization by Andrew Ng](https://www.coursera.org/specializations/machine-learning-introduction)
- [COURSE] [Machine Learning From Scratch by AssemblyAI](https://www.youtube.com/watch?v=p1hGz0w_OCo&list=PLcWfeUsAys2k_xub3mHks85sBHZvg24Jd)


## License

Distributed under the [MIT](https://choosealicense.com/licenses/mit/) License. See LICENSE for more information.

## Authors

A huge shoutout to everyone who contributed to the success of the project. [Check everyone here!](https://github.com/rafaelgreca/scratchml/blob/main/docs/AUTHORS.md).
