
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
    <a href="https://circleci.com/gh/rafaelgreca/scratchml/tree/master">
        <img src="https://img.shields.io/circleci/project/github/rafaelgreca/scratchml/main?color=greeb&style=for-the-badge" alt="build status"></a>
    <a href="https://coveralls.io/github/rafaelgreca/scratchml">
        <img src="https://img.shields.io/coveralls/github/rafaelgreca/scratchml?color=greeb&style=for-the-badge"
            alt="coverage"></a>
    <a alt="Version">
        <img src="https://img.shields.io/badge/version-2.1.0-orange.svg?color=greeb&style=for-the-badge" /></a>
    <a href="https://github.com/rafaelgreca/scratchml/blob/main/LICENSE" alt="License">
        <img src="https://img.shields.io/badge/license-MIT-blue?color=greeb&style=for-the-badge" /></a>

</p>

# ScratchML

A Python library called ScratchML was created to build the most fundamental Machine Learning models from scratch (using only Numpy), emphasizing producing user-friendly, straightforward, and easy-to-use implementations for novices and enthusiasts.

Disclaimer: This library is not intended to surpass those that already exist and which are better, more optimized and with more diversity of implemented algorithms (such as scikit-learn, PyTorch, Keras and Tensorflow), but rather to provide code that is easier to understand, simple and friendly for beginners and enthusiasts in the field of artificial intelligence who wish to gain a deeper understanding of how algorithms work.
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
python3 -m unittest discover
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
    - [x] StandardScaler
    - [x] MinMaxScaler
- [ ] Regularizations
    - [x] L1
    - [x] L2
    - [ ] Batch Normalization
- [ ] Activation functions
    - [x] Sigmoid
    - [ ] ReLU
    - [ ] Linear
    - [x] Softmax
    - [ ] TanH
    - [ ] Elu
    - [ ] Leaky ReLU
- [ ] Loss functions
    - [x] Binary Cross Entropy
- [ ] Metrics
    - [x] Regression Metrics
        - [x] Mean Squared Error (MSE)
        - [x] Root Mean Squared Error (RMSE)
        - [x] Mean Absolute Error (MAE)
        - [x] Median Absolute Error (MedAE)
        - [x] Mean Absolute Percentage Error (MAPE)
        - [x] Mean Squared Logarithmic Error (MSLE)
        - [x] Max Error (ME)
        - [x] R Squared (R2)
    - [x] Classification Metrics
        - [x] Accuracy
        - [x] Precision
        - [x] Recall
        - [x] F1-Score
        - [x] Confusion Matrix
        - [x] ROC AUC Score
- [x] Distances
    - [x] Euclidean
    - [x] Manhattan
    - [x] Chebyshev
    - [x] Minkowski
- [x] Encoders
    - [x] One-hot encoding
    - [x] Label encoding
- [x] Splitters
    - [x] KFold
    - [x] Stratify KFold
    - [x] Simple data split
- [ ] Models
    - [x] Linear Regression
    - [x] Logistic Regression
    - [ ] SVM
    - [x] KNN
    - [ ] Naive Bayes
    - [ ] Random Forest
    - [ ] Decision Tree
    - [ ] Perceptron
    - [ ] MLP

Features:

- [x] Extend classification metrics to multi-class output
- [ ] Run Linear Regression, Logistic Regression, and KNN on multiple CPUs using joblib
- [x] Create examples folder
- [x] Upgrade Linear and Regression models to use more than one loss function and metric
- [ ] Improving testing
- [x] Add verbose mode for Linear Regression and Logistic Regression
- [x] Fix Logistic Regression
- [ ] Update README
- [ ] Add visualization plots

Issues:

- [ ] Sometimes the Logistic Regression model gets stuck. This happens intermittently and was observed when wasn't a binary problem
- [ ] Optimize distance metrics and KNN looping functions


## Feedback

If you have any feedback, please feel free to create an issue pointing out whatever you want or reach out to me at rgvieira97@gmail.com


## Contributing

Contributions are what makes the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement". Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the [MIT](https://choosealicense.com/licenses/mit/) License. See LICENSE for more information.


## Authors

- [@rafaelgreca](https://www.github.com/rafaelgreca)

