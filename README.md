<a id="readme-top"></a>

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
        <img src="https://img.shields.io/badge/version-10.1.1-orange.svg?color=greeb&style=for-the-badge" /></a>
    <a href="https://github.com/rafaelgreca/scratchml/blob/main/LICENSE" alt="License">
        <img src="https://img.shields.io/badge/license-MIT-blue?color=greeb&style=for-the-badge" /></a>

</p>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#scratchml">About The Project</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#features">Features</a></li>
    <li><a href="#examples">Examples</a></li>
    <li><a href="#running-tests">Running Tests</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#authors">Authors</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

# ScratchML

With the goal of creating user-friendly, simple, and easy-to-use implementations for both beginners and enthusiasts, a Python library called ScratchML was developed to build the most basic Machine Learning (ML) models from scratch (using only Numpy).

What sets this library apart from other GitHub codes and libraries that also implement ML models from scratch is:

* well-organized implementations that make use of Object-oriented Programming (OOP) concepts
* straightforward and user-friendly implementations that help novices fully comprehend the concept behind the fundamental algorithms
* Continuous Integration (CI) that includes unit testing cases to ensure that the implementations are functioning as intended
* comparison of the variables and results, to the extent possible, with Scikit-Learn's implementation
* implementations for both binary and multiclass classifications
* does not cover only ML algorithms but also known metrics for classification and regression tasks, the most-used preprocessing steps (such as applying encoders, scalers, and splitting the data into different sets), and more!

Disclaimer: The goal of this library is to provide code that is simpler, easier to understand, and more approachable for artificial intelligence enthusiasts and beginners who want to contribute to an open-source repository or who want to learn more about how algorithms operate. It is not meant to replace existing libraries that are better, more optimized, and have a wider variety of implemented algorithms (such as scikit-learn, PyTorch, Keras, and Tensorflow).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Installation

To install this package, first clone the repository to the directory of your choice using the following command:

```bash
git clone https://github.com/rafaelgreca/scratchml.git
```

Use the following command to install the pre-commit package manager:

```bash
pip install pre-commit
```

Activate pre-commit using the following command:

```bash
pre-commit install
```

### Using Virtual Environment

Create a virtual environment (ideally using conda) and install the requirements with the following command:

```bash
conda create --name scratchml python=3.11.9
conda activate scratchml
pip install -r requirements/requirements.txt
```

### (RECOMMENDED) Using Docker

Build the Docker image using the following command:

```bash
sudo docker build -f Dockerfile -t scratchml . --no-cache
```

Run the Docker container using the following command:

```bash
sudo docker run -d -p 8000:5000 --name scratchml scratchml
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Features

Activation functions:

- ELU (added in version 8.1.0)
- Leaky ReLU (added in version 8.1.0)
- Linear (added in version 8.1.0)
- ReLU (added in version 8.1.0)
- SELU (added in version 8.1.0)
- Sigmoid (added in version 8.1.0)
- Softmax (added in version 8.1.0)
- SoftPlus (added in version 8.1.0)
- TanH (added in version 8.1.0)

Algorithms:

- Decision Tree Classifier (added in version 5.0.0) and Decision Tree Regressor (added in version 6.0.0)
- KMeans (added in version 4.0.0)
- KNN Classifier and KNN Regressor (added in version 2.0.0)
- Linear Regression (added in version 1.0.0)
- Logistic Regression (added in version 1.0.0)
- MLP Classifier (added in version 9.0.0) and MLP Regressor (added in version 10.0.0)
- Guassian Naive Bayes (added in version 3.0.0)
- Perceptron (added in version 4.0.0)
- PCA (added in version 8.0.0)
- Random Forest Classifier and Random Forest Regressor (added in version 7.0.0)
- Support Vector Classifier** and Support Vector Regressor (added in version 10.0.0)

** only available for binary classification at the moment

Data split functions:

- KFold (added in version 1.0.0)
- Split into Batches (added in version 9.0.0)
- Stratify KFold (added in version 1.0.0)
- Train Test Split (added in version 1.0.0)

Distance metrics:

- Chebyshev (added in version 2.0.0)
- Euclidean (added in version 2.0.0)
- Manhattan (added in version 2.0.0)
- Minkowski (added in version 2.0.0)

Encoders:

- Label Encoding (added in version 1.0.0)
- One-hot Encoding (added in version 1.0.0)

Kernels:

- Linear (added in version 10.0.0)
- Polynomial (added in version 10.0.0)
- RBF (added in version 10.0.0)

Loss functions:

- Binary Cross Entropy (added in version 9.0.0)
- Cross Entropy (added in version 9.0.0)

Metrics:

- Accuracy (added in version 1.0.0)
- Confusion Matrix (added in version 1.0.0)
- F1 Score (added in version 1.0.0)
- False Positive Rate (added in version 1.0.0)
- Max Error (added in version 1.0.0)
- Mean Absolute Error (added in version 1.0.0)
- Mean Absolute Percentage Error (added in version 1.0.0)
- Mean Squared Error (added in version 1.0.0)
- Mean Squared Logarithmic Error (added in version 1.0.0)
- Median Absolute Error (added in version 1.0.0)
- Precision (added in version 1.0.0)
- R Squared (added in version 1.0.0)
- Recall (added in version 1.0.0)
- ROC AUC Score (added in version 2.1.0)
- Root Mean Squared Error (added in version 1.0.0)
- True Positive Rate (added in version 1.0.0)

Regularization functions:

- L1 (added in version 1.0.0)
- L2 (added in version 1.0.0)

Scalers:

- Standard Scaler (added in version 1.0.0)
- Min Max Scaler (added in version 1.0.0)

## Examples

Check the `examples` folder to see some examples of how to use each functionality of this library.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Running Tests

### Locally

Run the following command on the root folder:

```bash
python3 -m unittest discover -p 'test_*.py'
```

### (RECOMMENDED) Using Docker

Build the Docker image using the following command:

```bash
sudo docker build -f test.Dockerfile -t test_scratchml . --no-cache
```

Run the Docker container using the following command:

```bash
sudo docker run -d -p 8001:5000 --name test_scratchml test_scratchml
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Roadmap

Check the `examples` folder to see the roadmap, some known issues, and ideas for enhancing the code.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contributing

Contributions are what makes the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated. If you have a suggestion that would make this better, please read carefully the [Contributing Guide](https://github.com/rafaelgreca/scratchml/blob/main/docs/CONTRIBUTING.md) and the [Code of Conduct](https://github.com/rafaelgreca/scratchml/blob/main/docs/CODE_OF_CONDUCT.md) before contributing.

If you agree with the [Code of Conduct](https://github.com/rafaelgreca/scratchml/blob/main/docs/CODE_OF_CONDUCT.md), read carefully the [Contributing Guide](https://github.com/rafaelgreca/scratchml/blob/main/docs/CONTRIBUTING.md), and still want to contribute, here's a step-by-step of how you contribute (you can also simply open an issue if you don't want to code anything):

1. Fork the project
2. Create a branch in your forked repository using a `feature` tag if you are implementing a new feature or a `bugfix` tag if you are fixing a issue/bug (examples: `git checkout -b feature/AmazingFeature` or `git checkout -b bugfix/FixingBugX`)
3. Commit your changes (`git commit -m <DETAILED_DESCRIPTION>`)
4. Push to the branch (`git push origin <YOUR_BRANCH_NAME>`)
5. Open a pull request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## License

Distributed under the [MIT](https://choosealicense.com/licenses/mit/) License. See LICENSE for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Authors

A huge shoutout to everyone who contributed to the success of the project. [Check everyone here!](https://github.com/rafaelgreca/scratchml/blob/main/docs/AUTHORS.md).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Acknowledgments

We would like to thank all these amazing materials and repositories for their amazing work, which indirectly contributed to or inspired us to create this project.

- [REPOSITORY] [SKADI by Douglas Oliveira](https://github.com/Dellonath/SKADI/)
- [REPOSITORY] [ML From Scratch by Erik Linder-Norén](https://github.com/eriklindernoren/ML-From-Scratch)
- [REPOSITORY] [Machine Learning from Scratch by AssemblyAI](https://github.com/AssemblyAI-Community/Machine-Learning-From-Scratch)
- [COURSE] [Machine Learning Specialization by Andrew Ng](https://www.coursera.org/specializations/machine-learning-introduction)
- [COURSE] [Machine Learning From Scratch by AssemblyAI](https://www.youtube.com/watch?v=p1hGz0w_OCo&list=PLcWfeUsAys2k_xub3mHks85sBHZvg24Jd)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
