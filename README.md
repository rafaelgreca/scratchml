# ScratchML

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
    - [ ] Softmax
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
    - [ ] Classification Metrics
        - [x] Accuracy
        - [x] Precision
        - [x] Recall
        - [x] F1-Score
        - [X] Confusion Matrix
        - [ ] AUC Score
- [X] Encoders
    - [X] One-hot encoding
    - [X] Label encoding
- [X] Splitters
    - [X] KFold
    - [X] Stratify KFold
    - [X] Simple data split
- [ ] Models
    - [x] Linear Regression
    - [x] Logistic Regression
    - [ ] SVM
    - [ ] KNN
    - [ ] Decision Tree
    - [ ] Perceptron
    - [ ] MLP

Features:

- [X] Extend classification metrics to multi-class output
- [ ] Run models on GPU and/or multiple CPUs
- [ ] Create examples folder
- [ ] Upgrade Linear and Regression models to use more than one loss function and metric
- [ ] Make a type and shape verification on the Label Encoder and One Hot Encoder
- [ ] Improving testing