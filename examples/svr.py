import numpy as np
from scratchml.models.svr  import BaseSVR

X_train = np.array([[1], [2], [3], [4], [5]])
y_train = np.array([2.1, 2.9, 3.7, 4.1, 5.2])

# Test dataset
X_test = np.array([[1.5], [2.5], [3.5], [4.5]])

# Initialize and fit the SVR model
svr_model = BaseSVR(kernel="linear", C=1.0, epsilon=0.1)
svr_model.fit(X_train, y_train)

# Make predictions on test data
predictions = svr_model.predict(X_test)

# Output the predictions
print("Test data:", X_test.flatten())
print("Predicted values:", predictions)

# Evaluate the model (R-squared score)
r2_score = svr_model.score(X_test, np.array([2.0, 3.0, 4.0, 5.0]), metric="r_squared")
print("R-squared score:", r2_score)