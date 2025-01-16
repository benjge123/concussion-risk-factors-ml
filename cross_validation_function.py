# cross_validation_utils.py
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.linear_model import LinearRegression
import numpy as np

def leave_one_out_cv(model, X, y):
    """
    Perform Leave-One-Out Cross-Validation and calculate RMSE and R².
    
    Parameters:
        model: LinearRegression, DecisionTreeRegressor, RandomForestRegressor).
        X: Features 
        y: Target variable
        
    Returns:
        dict: A dictionary containing average RMSE and R² scores.
    """

    loo = LeaveOneOut()
    rmse_list = []
    r2_list = []

    # Perform LOOCV
    for train_index, test_index in loo.split(X):
        # Split data
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Predict on the test set
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Append results
        rmse_list.append(rmse)
        r2_list.append(r2)
    
    # Return average metrics
    return {
        "Average RMSE": np.mean(rmse_list),
        "Average R²": np.mean(r2_list)
    }

X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 6, 8])  # Perfect linear relationship: y = 2x

# Model
model = LinearRegression()

# Run LOOCV
results = leave_one_out_cv(model, X, y)

print("Test Results:")
print(f"Average RMSE: {results['Average RMSE']}")
print(f"Average R²: {results['Average R²']}")
