import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

# ============================== DATA LOADING ============================== #

def load_data(file_path: str, sheet_name: str = "Sheet1") -> pd.DataFrame:
    """Loads an Excel file, strips spaces from column names, and returns the main DataFrame."""
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df.columns = df.columns.str.strip()  # Remove spaces in column names
    return df

# ============================== DATA PREPROCESSING ============================== #

def preprocess_data(df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    """Prepares the dataset by removing unnecessary columns and defining the target variable."""
    y_pcs = df["PCS Symptom Severity (132)"]  # Define dependent variable

    # Drop unwanted columns
    X = df.drop(columns=[
        "Participant ID",
        "PCS Symptom Frequency (22)",
        "Concussion History",
        "Age (Years)", "Sport",
        "Concussion Number",
        "PCS Symptom Severity (132)",
        "MFQ 66"
    ] + [f"PCS {i}" for i in range(1, 23)]  # Removing PCS1-22
      + [f"MFQ {i}" for i in range(1, 34)]  # Removing MFQ1-33
    )
    
    return X, y_pcs

# ============================== DATA SPLITTING ============================== #

def split_data(X: pd.DataFrame, y: pd.Series, test_size=0.2, random_state=42):
    """Splits the dataset into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# ============================== MODEL TRAINING ============================== #

def train_linear_model(X_train, y_train):
    """Trains a Linear Regression model and returns the fitted model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# ============================== MODEL EVALUATION ============================== #

def evaluate_model(model, X_test, y_test):
    """Evaluates a trained model and prints performance metrics."""
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")
    return y_pred, mae, rmse, r2

# ============================== VIF CALCULATION (MULTICOLLINEARITY) ============================== #

def compute_vif(X: pd.DataFrame):
    """Computes Variance Inflation Factor (VIF) for each feature to detect multicollinearity."""
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

# ============================== FEATURE IMPORTANCE ============================== #

def feature_importance(model, X_train):
    """Returns feature importance based on absolute regression coefficients."""
    importance = pd.Series(np.abs(model.coef_), index=X_train.columns)
    return importance.sort_values(ascending=False)

# ============================== RIDGE AND LASSO REGRESSION ============================== #

def train_ridge_lasso(X_train, y_train):
    """Finds the best alpha using GridSearchCV for Ridge and Lasso regression."""
    alphas = [0.001, 0.01, 0.1, 1, 10, 100]
    
    # Ridge Regression
    ridge = Ridge()
    ridge_cv = GridSearchCV(ridge, param_grid={"alpha": alphas}, cv=5, scoring="r2")
    ridge_cv.fit(X_train, y_train)
    best_ridge = ridge_cv.best_estimator_

    # Lasso Regression
    lasso = Lasso(max_iter=10000)
    lasso_cv = GridSearchCV(lasso, param_grid={"alpha": alphas}, cv=5, scoring="r2")
    lasso_cv.fit(X_train, y_train)
    best_lasso = lasso_cv.best_estimator_

    return best_ridge, best_lasso, ridge_cv.best_params_['alpha'], lasso_cv.best_params_['alpha']

# ============================== VISUALIZATIONS ============================== #

def plot_residuals(y_test, y_pred):
    """Plots residuals to check model assumptions."""
    plt.figure(figsize=(6, 5))
    residuals = y_test - y_pred
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title("Residual Plot")
    plt.xlabel("Predicted PCS Severity")
    plt.ylabel("Residuals")
    plt.show()

def plot_feature_importance(importance):
    """Plots feature importance as a bar chart."""
    plt.figure(figsize=(6, 5))
    sns.barplot(x=importance.values, y=importance.index, palette="viridis")
    plt.title("Feature Importance (Higher = More Impact on PCS Severity)")
    plt.xlabel("Importance Score (Absolute Regression Coefficients)")
    plt.ylabel("Feature")
    plt.tight_layout()  
    plt.show()

def plot_predicted_vs_actual(y_test, y_pred):
    """Plots predicted vs. actual values."""
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # 45-degree reference line
    plt.title("Predicted vs. Actual PCS Severity")
    plt.xlabel("Actual PCS Severity")
    plt.ylabel("Predicted PCS Severity")
    plt.show()

# ============================== MAIN SCRIPT ============================== #

if __name__ == "__main__":
    # File path
    file_path = r"C:\Users\思庭\AppData\Local\Programs\Python\Python313\dataverse_files tbi Dec 2024 ML Project\Cleaned_Data.xlsx"

    # Load and preprocess data
    df = load_data(file_path)
    X, y_pcs = preprocess_data(df)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y_pcs)

    # Train Linear Regression model
    linear_model = train_linear_model(X_train, y_train)

    # Evaluate Linear Regression model
    y_pred, mae, rmse, r2 = evaluate_model(linear_model, X_test, y_test)

    # Compute feature importance to rank risk factors
    importance = feature_importance(linear_model, X_train)
    
    # Compute VIF to test multicollinearity
    vif_data = compute_vif(X_train)
    print("\nVariance Inflation Factor (VIF) for each feature:")
    print(vif_data)

    # Visualizations
    plot_residuals(y_test, y_pred)
    plot_predicted_vs_actual(y_test, y_pred)
    plot_feature_importance(importance)
