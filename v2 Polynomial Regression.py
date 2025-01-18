import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import sys
import os
from cross_validation_function import leave_one_out_cv
from sklearn.model_selection import learning_curve

# Load Data
file_path = r"C:\Users\思庭\AppData\Local\Programs\Python\Python313\dataverse_files tbi Dec 2024 ML Project\Cleaned_Data.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")


####################### Residual Plots Code to Determine If Feature Engineering Is Nescessary ####################

##features = ["Age (Years)", "MFQ 66", "Sport"]  
##target = "PCS Symptom Severity (132)"
##
### Loop through each feature to check residuals
##for feature in features:
##    plt.figure(figsize=(6, 4))
##    
##    # Fit a simple linear regression
##    X = df[[feature]]
##    y = df[target]
##    model = LinearRegression().fit(X, y)
##    y_pred = model.predict(X)
##    
##    # Compute residuals
##    residuals = y - y_pred
##    
##    # Plot residuals
##    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
##    plt.axhline(y=0, color='red', linestyle='--')
##    plt.xlabel(f"Predicted {target}")
##    plt.ylabel("Residuals")
##    plt.title(f"Residual Plot: {feature}")
##    plt.show()

########################## LOOCV Testing #######################################

sys.path.append(os.path.abspath(r"C:\Users\思庭\AppData\Local\Programs\Python\Python313\dataverse_files tbi Dec 2024 ML Project"))


####################### Polynomial Regression with Transformed Feature Engineering ################

print (df.columns)
feature_list = ["MFQ Cut off",
                "Learning Disability",
                "Aggregate Medical History",
                "Anxiety Diagnosis",
                "Depression Diagnosis",
                "Anxiety Symptoms",
                "# of Prior Depressive Episodes",
                "Sex",
                "Prior Depressive Episode(s) Y/N",
                "Age (Years)"] 
target_variable = "PCS Symptom Severity (132)"

# Extract Features and Target Variable
X = df[feature_list]
y = df[target_variable]

# Convert Data to NumPy Format
X = X.to_numpy()
y = y.to_numpy()

# Perform LOOCV (Leave-One-Out Cross-Validation)
model = LinearRegression()  # Initialize the model
results = leave_one_out_cv(model, X, y)  # Run LOOCV
print(f"LOOCV Results: Average RMSE = {results['Average RMSE']:.4f}")

# Split Data into Train & Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Polynomial Transformation (With Multiple Features)
degree = 4  
poly = PolynomialFeatures(degree=degree, include_bias=False)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

# Train Polynomial Regression Model
model.fit(X_poly_train, y_train)

# Make Predictions
y_train_pred = model.predict(X_poly_train)
y_test_pred = model.predict(X_poly_test)

# Compute RMSE (Root Mean Squared Error)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

### Compute R² Score
##train_r2 = r2_score(y_train, y_train_pred)
##test_r2 = r2_score(y_test, y_test_pred)

# Print Evaluation Metrics
print(f"Polynomial Regression (Degree={degree}) Evaluation Metrics:")
print(f"Train RMSE: {train_rmse:.2f}, Test RMSE: {test_rmse:.2f}")
##print(f"Train R² Score: {train_r2:.4f}, Test R² Score: {test_r2:.4f}")


# Compute Residuals
train_residuals = y_train - y_train_pred
test_residuals = y_test - y_test_pred
# Plot 1: Residuals Plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_train_pred, train_residuals, color='blue', alpha=0.5, label="Train Residuals")
plt.scatter(y_test_pred, test_residuals, color='red', alpha=0.5, label="Test Residuals")
plt.axhline(y=0, color='black', linestyle='dashed')  # Horizontal line at zero
plt.xlabel("Predicted PCS Severity")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residuals Plot")
plt.legend()

# Plot 2: Predicted vs. Actual PCS Severity
plt.subplot(1, 2, 2)
plt.scatter(y_train, y_train_pred, color='blue', alpha=0.5, label="Train Data")
plt.scatter(y_test, y_test_pred, color='red', alpha=0.5, label="Test Data")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'g--', label="Perfect Fit")  # Diagonal reference line
plt.xlabel("Actual PCS Severity")
plt.ylabel("Predicted PCS Severity")
plt.title("Predicted vs Actual PCS Severity")
plt.legend()

plt.tight_layout()
plt.show()

# Plot 3: Learning Curve
train_sizes = np.linspace(0.1, 1.0, 10)  # Define fractions of the training set
train_rmse_list = []
test_rmse_list = []

for train_size in train_sizes:
    # Sample a subset of the training data
    subset_size = int(train_size * len(X_train))
    X_subset = X_train[:subset_size]
    y_subset = y_train[:subset_size]

    # Polynomial Transformation for Subset
    X_poly_subset = poly.fit_transform(X_subset)
    X_poly_full_train = poly.transform(X_train)
    X_poly_test = poly.transform(X_test)

    # Train the Model on Subset
    model.fit(X_poly_subset, y_subset)

    # Predict on Full Training Set and Test Set
    y_train_pred = model.predict(X_poly_full_train)
    y_test_pred = model.predict(X_poly_test)

    # Calculate RMSE
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    # Append RMSE to Lists
    train_rmse_list.append(train_rmse)
    test_rmse_list.append(test_rmse)

# Plot the Learning Curve
plt.figure(figsize=(8, 6))
plt.plot(train_sizes * len(X_train), train_rmse_list, label="Train RMSE", marker='o')
plt.plot(train_sizes * len(X_train), test_rmse_list, label="Test RMSE", marker='o')
plt.xlabel("Training Set Size")
plt.ylabel("RMSE")
plt.title(f"Learning Curve (Polynomial Regression, Degree={degree})")
plt.legend()
plt.grid()
plt.show()



########################## Code For Plots From Weekend Tasks ############################################
##
##file_path = r"C:\Users\思庭\AppData\Local\Programs\Python\Python313\dataverse_files tbi Dec 2024 ML Project\Cleaned_Data.xlsx"
##df = pd.read_excel(file_path)
##
##print (df.columns)
##
### Define categorical and target variables
##target_variable = "PCS Symptom Severity (132)"  
##categorical_variables = ["Sex", "Sport"]
##
### Step 1: Define Age Groups (Younger Teens vs. Older Teens)
##df["Age Group"] = pd.cut(df["Age (Years)"], bins=[0, 14, 19], labels=["Younger Teens", "Older Teens"])
##
### Step 2: Generate Scatter Plots for Numerical Features vs Target
##numerical_features = ["Age (Years)", "MFQ 66"]  
##
##for feature in numerical_features:
##    for cat_var in categorical_variables + ["Age Group"]:
##        plt.figure(figsize=(8, 5))
##        sns.scatterplot(data=df, x=feature, y=target_variable, hue=df["Sport"], palette="husl", alpha=0.7)
##
##        plt.title(f"{feature} vs {target_variable} (Color-coded by {cat_var})")
##        plt.xlabel(feature)
##        plt.ylabel(target_variable)
##        plt.legend(title=cat_var)
##       plt.show()

