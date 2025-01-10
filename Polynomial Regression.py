import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load your dataset
df = pd.read_excel("Cleaned_Data.xlsx") 

# Select multiple features
print("Columns in dataset:", df.columns)  

# Define relevant features 
feature_list = ["Age (Years)",
                "MFQ Cut off",
                "Learning Disability",
                "Aggregate Medical History",
                "Anxiety Diagnosis",
                "Depression Diagnosis",
                "Anxiety Symptoms",
                "Prior Depressive Episode(s) Y/N",
                "Sex",
                "# of Prior Depressive Episodes"]  
X = df[feature_list].to_numpy()  # Ensure X is a NumPy array
y = df["PCS Symptom Severity (132)"]  # Target variable

# Ensure there are no missing values
df = df.dropna(subset=["PCS Symptom Severity (132)"] + feature_list)
X = df[feature_list]
y = df["PCS Symptom Severity (132)"]

# Reset index to avoid misalignment
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

# Split into Training and Testing Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Train Polynomial Regression Model
# Define degrees of polynomial to test
degrees = [1, 2,10]

# Initialize the figure for plotting
plt.figure(figsize=(20, 5))

for i, degree in enumerate(degrees):
    # Transform features into polynomial form (MULTIPLE FEATURES)
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    
    # Ensure X_train and X_test retain feature names
    X_train_df = pd.DataFrame(X_train, columns=feature_list)
    X_test_df = pd.DataFrame(X_test, columns=feature_list)
    
    X_poly_train = poly.fit_transform(X_train_df)
    X_poly_test = poly.transform(X_test_df)

    # Train the model
    model = LinearRegression()
    model.fit(X_poly_train, y_train)

    # Make Predictions
    y_pred_train = model.predict(X_poly_train)
    y_pred_test = model.predict(X_poly_test)

    # Calculate Errors
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)

    # Generate Predictions for Plotting (USE MULTIPLE FEATURES)
    X_range = np.zeros((100, len(feature_list)))  
    for j in range(len(feature_list)):  
        X_range[:, j] = np.linspace(X_train_df.iloc[:, j].min(), X_train_df.iloc[:, j].max(), 100)

    X_poly_range = poly.transform(X_range)
    y_range_pred = model.predict(X_poly_range)

    # Plot Results (Using the first feature for visualization)
    plt.subplot(1, len(degrees), i + 1)
    plt.scatter(X_train_df.iloc[:, 0], y_train, color='blue', alpha=0.5, label="Train Data")
    plt.scatter(X_test_df.iloc[:, 0], y_test, color='red', alpha=0.5, label="Test Data")
    plt.plot(X_range[:, 0], y_range_pred, color='green', label="Polynomial Fit")
    plt.title(f"Degree={degree}\nTrain MSE={train_mse:.2f}, Test MSE={test_mse:.2f}")
    plt.xlabel(f"{feature_list[0]} (First Feature)")
    plt.ylabel("PCS Severity")
    plt.legend()

# Show the Final Plot
plt.tight_layout()
plt.show()
