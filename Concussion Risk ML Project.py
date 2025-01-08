import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Finds long file path, rip
file_path = r"C:\Users\思庭\AppData\Local\Programs\Python\Python313\dataverse_files tbi Dec 2024 ML Project\Cleaned_Data.xlsx"


# Loads the Excel file
df = pd.read_excel(file_path, sheet_name="Sheet1")

# Print available sheet names
#print("Sheets in the file:", df.keys())

# Loads the first sheet
first_sheet_name = list(df.keys())[0]
df_main = df[first_sheet_name]

### Display basic info
##print("\nDataset Overview:")
##print(df_main.info())  
##print(df_main.head())


# Save as Excel for external analysis/cleaned dataset to post on Github
##df_main.to_csv("imported_data.csv", index=False)
##print("Data saved as 'imported_data.csv'")

# Checks for duplicated values. Returns 0, which is good
#print(df.duplicated().sum())  
#print(df.dtypes)

# Identify columns with incorrect data types. Everything is int64
#object_cols = df.select_dtypes(include=['object']).columns
#print("Columns stored as object:\n", object_cols)

# SHould get rid of spaces in columns due to running into a column name error for dependent variables
df.columns = df.columns.str.strip()

# Define dependent variables
y_pcs = df["PCS Symptom Severity (132)"] #132 is the most severe Post Concussion Symptom scrore
#y_mfq = df["MFQ 66"]                     #66 is the most severe Mood and Feelings Questionaire. No longer considering MFQ

# Drop Participant ID, target variables (PCS Total Symptom Severity, MFQ Total Score), and individual PCS/MFQ items
X = df.drop(columns=[
    "Participant ID",
    "PCS Symptom Frequency (22)",     # How many PCS symptoms do they have
    "Concussion History",             # Not statistically significant. Odd given multiple concussions lead to worse ones, although they could be far enough apart so they don't compound
    "Age (Years)",                    # Although it improved adj r^2 by 0.013, this had a correlation >10 so I still chose to remove
    "Sport",                          # Made regression worse
    "Concussion Number",              # Made regression worse             
    "PCS Symptom Severity (132)",     # Target variable
    "MFQ 66",                         # Target variable
] + [f"PCS {i}" for i in range(1, 23)]  # Removing PCS1-22
  + [f"MFQ {i}" for i in range(1, 34)]  # Removing MFQ1-33
)


# Train-test split for PCS model
X_train_pcs, X_test_pcs, y_train_pcs, y_test_pcs = train_test_split(X, y_pcs, test_size=0.2, random_state=42)

# Train-test split for MFQ model
#X_train_mfq, X_test_mfq, y_train_mfq, y_test_mfq = train_test_split(X, y_mfq, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression

# Initialize models
linear_pcs = LinearRegression()
#linear_mfq = LinearRegression()

# Train model on PCS data 
linear_pcs.fit(X_train_pcs, y_train_pcs)

# Train model on MFQ data
#linear_mfq.fit(X_train_mfq, y_train_mfq)

### Code to plot histograms of data to see if there's anything interesting. There was not.
##columns_to_plot = df.columns[:10]  # Gets the first 10 column names
##df[columns_to_plot].hist(bins=30, figsize=(16, 12))
##plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Predict on test set
y_pred_pcs = linear_pcs.predict(X_test_pcs)
#y_pred_mfq = linear_mfq.predict(X_test_mfq)

# Evaluate PCS Model
mae_pcs = mean_absolute_error(y_test_pcs, y_pred_pcs)
rmse_pcs = np.sqrt(mean_squared_error(y_test_pcs, y_pred_pcs))
r2_pcs = r2_score(y_test_pcs, y_pred_pcs)

# Evaluate MFQ Model
##mae_mfq = mean_absolute_error(y_test_mfq, y_pred_mfq)
##rmse_mfq = np.sqrt(mean_squared_error(y_test_mfq, y_pred_mfq))
##r2_mfq = r2_score(y_test_mfq, y_pred_mfq)

# Print results
print(f"PCS Model - MAE: {mae_pcs:.2f}, RMSE: {rmse_pcs:.2f}, R²: {r2_pcs:.2f}")
#print(f"MFQ Model - MAE: {mae_mfq:.2f}, RMSE: {rmse_mfq:.2f}, R²: {r2_mfq:.2f}")

import statsmodels.api as sm

### Add a constant (intercept) for OLS analysis
##X_train_pcs_const = sm.add_constant(X_train_pcs)
##
### Fit OLS(p-values) model
##ols_model = sm.OLS(y_train_pcs, X_train_pcs_const).fit()
##
### Print summary
##print(ols_model.summary())
##
print("PCS Train R²:", linear_pcs.score(X_train_pcs, y_train_pcs)) # Prints PCS training r^2
print("PCS Test R²:", linear_pcs.score(X_test_pcs, y_test_pcs))    # Prints PCS testing r^2

##print("MFQ Train R²:", linear_mfq.score(X_train_pcs, y_train_pcs)) # Prints PCS training r^2
##print("MFQ Test R²:", linear_mfq.score(X_test_pcs, y_test_pcs))    # Prints PCS testing r^2



from statsmodels.stats.outliers_influence import variance_inflation_factor

# Compute VIF for each feature. Tests for multicolinearity.
##vif_data = pd.DataFrame()
##vif_data["Feature"] = X_train_pcs.columns
##vif_data["VIF"] = [variance_inflation_factor(X_train_pcs.values, i) for i in range(X_train_pcs.shape[1])]
##print(vif_data)
##
##from sklearn.linear_model import Ridge, Lasso
##from sklearn.model_selection import GridSearchCV

### Define the alphas (regularization strengths) to test
##alphas = [0.001, 0.01, 0.1, 1, 10, 100]
##
### Ridge Regression with GridSearchCV to find the best alpha
##ridge = Ridge()
##ridge_cv = GridSearchCV(ridge, param_grid={"alpha": alphas}, cv=5, scoring="r2")
##ridge_cv.fit(X_train_pcs, y_train_pcs)
##best_ridge = ridge_cv.best_estimator_
##
### Lasso Regression with GridSearchCV to find the best alpha
##lasso = Lasso(max_iter=10000)  # High max_iter to ensure convergence
##lasso_cv = GridSearchCV(lasso, param_grid={"alpha": [0.1, 1, 10, 50, 100]}, cv=5, scoring="r2")
##lasso_cv.fit(X_train_pcs, y_train_pcs)
##best_lasso = lasso_cv.best_estimator_
##
### Evaluate Ridge and Lasso on the test set
##ridge_pred = best_ridge.predict(X_test_pcs)
##lasso_pred = best_lasso.predict(X_test_pcs)
##
### Compute performance metrics
##ridge_r2 = r2_score(y_test_pcs, ridge_pred)
##ridge_mae = mean_absolute_error(y_test_pcs, ridge_pred)
##ridge_rmse = np.sqrt(mean_squared_error(y_test_pcs, ridge_pred))
##
##lasso_r2 = r2_score(y_test_pcs, lasso_pred)
##lasso_mae = mean_absolute_error(y_test_pcs, lasso_pred)
##lasso_rmse = np.sqrt(mean_squared_error(y_test_pcs, lasso_pred))
##
### Print results
##print(f"Best Ridge Alpha: {ridge_cv.best_params_['alpha']}")
##print(f"Ridge Model - MAE: {ridge_mae:.2f}, RMSE: {ridge_rmse:.2f}, R²: {ridge_r2:.2f}")
##
##print(f"Best Lasso Alpha: {lasso_cv.best_params_['alpha']}")
##print(f"Lasso Model - MAE: {lasso_mae:.2f}, RMSE: {lasso_rmse:.2f}, R²: {lasso_r2:.2f}")
##
### Check which features Lasso eliminated (set coefficients to zero)
##lasso_coefficients = pd.Series(best_lasso.coef_, index=X_train_pcs.columns)
##print("\nLasso eliminated the following features:")
##print(lasso_coefficients[lasso_coefficients == 0].index.tolist())


# Get the absolute values of the coefficients to compare them to each other for feature importance 
feature_importance = pd.Series(np.abs(linear_pcs.coef_), index=X_train_pcs.columns)

# Sort by importance
feature_importance = feature_importance.sort_values(ascending=False)

# Display feature importance
print("Feature Importance (Higher = More Impact on PCS Severity):")
print(feature_importance)


import seaborn as sns

### Compute correlation matrix. Checking to see why age had such high multi collinearity since I didn't see why conceptually. Could not find an answer.
##correlation_matrix = X_train_pcs.corr()
##
### Get correlation values for "Age (Years)" and sort them
##age_correlations = correlation_matrix["Age (Years)"].sort_values(ascending=False)
##
### Display the highest correlations
##print("\nCorrelation of 'Age (Years)' with other variables:")
##print(age_correlations)
##
### Plot heatmap for visualization
##plt.figure(figsize=(8, 6))
##sns.heatmap(correlation_matrix, cmap="coolwarm", annot=False)
##plt.title("Feature Correlation Heatmap")
##plt.show()


### Compute the Pearson correlation between PCS and MFQ scores to help deceide to drop MFQ model
##correlation = df["PCS Symptom Severity (132)"].corr(df["MFQ 66"])
##
### Print the result
##print(f"Correlation between PCS and MFQ: {correlation:.4f}")


# Residual Plot (Checking Model Assumptions)
plt.figure(figsize=(6, 5))
residuals = y_test_pcs - y_pred_pcs
sns.scatterplot(x=y_pred_pcs, y=residuals, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.title("Residual Plot")
plt.xlabel("Predicted PCS Severity")
plt.ylabel("Residuals")
plt.show()

# Predicted vs. Actual Plot
plt.figure(figsize=(6, 5))
sns.scatterplot(x=y_test_pcs, y=y_pred_pcs, alpha=0.6)
plt.plot([y_test_pcs.min(), y_test_pcs.max()], [y_test_pcs.min(), y_test_pcs.max()], 'r--')  # 45-degree reference line
plt.title("Predicted vs. Actual PCS Severity")
plt.xlabel("Actual PCS Severity")
plt.ylabel("Predicted PCS Severity")
plt.show()

# Feature Importance Bar Chart
plt.figure(figsize=(8, 5))
sns.barplot(x=feature_importance.values, y=feature_importance.index, palette="viridis")
plt.title("Feature Importance (Higher = More Impact on PCS Severity)")
plt.xlabel("Importance Score (Absolute Regression Coefficients)")
plt.ylabel("Feature")
plt.show()

import numpy as np

# Apply log transformation to the target variable
y_train_log = np.log1p(y_train_pcs)  # log1p(x) = log(x+1) to handle zeros
y_test_log = np.log1p(y_test_pcs)

# Train the regression again
linear_pcs.fit(X_train_pcs, y_train_log)

# Predict and transform back
y_pred_log = linear_pcs.predict(X_test_pcs)
y_pred_pcs = np.expm1(y_pred_log)  # Reverse the log transformation





    

