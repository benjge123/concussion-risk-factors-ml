import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import resample
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor


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
#print(f"PCS Model - MAE: {mae_pcs:.2f}, RMSE: {rmse_pcs:.2f}, R²: {r2_pcs:.2f}")
#print(f"MFQ Model - MAE: {mae_mfq:.2f}, RMSE: {rmse_mfq:.2f}, R²: {r2_mfq:.2f}")



### Add a constant (intercept) for OLS analysis
##X_train_pcs_const = sm.add_constant(X_train_pcs)
##
### Fit OLS(p-values) model
##ols_model = sm.OLS(y_train_pcs, X_train_pcs_const).fit()
##
### Print summary
##print(ols_model.summary())
##
##print("PCS Train R²:", linear_pcs.score(X_train_pcs, y_train_pcs)) # Prints PCS training r^2
##print("PCS Test R²:", linear_pcs.score(X_test_pcs, y_test_pcs))    # Prints PCS testing r^2

##print("MFQ Train R²:", linear_mfq.score(X_train_pcs, y_train_pcs)) # Prints PCS training r^2
##print("MFQ Test R²:", linear_mfq.score(X_test_pcs, y_test_pcs))    # Prints PCS testing r^2





# Compute VIF for each feature. Tests for multicolinearity.
##vif_data = pd.DataFrame()
##vif_data["Feature"] = X_train_pcs.columns
##vif_data["VIF"] = [variance_inflation_factor(X_train_pcs.values, i) for i in range(X_train_pcs.shape[1])]
##print(vif_data)
##

### Start of Ridege/Lasso Regression
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


### Get the absolute values of the coefficients to compare them to each other for feature importance 
##feature_importance = pd.Series(np.abs(linear_pcs.coef_), index=X_train_pcs.columns)
##
### Sort by importance
##feature_importance = feature_importance.sort_values(ascending=False)
##
### Display feature importance
##print("Feature Importance (Higher = More Impact on PCS Severity):")
##print(feature_importance)




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


### Residual Plot (Checking Model Assumptions)
##plt.figure(figsize=(6, 5))
##residuals = y_test_pcs - y_pred_pcs
##sns.scatterplot(x=y_pred_pcs, y=residuals, alpha=0.6)
##plt.axhline(y=0, color='red', linestyle='--')
##plt.title("Residual Plot")
##plt.xlabel("Predicted PCS Severity")
##plt.ylabel("Residuals")
##plt.show()
##
### Predicted vs. Actual Plot
##plt.figure(figsize=(6, 5))
##sns.scatterplot(x=y_test_pcs, y=y_pred_pcs, alpha=0.6)
##plt.plot([y_test_pcs.min(), y_test_pcs.max()], [y_test_pcs.min(), y_test_pcs.max()], 'r--')  # 45-degree reference line
##plt.title("Predicted vs. Actual PCS Severity")
##plt.xlabel("Actual PCS Severity")
##plt.ylabel("Predicted PCS Severity")
##plt.show()
##
### Feature Importance Bar Chart
##plt.figure(figsize=(8, 5))
##sns.barplot(x=feature_importance.values, y=feature_importance.index, palette="viridis")
##plt.title("Feature Importance (Higher = More Impact on PCS Severity)")
##plt.xlabel("Importance Score (Absolute Regression Coefficients)")
##plt.ylabel("Feature")
##plt.show()
##
##
##
### Plot the distribution of PCS Severity Scores. This shows 43 have a score of 0 and very few have scores past 60.
##plt.figure(figsize=(8, 5))
##sns.histplot(y_train_pcs, bins=20, kde=True)
##plt.title("PCS Severity Score Distribution (Before Resampling)")
##plt.xlabel("PCS Severity Score")
##plt.ylabel("Frequency")
##plt.show()
##

##
### Define severity threshold
##severe_threshold = 40
##
### Split severe and non-severe cases
##severe_cases = X_train_pcs[y_train_pcs >= severe_threshold]
##severe_labels = y_train_pcs[y_train_pcs >= severe_threshold]
##
##non_severe_cases = X_train_pcs[y_train_pcs < severe_threshold]
##non_severe_labels = y_train_pcs[y_train_pcs < severe_threshold]
##
### Oversample severe cases using bootstrapping
##severe_cases_resampled, severe_labels_resampled = resample(
##    severe_cases, severe_labels,
##    replace=True,        # Allow duplicate samples (bootstrapping)
##    n_samples=len(non_severe_labels) // 2,  # Upsample to 50% of non-severe cases
##    random_state=42
##)
##
### Concatenate resampled severe cases with original non-severe cases
##X_train_resampled = pd.concat([non_severe_cases, severe_cases_resampled])
##y_train_resampled = pd.concat([non_severe_labels, severe_labels_resampled])
##
### --- Step 2: Plot the Final Resampled Distribution ---
##plt.figure(figsize=(8, 5))
##sns.histplot(y_train_resampled, bins=20, kde=True)
##plt.title("PCS Severity Score Distribution (After Bootstrap Resampling)")
##plt.xlabel("PCS Severity Score")
##plt.ylabel("Frequency")
##plt.show()



# Function to train a Decision Tree model
def train_decision_tree(X_train, y_train, max_depth=None, min_samples_split=2, min_samples_leaf=1):
    """
    Trains a Decision Tree Regressor and returns the trained model.
    
    Parameters:
        X_train (DataFrame): Training feature set.
        y_train (Series): Training target variable.
        max_depth (int, optional): Maximum depth of the tree. Defaults to None (fully grown tree).
        min_samples_split (int, optional): Minimum samples required to split a node. Defaults to 2.
        min_samples_leaf (int, optional): Minimum samples required in a leaf node. Defaults to 1.

    Returns:
        model (DecisionTreeRegressor): Trained Decision Tree model.
    """
    model = DecisionTreeRegressor(
        max_depth=max_depth, 
        min_samples_split=min_samples_split, 
        min_samples_leaf=min_samples_leaf, 
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

# Function to train Decision Tree with Cross validation
def cross_validate_decision_tree(X, y, max_depth=3, min_samples_split=10, min_samples_leaf=5, cv=5):
    """Performs K-Fold cross-validation on a Decision Tree model."""
    
    model = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    
    # Perform K-Fold Cross-Validation (default cv=5)
    scores = cross_val_score(model, X, y, cv=cv, scoring="r2")  # R² score for each fold
    
    print(f"Cross-Validation R² Scores: {scores}")
    print(f"Mean R²: {np.mean(scores):.2f}")
    print(f"Standard Deviation of R²: {np.std(scores):.2f}")
    
    return scores



# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    """
    Evaluates a trained model and prints performance metrics.
    
    Parameters:
        model: Trained model (DecisionTreeRegressor or other regressors).
        X_test (DataFrame): Test feature set.
        y_test (Series): Actual target values.
    
    Returns:
        metrics (dict): Dictionary of evaluation metrics.
    """
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

##    print(f"Model Evaluation Metrics:")
##    print(f"MAE: {mae:.2f}")
##    print(f"RMSE: {rmse:.2f}")
##    print(f"R²: {r2:.2f}")

    return {"MAE": mae, "RMSE": rmse, "R²": r2}

# Function to plot feature importance
def plot_feature_importance(model, feature_names):
    """
    Plots the feature importance of a trained Decision Tree model.
    
    Parameters:
        model: Trained DecisionTreeRegressor model.
        feature_names (list): List of feature names from the dataset.
    """
    
    importance = model.feature_importances_
    sorted_idx = importance.argsort()[::-1]  # Sort in descending order

##    plt.figure(figsize=(8, 5))
##    sns.barplot(x=importance[sorted_idx], y=[feature_names[i] for i in sorted_idx], palette="viridis")
##    plt.title("Feature Importance - Decision Tree")
##    plt.xlabel("Importance Score")
##    plt.ylabel("Feature")
##    plt.show()


### Train Decision Tree Model
##dt_model = train_decision_tree(X_train_pcs, y_train_pcs, max_depth=5, min_samples_split=5, min_samples_leaf=2)
##
### Evaluate Decision Tree Model
##dt_metrics = evaluate_model(dt_model, X_test_pcs, y_test_pcs)
##
### Plot Feature Importance
##plot_feature_importance(dt_model, X_train_pcs.columns)

##def tune_decision_tree(X, y):
##    """Finds the best Decision Tree parameters using GridSearchCV."""
##    
##    param_grid = {
##        "max_depth": [3, 4, 5, 6], 
##        "min_samples_split": [5, 10, 15],  
##        "min_samples_leaf": [2, 5, 10]  
##    }
##    
##    model = DecisionTreeRegressor(random_state=42)
##    grid_search = GridSearchCV(model, param_grid, cv=5, scoring="r2", n_jobs=-1)
##    grid_search.fit(X, y)
##    
##    print(f"Best Parameters: {grid_search.best_params_}")
##    return grid_search.best_estimator_
##
### Run hyperparameter tuning
##best_tree = tune_decision_tree(X_train_pcs, y_train_pcs)
##
### Evaluate the tuned model
##evaluate_model(best_tree, X_test_pcs, y_test_pcs)
##
##
# Run Cross-Validation Before Training Final Model
print("\nRunning Cross-Validation for Decision Tree:")
cross_validate_decision_tree(X_train_pcs, y_train_pcs, cv=5)

# Train Decision Tree Model (After Cross-Validation)
dt_model = train_decision_tree(X_train_pcs, y_train_pcs, max_depth=5, min_samples_split=5, min_samples_leaf=2)

# Evaluate Decision Tree Model on Test Set
dt_metrics = evaluate_model(dt_model, X_test_pcs, y_test_pcs)

# Plot Feature Importance
plot_feature_importance(dt_model, X_train_pcs.columns)


def train_random_forest(X_train, y_train, max_depth=3, min_samples_split=5, min_samples_leaf=10, n_estimators=100):
    """
    Trains a Random Forest Regressor and returns the trained model.

    Parameters:
        X_train (DataFrame): Training feature set.
        y_train (Series): Training target variable.
        max_depth (int): Maximum depth of each tree.
        min_samples_split (int): Minimum samples required to split a node.
        min_samples_leaf (int): Minimum samples required in a leaf node.
        n_estimators (int): Number of trees in the forest.

    Returns:
        model (RandomForestRegressor): Trained Random Forest model.
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1  # Uses all available processors
    )
    model.fit(X_train, y_train)
    return model

# Train the Random Forest Model
rf_model = train_random_forest(X_train_pcs, y_train_pcs)

# Evaluate the Random Forest Model on Test Data
rf_metrics = evaluate_model(rf_model, X_test_pcs, y_test_pcs)

from sklearn.model_selection import cross_val_score

# Run Cross-Validation on Random Forest
rf_scores = cross_val_score(rf_model, X_train_pcs, y_train_pcs, cv=5, scoring="r2")

print(f"Random Forest Cross-Validation R² Scores: {rf_scores}")
print(f"Mean R²: {np.mean(rf_scores):.2f}")
print(f"Standard Deviation of R²: {np.std(rf_scores):.2f}")





