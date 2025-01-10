#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
##from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
##from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
#from sklearn.utils import resample
#import statsmodels.api as sm
#from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import PolynomialFeatures


################################################################################

def read_data( file_path, plot_histogram=None, verbose=False): 
    """
    Reads data file & performs data cleaning.
    
    ARGS:
        file_path (str): Full path to the data file
        plot_histogram (str): Full path to plot. If None, no plot is generated.
        verbose (bool): Binary flag for printing information.
        
    RETURNS:
        X (pd.DataFrame): Training features
        y_pcs (pd.Series): Training labels
    """
    
    # Loads the Excel file
    df = pd.read_excel(file_path, sheet_name="Sheet1")

    # Loads the first sheet
    first_sheet_name = list(df.keys())[0]
    df_main = df[first_sheet_name]

    # Print available sheet names & Display basic info
    if verbose:
        print("Sheets in the file:", df.keys())
        print("\nDataset Overview:")
        print(df_main.info())
        print(df_main.head())

    # Save as Excel for external analysis/cleaned dataset to post on Github
    if verbose:
        df_main.to_csv("imported_data.csv", index=False)
        print("Data saved as 'imported_data.csv'")

    # Checks for duplicated values. Returns 0, which is good
    if verbose:
        print(df.duplicated().sum())
        print(df.dtypes)

    # Identify columns with incorrect data types. Everything is int64
    if verbose:
        object_cols = df.select_dtypes(include=['object']).columns
        print("Columns stored as object:\n", object_cols)

    # Removes spaces in columns
    # Otherwise, column name error results for dependent variables
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
    
    # Plot histograms of data. Nothing of interest was observed.
    if plot_histogram:
        columns_to_plot = df.columns[:10]  # Gets the first 10 column names
        df[columns_to_plot].hist(bins=30, figsize=(16, 12))
        plt.savefig(plot_histogram)
        
    
    return X, y_pcs


################################################################################

def predict_LR(X, y_pcs):
    """
    ADD METHOD DESCRIPTION
    
    ARGS:
        X (pd.DataFrame): add info
        y_pcs (pd.Series): add info
        
    RETURNS:
        X_train_pcs (pd.DataFrame): add info
        y_train_pcs (pd.Series): add info
        X_test_pcs (pd.DataFrame): add info
        y_test_pcs (pd.Series): add info
        y_pred_pcs (pd.Series): add info
    """

    # Train-test split for PCS model
    X_train_pcs, X_test_pcs, \
    y_train_pcs, y_test_pcs  = train_test_split(X, y_pcs,
                                                test_size=0.2, random_state=42)

    # Train-test split for MFQ model
    #X_train_mfq, X_test_mfq,
    #y_train_mfq, y_test_mfq = train_test_split(X, y_mfq,
    #                                           test_size=0.2, random_state=42)
    
    # Initialize models
    linear_pcs = LinearRegression()
    #linear_mfq = LinearRegression()

    # Train model on PCS data
    linear_pcs.fit(X_train_pcs, y_train_pcs)

    # Train model on MFQ data
    #linear_mfq.fit(X_train_mfq, y_train_mfq)

    # Predict on test set
    y_pred_pcs = linear_pcs.predict(X_test_pcs)
    #y_pred_mfq = linear_mfq.predict(X_test_mfq)

        
    return linear_pcs, X_train_pcs, y_train_pcs, \
           X_test_pcs, y_test_pcs,  y_pred_pcs


################################################################################


def evaluate(linear_pcs, X_train_pcs, y_train_pcs, X_test_pcs, y_test_pcs,
             y_pred_pcs, plot_predictions=None, plot_residuals=None):
    """
    ADD METHOD DESCRIPTION

    ARGS:
        X_train_pcs (pd.DataFrame): add info
        y_train_pcs (pd.Series): add info
        X_test_pcs (pd.DataFrame): add info
        y_test_pcs (pd.Series): add info
        y_pred_pcs (pd.Series): add info
        plot_pred-vs-true (str): add info
        plot_residuals (str): add info
        
    RETURNS:
        mae_pcs (): add info
        rmse_pcs (): add info
        r2_pcs (): add info
    """

    
    # Evaluate PCS Model
    mae_pcs  = mean_absolute_error(y_test_pcs, y_pred_pcs)
    rmse_pcs = np.sqrt(mean_squared_error(y_test_pcs, y_pred_pcs))
    r2_pcs   = r2_score(y_test_pcs, y_pred_pcs)

    # Evaluate MFQ Model
    ##mae_mfq  = mean_absolute_error(y_test_mfq, y_pred_mfq)
    ##rmse_mfq = np.sqrt(mean_squared_error(y_test_mfq, y_pred_mfq))
    ##r2_mfq   = r2_score(y_test_mfq, y_pred_mfq)

    # Print results
    print(f"PCS Model - MAE: {mae_pcs:.2f}, RMSE: {rmse_pcs:.2f}, R²: {r2_pcs:.2f}")
    #print(f"MFQ Model - MAE: {mae_mfq:.2f}, RMSE: {rmse_mfq:.2f}, R²: {r2_mfq:.2f}")

    """
    # Add a constant (intercept) for OLS analysis
    X_train_pcs_const = sm.add_constant(X_train_pcs)
    
    # Fit OLS(p-values) model
    ols_model = sm.OLS(y_train_pcs, X_train_pcs_const).fit()
    
    # Print summary
    print(ols_model.summary())
    """
    
    print("PCS Train R²:", linear_pcs.score(X_train_pcs, y_train_pcs)) # Prints PCS training r^2
    print("PCS Test R²:", linear_pcs.score(X_test_pcs, y_test_pcs))    # Prints PCS testing r^2

    ##print("MFQ Train R²:", linear_mfq.score(X_train_pcs, y_train_pcs)) # Prints PCS training r^2
    ##print("MFQ Test R²:", linear_mfq.score(X_test_pcs, y_test_pcs))    # Prints PCS testing r^2

    ########################################################################

    # Predicted vs. Actual Plot
    if plot_predictions:
        plt.figure(figsize=(6, 5))
        sns.scatterplot(x=y_test_pcs, y=y_pred_pcs, alpha=0.6)
        plt.plot( [y_test_pcs.min(), y_test_pcs.max()],
                  [y_test_pcs.min(), y_test_pcs.max()], 'r--')  # 45-degree reference line
        plt.title("Predicted vs. Actual PCS Severity")
        plt.xlabel("Actual PCS Severity")
        plt.ylabel("Predicted PCS Severity")
        plt.savefig(plot_predictions)
    
    # Residual Plot (Checking Model Assumptions)
    if plot_residuals:
        plt.figure(figsize=(6, 5))
        residuals = y_test_pcs - y_pred_pcs
        sns.scatterplot(x=y_pred_pcs, y=residuals, alpha=0.6)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.title("Residual Plot")
        plt.xlabel("Predicted PCS Severity")
        plt.ylabel("Residuals")
        plt.savefig(plot_residuals)


    return mae_pcs, rmse_pcs, r2_pcs


################################################################################

def importance(linear_pcs, X_train_pcs, plot_importance=None):
    """
    ADD METHOD DESCRIPTION
    # You'll need to y_train_pcs, X_test_pcs, y_test_pcs variables as an input
    # to the function definded above if you uncomment sections of this code.

    ARGS:
        X_train_pcs (pd.DataFrame): add info
        y_train_pcs (pd.Series): add info
        X_test_pcs (pd.DataFrame): add info
        y_test_pcs (pd.Series): add info
        y_pred_pcs (pd.Series): add info
        plot_importance (str): add info
        
    RETURNS:
        mae_pcs (): add info
        rmse_pcs (): add info
        r2_pcs (): add info
    """

    # Compute VIF for each feature. Tests for multicolinearity.
    ##vif_data = pd.DataFrame()
    ##vif_data["Feature"] = X_train_pcs.columns
    ##vif_data["VIF"] = [variance_inflation_factor(X_train_pcs.values, i) for i in range(X_train_pcs.shape[1])]
    ##print(vif_data)

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

    # Feature Importance Bar Chart
    if plot_importance:
        plt.figure(figsize=(8, 5))
        sns.barplot(x=feature_importance.values, y=feature_importance.index, palette="viridis")
        plt.title("Feature Importance (Higher = More Impact on PCS Severity)")
        plt.xlabel("Importance Score (Absolute Regression Coefficients)")
        plt.ylabel("Feature")
        plt.savefig(plot_importance)


################################################################################

def correlation(X_train_pcs, plot_correlation=None):
    """ 
    """
    # Compute correlation matrix. Checking to see why age had such high multi
    # collinearity. Could not find an answer.
    correlation_matrix = X_train_pcs.corr()
    
    # Get correlation values for "Age (Years)" and sort them
    age_correlations = correlation_matrix["Age (Years)"].sort_values(ascending=False)
    
    # Display the highest correlations
    print("\nCorrelation of 'Age (Years)' with other variables:")
    print(age_correlations)
    
    # Plot heatmap for visualization
    if plot_file:
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, cmap="coolwarm", annot=False)
        plt.title("Feature Correlation Heatmap")
        plt.savefig(plot_correlation)


    # Compute the Pearson correlation between PCS and MFQ scores to help deceide to drop MFQ model
    correlation = df["PCS Symptom Severity (132)"].corr(df["MFQ 66"])
    
    # Print the result
    print(f"Correlation between PCS and MFQ: {correlation:.4f}")


################################################################################

### Plot the distribution of PCS Severity Scores. This shows 43 have a score of 0 and very few have scores past 60.
##plt.figure(figsize=(8, 5))
##sns.histplot(y_train_pcs, bins=20, kde=True)
##plt.title("PCS Severity Score Distribution (Before Resampling)")
##plt.xlabel("PCS Severity Score")
##plt.ylabel("Frequency")
##plt.show()

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

### Concatenate resampled severe cases with original non-severe cases
##X_train_resampled = pd.concat([non_severe_cases, severe_cases_resampled])
##y_train_resampled = pd.concat([non_severe_labels, severe_labels_resampled])

### --- Step 2: Plot the Final Resampled Distribution ---
##plt.figure(figsize=(8, 5))
##sns.histplot(y_train_resampled, bins=20, kde=True)
##plt.title("PCS Severity Score Distribution (After Bootstrap Resampling)")
##plt.xlabel("PCS Severity Score")
##plt.ylabel("Frequency")
##plt.show()


################################################################################

#def polynomial_regression(X_train, X_test, y_train, y_test, degrees=[1, 2, 5]):  This part has crashed and burned. I give up on turning it into a defined function for now.
    """
    Performs Polynomial Regression for multiple polynomial degrees and visualizes the results.

    Parameters:
    - X_train: Training feature set
    - X_test: Testing feature set
    - y_train: Training target variable (PCS Severity)
    - y_test: Testing target variable (PCS Severity)
    - degrees: List of polynomial degrees to test (default = [1, 2, 20])

    Returns:
    - Prints MSE scores for train and test sets
    - Displays Predicted vs. Actual scatter plots for each polynomial degree
    """
    plt.figure(figsize=(15, 5))

    for i, degree in enumerate(degrees):
        # Transform features into polynomial form
        X_train = X_train.to_numpy().reshape(-1, 1)
        X_test = X_test.to_numpy().reshape(-1, 1)

        poly = PolynomialFeatures(degree=degree)
        X_poly_train = poly.fit_transform(X_train)
        X_poly_test = poly.transform(X_test)

        # Creating the model
        model = LinearRegression()
        
        # Train Polynomial Regression Model
        model.fit(X_poly_train, y_train)    

        # Evaluating the model  
        y_pred_train = model.predict(X_poly_train)
        y_pred_test = model.predict(X_poly_test)

        # Compute Mean Squared Error
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)

        # Plot Predicted vs. Actual PCS Severity
        plt.subplot(1, len(degrees), i + 1)
        plt.scatter(y_train, y_pred_train, color='blue', alpha=0.5, label="Train Data")
        plt.scatter(y_test, y_pred_test, color='red', alpha=0.5, label="Test Data")
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'g--', label="Perfect Fit")

        plt.title(f'Degree={degree}\nTrain MSE={train_mse:.2f}, Test MSE={test_mse:.2f}')
        plt.xlabel("Actual PCS Severity")
        plt.ylabel("Predicted PCS Severity")
        plt.legend()

        # Print MSE Scores
        print(f"Polynomial Regression (Degree {degree}) - Train MSE: {train_mse:.2f}, Test MSE: {test_mse:.2f}")

    plt.show()

def main():
    # DEFINE DATA / PLOT LOCATIONS
    PROJECT_DIR = r"C:\Users\思庭\AppData\Local\Programs\Python\Python313\dataverse_files tbi Dec 2024 ML Project"
    DATA_FILE = "Cleaned_Data.xlsx"
    DATA_PATH = os.path.join(PROJECT_DIR, DATA_FILE)

##    print("Checking file path:", DATA_PATH)
##    print("File exists:", os.path.exists(DATA_PATH))  # Debugging step

    if not os.path.exists(DATA_PATH):
        print("❌ ERROR: The file path is incorrect or the file does not exist!")
        return  # Exit the function to avoid further errors

    # Reads data
    X, y_pcs = read_data(file_path=DATA_PATH, plot_histogram=None, verbose=False)

    # Models and predicts data using linear regression
    linear_pcs, X_train_pcs, y_train_pcs, \
    X_test_pcs, y_test_pcs,  y_pred_pcs = predict_LR(X, y_pcs)

    # Evaluates the performance of the predictions from the linear regressions
##    evaluate(linear_pcs, X_train_pcs, y_train_pcs,
##             X_test_pcs, y_test_pcs,  y_pred_pcs)
    
    # Run Polynomial Regression for degrees 1, 2, and 3
    polynomial_regression(X_train_pcs, X_test_pcs, y_train_pcs, y_test_pcs, degrees=[1, 2, 5])


#################################################################################

##if __name__ == "__main__":
##    main()

