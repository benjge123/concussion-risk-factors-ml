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


import pandas as pd

# Define dependent variables
y_pcs = df["PCS Symptom Severity (132)"] #132 is the most severe Post Concussion Symptom scrore
y_mfq = df["MFQ 66"]                     #66 is the most severe Mood and Feelings Questionaire

# Drop Participant ID, target variables (PCS Total Symptom Severity, MFQ Total Score), and individual PCS/MFQ items
X = df.drop(columns=[
    "Participant ID",
    "PCS Symptom Frequency (22)",     # How many PCS symptoms do they have
    "Concussion History",             # Not statistically significant
    "# of Prior Depressive Episodes", # Not statistically significant
    "Age (Years)",                    # Although it improved adj r^2 by 0.013, this had a correlation >10 so I still chose to remove it
    "PCS Symptom Severity (132)",     # Target variable
    "MFQ 66",                         # Target variable
] + [f"PCS {i}" for i in range(1, 23)]  # Removing PCS1-22
  + [f"MFQ {i}" for i in range(1, 34)]  # Removing MFQ1-33
)


# Train-test split for PCS model
X_train_pcs, X_test_pcs, y_train_pcs, y_test_pcs = train_test_split(X, y_pcs, test_size=0.2, random_state=42)

# Train-test split for MFQ model
X_train_mfq, X_test_mfq, y_train_mfq, y_test_mfq = train_test_split(X, y_mfq, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression

# Initialize models
linear_pcs = LinearRegression()
linear_mfq = LinearRegression()

# Train model on PCS data 
linear_pcs.fit(X_train_pcs, y_train_pcs)

# Train model on MFQ data
linear_mfq.fit(X_train_mfq, y_train_mfq)

### Code to plot histograms of data to see if there's anything interesting. There was not.
##columns_to_plot = df.columns[:10]  # Gets the first 10 column names
##df[columns_to_plot].hist(bins=30, figsize=(16, 12))
##plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Predict on test set
y_pred_pcs = linear_pcs.predict(X_test_pcs)
y_pred_mfq = linear_mfq.predict(X_test_mfq)

# Evaluate PCS Model
mae_pcs = mean_absolute_error(y_test_pcs, y_pred_pcs)
rmse_pcs = np.sqrt(mean_squared_error(y_test_pcs, y_pred_pcs))
r2_pcs = r2_score(y_test_pcs, y_pred_pcs)

# Evaluate MFQ Model
mae_mfq = mean_absolute_error(y_test_mfq, y_pred_mfq)
rmse_mfq = np.sqrt(mean_squared_error(y_test_mfq, y_pred_mfq))
r2_mfq = r2_score(y_test_mfq, y_pred_mfq)

# Print results
print(f"PCS Model - MAE: {mae_pcs:.2f}, RMSE: {rmse_pcs:.2f}, R²: {r2_pcs:.2f}")
print(f"MFQ Model - MAE: {mae_mfq:.2f}, RMSE: {rmse_mfq:.2f}, R²: {r2_mfq:.2f}")

import statsmodels.api as sm

### Add a constant (intercept)
##X_train_pcs_const = sm.add_constant(X_train_pcs)
##
### Fit OLS(p-values) model
##ols_model = sm.OLS(y_train_pcs, X_train_pcs_const).fit()
##
### Print summary
##print(ols_model.summary())

##print("Train R²:", linear_pcs.score(X_train_pcs, y_train_pcs)) # Prints PCS training r^2
##print("Test R²:", linear_pcs.score(X_test_pcs, y_test_pcs))    # Prints PCS testing r^2

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Compute VIF for each feature
vif_data = pd.DataFrame()
vif_data["Feature"] = X_train_pcs.columns
vif_data["VIF"] = [variance_inflation_factor(X_train_pcs.values, i) for i in range(X_train_pcs.shape[1])]

print(vif_data)



