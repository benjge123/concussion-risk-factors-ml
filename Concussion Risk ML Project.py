import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Finds long file path, rip
file_path = r"C:\Users\思庭\AppData\Local\Programs\Python\Python313\dataverse_files tbi Dec 2024 ML Project\Normative Athlete Data - PCSS & MFQ.xlsx"


# Loads the Excel file
df = pd.read_excel(file_path, sheet_name="Sheet1")

# Print available sheet names
print("Sheets in the file:", df.keys())

# Loads the first sheet
first_sheet_name = list(df.keys())[0]
df_main = df[first_sheet_name]

### Display basic info
##print("\nDataset Overview:")
##print(df_main.info())  
##print(df_main.head())


# Save as CSV for external analysis
df_main.to_csv("imported_data.csv", index=False)
print("Data saved as 'imported_data.csv'")

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
y_mfq = df["MFQ 66"]                     #66 is the most severe Mood and Feelings Questionaire

# Drop non-predictive columns (Participant ID and target variables)
X = df.drop(columns=["Participant ID", "PCS Symptom Severity (132)", "MFQ 66"])

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

### Code to plot histograms of data to see if there's anything interesting. There was not
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
