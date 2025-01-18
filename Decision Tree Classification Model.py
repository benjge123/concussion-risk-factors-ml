import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree

# Step 1: Load the dataset
file_path = r"C:\Users\思庭\AppData\Local\Programs\Python\Python313\dataverse_files tbi Dec 2024 ML Project\Cleaned_Data.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")


# Step 3: Map PCS Total Symptom Severity to categories
def categorize_pcs(score):
    if score <= 15:
        return "Mild"
    elif score <= 30:
        return "Moderate"
    elif score <= 60:
        return "Severe"
    else:
        return "Very Severe"

df['PCS_Category'] = df['PCS Symptom Severity (132)'].apply(categorize_pcs)

# Step 4: Prepare features and labels
X =   df[
    ["MFQ Cut off",
        "Learning Disability",
        "Aggregate Medical History",
        "Anxiety Diagnosis",
        "Depression Diagnosis",
        "Anxiety Symptoms",
        "# of Prior Depressive Episodes",
        "Sex",
        "Prior Depressive Episode(s) Y/N",
       #"Age (Years)"] Drop irrelevant columns
       ]
    ]
y = df['PCS_Category']

# Print the shape of features and labels
print(f"Shape of X (features): {X.shape}")
print(f"Shape of y (labels): {y.shape}")


# Step 5: Encode categorical labels (if necessary)
y = y.astype('category').cat.codes

# Step 6: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train the Decision Tree model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 8: Evaluate the model
y_pred = clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


# Step 9: Visualize the Decision Tree
plt.figure(figsize=(20,10))
tree.plot_tree(clf, feature_names=X.columns, class_names=['Mild', 'Moderate', 'Severe', 'Very Severe'], filled=True)
plt.show()
