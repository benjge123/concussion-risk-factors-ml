import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV


# Load the dataset
file_path = r"C:\Users\思庭\AppData\Local\Programs\Python\Python313\dataverse_files tbi Dec 2024 ML Project\Cleaned_Data.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")

###################### Random Forest Classification Model ###########################

# Step 3: Map PCS Total Symptom Severity to categories
def categorize_pcs(score):
    if score == 0:
        return "None"
    elif score <= 15:
        return "Mild"
    elif score <= 30:
        return "Moderate"
    elif score <= 60:
        return "Severe"
    else:
        return "Very Severe"

df['PCS_Category'] = df['PCS Symptom Severity (132)'].apply(categorize_pcs)

# Prepare features and labels
X = df[
    [
        "MFQ Cut off",
        "Learning Disability",
        "Aggregate Medical History",
        "Anxiety Diagnosis",
        "Depression Diagnosis",
        "Anxiety Symptoms",
        "# of Prior Depressive Episodes",
        "Sex",
        "Prior Depressive Episode(s) Y/N",
        "Age (Years)"
    ]
]
y = df['PCS_Category']

# Encode categorical labels 
y = y.astype('category').cat.codes

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model with initial parameters
rf = RandomForestClassifier(
    class_weight={0: 1, 1: 1.5, 2: 2, 3: 2.5, 4: 3},  
    random_state=42,
    n_estimators=50,  
    max_depth=3,    
)
rf.fit(X_train, y_train)

# Evaluate the model
y_pred = rf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


# Hyperparameter tuning with Grid Search
param_grid = {
    'n_estimators': [10,20,40, 50, 100, 200,300],
    'max_depth': [1,3,5, 10,20, None]      
##    'class_weight': [
##        {0: 1, 1: 2, 2: 3, 3: 5, 4: 10},
##        {0: 1, 1: 3, 2: 5, 3: 7, 4: 10},
##        {0: 1, 1: 4, 2: 6, 3: 10, 4: 15},
##        {0: 1, 1: 5, 2: 8, 3: 12, 4: 20},
##        {0: 1, 1: 6, 2: 9, 3: 15, 4: 25},
##        'balanced',
##        {0: 1, 1: 6, 2: 8, 3: 12, 4: 20},  
##        {0: 1, 1: 8, 2: 10, 3: 15, 4: 25}, 
##        {0: 1, 1: 10, 2: 12, 3: 18, 4: 30},  
##        {0: 1, 1: 12, 2: 15, 3: 20, 4: 35},
##        {0: 1, 1: 1, 2: 1, 3: 1, 4: 1},  
##        {0: 1, 1: 1.5, 2: 2, 3: 2.5, 4: 3},  
##        {0: 1, 1: 2, 2: 2.5, 3: 3, 4: 4},  
##        {0: 1, 1: 2, 2: 3, 3: 4, 4: 5},  
    
        }

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=2,  # Cross-validation splits
    scoring='f1_weighted'
)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)




###################### Decision Tree Classification Model ###########################



### Map PCS Total Symptom Severity to categories
##def categorize_pcs(score):
##    if score == 0:
##        return "None"
##    elif score <= 15:
##        return "Mild"
##    elif score <= 30:
##        return "Moderate"
##    elif score <= 60:
##        return "Severe"
##    else:
##        return "Very Severe"
##
##df['PCS_Category'] = df['PCS Symptom Severity (132)'].apply(categorize_pcs)
##
### Prepare features and labels
##X =   df[
##    [   "MFQ Cut off",
##        "Learning Disability",
##        "Aggregate Medical History",
##        "Anxiety Diagnosis",
##        "Depression Diagnosis",
##        "Anxiety Symptoms",
##        "# of Prior Depressive Episodes",
##        "Sex",
##        "Prior Depressive Episode(s) Y/N",
##        "Age (Years)" 
##       ]
##    ]
##y = df['PCS_Category']
##
### Print the shape of features and labels
##print(f"Shape of X (features): {X.shape}")
##print(f"Shape of y (labels): {y.shape}")
##
##
### Encode categorical labels 
##y = y.astype('category').cat.codes
##
### Split the data
##X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
##
### Train the Decision Tree model
##clf = DecisionTreeClassifier(class_weight={0: 1, 1: 6, 2: 2.5, 3: 4, 4:7}, random_state=42)
##clf.fit(X_train, y_train)
##
### Evaluate the model
##y_pred = clf.predict(X_test)
##print("Classification Report:")
##print(classification_report(y_test, y_pred))
##
##accuracy = accuracy_score(y_test, y_pred)
##print(f"Accuracy: {accuracy:.2f}")
##
##print("Classes in y_test:", set(y_test))
##print("Classes in y_pred:", set(y_pred))
##
##from sklearn.model_selection import GridSearchCV
##
##param_grid = {
##    'class_weight': [
##        {0: 1, 1: 2, 2: 3, 3: 5, 4: 10},
##        {0: 1, 1: 1.5, 2: 2.5, 3: 4, 4: 6},
##        {0: 1, 1: 1, 2: 2, 3: 3, 4: 5},
##    ]
##}
##clf = DecisionTreeClassifier(random_state=42)
##grid_search = GridSearchCV(clf, param_grid, cv=2, scoring='f1_weighted')
##grid_search.fit(X_train, y_train)
##print(grid_search.best_params_)
##
##
##
### Visualize the Decision Tree
####plt.figure(figsize=(20,10))
####tree.plot_tree(clf, feature_names=X.columns, class_names=['Mild', 'Moderate', 'Severe', 'Very Severe'], filled=True)
####plt.show()
