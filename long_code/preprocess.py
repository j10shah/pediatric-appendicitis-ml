import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Load the specific CSV file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, '../pediatric-appendicitis-ml/dataset/allcases.csv'))

#  Drop the "ghost" Excel columns (anything starting with 'Unnamed')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

#  Drop columns missing more than 50% of their data
# thresh requires a minimum number of NON-NA values to keep the column
threshold = len(df) * 0.5 
df = df.dropna(thresh=threshold, axis=1)

# Save all our potential targets separately 
y_diagnosis = df['Diagnosis']
y_management = df['Management']
y_severity = df['Severity']

# Make sure NO targets are in our features (X)
# We also drop 'Diagnosis_Presumptive' as it's a doctor's guess before the final diagnosis
targets_to_drop = ['Diagnosis', 'Management', 'Severity', 'Diagnosis_Presumptive']
X = df.drop(columns=targets_to_drop)

# Impute missing values (from the step we just did)
numeric_cols = X.select_dtypes(include=['number']).columns
X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

categorical_cols = X.select_dtypes(include=['object']).columns
for col in categorical_cols:
    X[col] = X[col].fillna(X[col].mode()[0])

# Encode Categorical Variables (One-Hot Encoding)
# This turns text columns into multiple binary (0 or 1) columns
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Scale the Data
# Logistic Regression needs all numbers to be on a similar scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Convert back to a DataFrame just so we can see the column names if we want to
X_final = pd.DataFrame(X_scaled, columns=X_encoded.columns)

# Split the Data! (80% for training, 20% held out for final testing)
# We will use y_diagnosis as our target for this first Logistic Regression model
X_train, X_test, y_train, y_test = train_test_split(X_final, y_diagnosis, test_size=0.2, random_state=42)

print("\n--- Final Data Prep ---")
print(f"Training features shape: {X_train.shape}")
print(f"Testing features shape: {X_test.shape}")
print("Data is scaled, encoded, and ready for TensorFlow!")