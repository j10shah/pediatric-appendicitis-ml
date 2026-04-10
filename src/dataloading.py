from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataLoader():
    def __init__(self, csv_path = '', from_csv = False):
        if from_csv:
            self.X = pd.read_csv(csv_path + '_x.csv')
            self.y = pd.read_csv(csv_path + '_y.csv')
            self.cleaning(self.X, self.y)
        else:
            regensburg_pediatric_appendicitis = fetch_ucirepo(id=938)
            self.cleaning(regensburg_pediatric_appendicitis.data.features, regensburg_pediatric_appendicitis.data.targets)

    def save_to_csv(self, path):
        self.X.to_csv(path + '_x.csv', index = False)
        self.y.to_csv(path + '_y.csv', index = False)

    def cleaning(self, X: pd.DataFrame, y : pd.DataFrame):
        
        #  Drop the "ghost" Excel columns (anything starting with 'Unnamed')
        X = X.loc[:, ~X.columns.str.contains('^Unnamed')]

        #  Drop columns missing more than 50% of their data
        # thresh requires a minimum number of NON-NA values to keep the column
        threshold = len(X) * 0.5 
        X = X.dropna(thresh=threshold, axis=1)

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
        self.X = pd.DataFrame(X_scaled, columns=X_encoded.columns)

        #remove any rows where there is a nan y value
        mask = ~y.isna().any(axis=1)
        self.X = self.X[mask].reset_index(drop=True)
        self.y = y[mask].reset_index(drop=True)
        


    def to_np(self):
        return self.X.to_numpy(), self.y.to_numpy()
    
    def clean_y(self, target_num, train_y, test_y, val_y = None, feature_1 = None):
        if not feature_1:
            feature_1 = np.unique(train_y[:, target_num])[0]

        res_train_y = np.array([1 if v == feature_1 else 0 for v in train_y[:, target_num]])
        res_test_y = np.array([1 if v == feature_1 else 0 for v in test_y[:, target_num]])

        if val_y:
            res_val_y = np.array([1 if v == feature_1 else 0 for v in val_y[:, target_num]])
            return res_train_y, res_val_y, res_test_y
        return res_train_y, res_test_y
    
    def train_test_split(self, train_amount, validation_amount = 0):
        if train_amount < 0 or train_amount > 1 or validation_amount < 0 or validation_amount > 1 or train_amount + validation_amount > 1:
            raise ValueError('train and validation must be between 0 and 1')

        X, y = self.to_np()        
        indeces = np.random.permutation(len(X))

        train_ind = indeces[:int(train_amount * len(X))]

        if validation_amount > 0:
            val_ind = indeces[int(train_amount * len(X)):int(train_amount * len(X)) + int(validation_amount * len(X))]
            test_ind = indeces[int(train_amount * len(X)) + int(validation_amount * len(X)):]
            return X[train_ind], y[train_ind], X[val_ind], y[val_ind], X[test_ind], y[test_ind]
        
        test_ind = indeces[int(train_amount * len(X)):]

        return X[train_ind], y[train_ind], X[test_ind], y[test_ind]

