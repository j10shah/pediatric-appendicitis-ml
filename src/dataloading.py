from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np


class DataLoader():
    def __init__(self, csv_path = '', from_csv = False):
        if from_csv:
            self.X = pd.read_csv(csv_path + '_x.csv')
            self.y = pd.read_csv(csv_path + '_y.csv')
        else:
            regensburg_pediatric_appendicitis = fetch_ucirepo(id=938) 
            self.X = regensburg_pediatric_appendicitis.data.features
            self.y = regensburg_pediatric_appendicitis.data.targets

    def save_to_csv(self, path):
        self.X.to_csv(path + '_x.csv', index = False)
        self.y.to_csv(path + '_y.csv', index = False)

    def cleaning_preprocessing_idk_whatver_we_want(self):
        #TODO: FILL OR REWORK ME!!!
        raise NotImplementedError

    def to_np(self):
        return self.X.to_numpy(), self.y.to_numpy()
    
    def train_test_split(self, train_amount, validation_amount = 0):
        if train_amount < 0 or train_amount > 1 or validation_amount < 0 or validation_amount > 1 or train_amount + validation_amount > 1:
            raise ValueError('train and validation must be between 0 and 1')
        
        indeces = np.random.permutation(len(self.X))

        train_ind = indeces[:int(train_amount * len(self.X))]

        if validation_amount > 0:
            val_ind = indeces[int(train_amount * len(self.X)):int(train_amount * len(self.X)) + int(validation_amount * len(self.X))]
            test_ind = indeces[int(train_amount * len(self.X)) + int(validation_amount * len(self.X)):]
            return self.X[train_ind], self.y[train_ind], self.X[val_ind], self.y[val_ind], self.X[test_ind], self.y[test_ind]
        
        test_ind = indeces[int(train_amount * len(self.X)):]

        return self.X[train_ind], self.y[train_ind], self.X[test_ind], self.y[test_ind]

