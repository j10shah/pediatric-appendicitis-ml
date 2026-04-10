from abc import ABC, abstractmethod
import csv
import numpy as np

class Model(ABC):

    def __init__(self, hyper_parms : dict, loss : callable, weight_path = None, log_path = 'log.csv'):
        self.loss = loss
        self.log_path = log_path
        if weight_path:
            #TODO: load weights
            pass
            self.weight_path = weight_path
        else:
            self.weight_path = '' #TODO: DEFAULT WEIGHT SAVING

        self.hyper_parms = hyper_parms
        self.assign_hyperparams()

    @abstractmethod
    def assign_hyperparams(self):
        pass

    @abstractmethod
    def get_weights(self):
        pass

    def train(self, X, y, epochs, checkpoint_amount = 0, validation_checkpoint = 0, Val_X = None, Val_y = None, k = -1):
        
        for epoch in range(epochs):

            predictions = self.predict(X,y)
            train_loss = self.loss(predictions, y)

            if checkpoint_amount > 0 and (epoch + 1) % checkpoint_amount == 0:
                self.save_weights()

            self.update(X, y, train_loss, predictions)

            if validation_checkpoint > 0 and Val_X is not None and (epoch + 1) % validation_checkpoint == 0:
                val_pr = self.predict(Val_X, Val_y)
                val_loss = self.loss(val_pr, Val_y)
                self.store_checkpoint(k, epoch, train_loss, val_loss)

    def store_checkpoint(self, k, epoch, train_loss, val_loss):
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([k, epoch, train_loss, val_loss])

    def predict_class(self, predictions, threshold):
        return (predictions > threshold).astype(int)


    @abstractmethod
    def predict(self, X,y):
        pass

    @abstractmethod
    def update(self, X, y, predictions):
        pass

    @abstractmethod
    def save_weights(self):
        pass