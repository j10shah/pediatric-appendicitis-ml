from abc import ABC, abstractmethod
import csv
import os

class Model(ABC):

    def __init__(self, hyper_parms : dict, loss : callable, weight_path = None, val_x = None, val_y = None, log_path = 'log.csv'):
        self.val_x = val_x
        self.val_y = val_y
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

    def train(self, X, y, epochs, checkpoint_amount = 0, validation_checkpoint = 0):
        
        for epoch in range(epochs):
            if checkpoint_amount > 0 and (epoch + 1) % checkpoint_amount == 0:
                self.save_weights()

            predictions = self.predict(X,y)
            train_loss = self.loss(predictions, y)
            self.store_checkpoint(epoch, train_loss)
            self.update(X, y, train_loss, predictions)

            if validation_checkpoint > 0 and self.val_x and (epoch + 1) % validation_checkpoint == 0:
                val_pr = self.predict(self.val_x)
                val_loss = self.predict(val_pr, y)
                self.store_checkpoint(epoch, val_loss)

    def store_checkpoint(self, epoch, value):
        print('LOGGING VALUE: ', epoch, ' LOSS: ', value)
        print('path:', os.getcwd())
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, value])

    @abstractmethod
    def predict(self, X,y):
        pass

    @abstractmethod
    def update(self, X, y, predictions):
        pass

    @abstractmethod
    def save_weights(self):
        pass