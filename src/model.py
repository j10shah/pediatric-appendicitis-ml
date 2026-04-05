from abc import ABC, abstractmethod

class Model(ABC):

    def __init__(self, hyper_parms : dict, loss : callable, weight_path = None, val_x = None, val_y = None):
        self.val_x = val_x
        self.val_y = val_y
        self.loss = loss
        if weight_path:
            #TODO: load weights
            pass
            self.weight_path = weight_path
        else:
            self.weight_path = '' #TODO: DEFAULT WEIGHT SAVING

        self.hyper_parms = hyper_parms

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

            predictions = self.predict(X)
            train_loss = self.loss(predictions, y)
            #TODO: store train loss
            self.update(X, y, predictions)

            if validation_checkpoint > 0 and self.val_x and (epoch + 1) % validation_checkpoint == 0:
                val_pr = self.predict(self.val_x)
                val_loss = self.predict(val_pr, y)
                #TODO: Store validation loss somewhere


    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def update(self, X, y, predictions):
        pass

    @abstractmethod
    def save_weights(self):
        pass