from model import Model
from sklearn import tree
import math
import numpy as np

#setup: Want weak learners, and a loss funciton that is differentiable and minimizing will result in a better solution
#Then, for the first learner f_1, predict something
#Then compute R_1_i = -dL(y_act, y_pr)/dy_pr, over all datapoints i in 1 to N
#Then, train next weak learner to approximate R_1_i given x
#Then, need to figure out how much of f_2 to add to f_1: F = f_1(x) + gamma_2 f_2(x)
#Where gamma = argmin_gamma SUM_i L[y_act, f_1(x) + gamma f_2(x)]

class GradientBoosting(Model):

    def __init__(self, hyper_parms: dict, weight_path=None, val_x=None, val_y=None, log_path = 'gb_log.csv'):
        super().__init__(hyper_parms, self.cross_entropy, weight_path, val_x, val_y, log_path=log_path)

    def cross_entropy(self, y_pred, y_act):
        y_pred = np.clip(y_pred ,.00001, .99999)
        return -np.mean(y_act * np.log(y_pred) + (1-y_act) * np.log(1-y_pred))
    
    def sig(self, A):
        return 1 / (1 + np.exp(-A))

    def grad_cross_entropy(self, y_pred, y_act):
        # y_pred = np.clip(y_pred ,.00001, .99999)
        return y_act - self.sig(y_pred)

    def assign_hyperparams(self): #don't need number of trees, that's covered with number of epochs
        self.tree_depth = self.hyper_parms["tree_depth"]
        self.learning_rate = self.hyper_parms["learning_rate"]
        self.models = []
        self.gammas = []
        self.F0 = 0

    def get_weights(self):
        return self.models, self.gammas

    def predict(self, X, y):
        #initialize f_0 as the class distribution
        if len(self.models) == 0:
            p = np.clip(np.sum(y == 1) / len(y), .00001, .99999)
            self.F0 = math.log(p/(1-p))

        #prediction = F_0 + sum over all models of lr * gamma * model(X)
        prediction = np.ones(len(y),) * self.F0

        for i in range(len(self.models)):
            #LEAF WISE:
            # leaf_idxes = self.models[i].apply(X)
            # prediction += self.learning_rate * np.array([self.gammas[i][li] for li in leaf_idxes]) #this is slow

            prediction += self.learning_rate * self.gammas[i] * self.models[i].predict(X)

        return prediction
        


    def update(self, X, y, train_loss, predictions):
        #first calculate residuals
        R_n = self.grad_cross_entropy(predictions, y)
        
        #then train the new model to predict said risidiauls
        f_n = tree.DecisionTreeRegressor(max_depth=self.tree_depth)
        f_n.fit(X, R_n)
        
        p = self.sig(predictions)
        gamma = np.sum(y - predictions) / np.sum(p * (1 - p))

        #then calculate model contribution to the ensemble - LEAF WISE
        # leaves = f_n.apply(X)
        # gammas = {}
        # for leaf in np.unique(leaves):
        #     leaf_indexes = np.where(leaves == leaf)
        #     p = self.sig(predictions[leaf_indexes])
        #     gamma = np.sum(y[leaf_indexes] - predictions[leaf_indexes]) / np.sum(p * (1 - p))
        #     gammas[leaf] = gamma

        self.models.append(f_n)
        # self.gammas.append(gammas)
        self.gammas.append(gamma)

    def save_weights(self):
        # raise NotImplementedError
        print(self.gammas[-1])
        pass
    

