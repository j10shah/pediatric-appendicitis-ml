from dataloading import DataLoader
from model import Model
from gradient_boosting import GradientBoosting
import numpy as np





def run_model(model : Model, epochs = 500):
    data = DataLoader('data.csv', False)

    train_X, train_y, test_X, test_y = data.train_test_split(.7,0)
    train_y, test_y = data.clean_y(0, train_y, test_y)

    # # print(train_y)
    # # print(np.unique(train_y[:, 0]))
    # # print(np.unique(test_y[:, 0]))
    # # counter = 0
    # # last = 0
    # # for i in train_y[:, 2]:
    # #     if type(i) == type(1.0):
    # #         print("FLOAT?", 1, 'indes:', counter)
    # #         print('ROW:', train_y[counter])
    # #         last = counter
    # #     counter+=1

    # # print(train_y[last, :])

    model.train(train_X, train_y, epochs, 15, -1)
    preds = model.predict(train_X, train_y)
    correct = 0
    for i in range(len(preds)):
        if (preds[i] > .5 and train_y[i] == 1) or (preds[i] <= .5 and train_y[i] == 0):
            correct+=1
    print(correct/len(preds))

    preds = model.predict(test_X, test_y)
    correct = 0
    for i in range(len(preds)):
        if (preds[i] > .5 and test_y[i] == 1) or (preds[i] <= .5 and test_y[i] == 0):
            correct+=1
    print(correct/len(preds))
    


if __name__ == "__main__":
    gb = GradientBoosting(
        {
            "tree_depth": 3,
            "learning_rate": .01,
        },
        log_path= 'src/logs/gb_log.csv'
    )
    run_model(gb) 