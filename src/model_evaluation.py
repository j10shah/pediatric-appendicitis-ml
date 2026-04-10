from dataloading import DataLoader
from model import Model
from gradient_boosting import GradientBoosting
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)


def sigmoid(A):
    return 1 / (1 + np.exp(-A))


def run_model(model : Model, train_X, train_y, test_X, test_y, epochs = 500, k = 5):

    kf = StratifiedKFold(n_splits=k, shuffle=True)

    acc_train = []
    precision_train = []
    recall_train = []
    f1_train= []
    roc_train = []

    acc_test = []
    precision_test = []
    recall_test = []
    f1_test = []
    roc_test = []

    k_ind = 0
    for train_idx, val_idx in kf.split(train_X, train_y):
        X_tr, X_val = train_X[train_idx], train_X[val_idx]
        y_tr, y_val = train_y[train_idx], train_y[val_idx]

        model.train(X_tr, y_tr, epochs, Val_X=X_val, Val_y=y_val, checkpoint_amount= -1, validation_checkpoint = -15, k = k_ind)
        preds_train = model.predict_class(model.predict(train_X, train_y), .5)
        preds_test = model.predict_class(model.predict(test_X, test_y), .5)

        acc_train.append(accuracy_score(train_y, preds_train))
        precision_train.append(precision_score(train_y, preds_train))
        recall_train.append(recall_score(train_y, preds_train))
        f1_train.append(f1_score(train_y, preds_train))
        roc_train.append(roc_auc_score(train_y, preds_train))

        acc_test.append(accuracy_score(test_y, preds_test))
        precision_test.append(precision_score(test_y, preds_test))
        recall_test.append(recall_score(test_y, preds_test))
        f1_test.append(f1_score(test_y, preds_test))
        roc_test.append(roc_auc_score(test_y, preds_test))
        k_ind+=1
        print('FINISHED FOLD')


    print('MODEL PERFORMANCE: -------------------')
    print('train accuracy: ', sum(acc_train)/len(acc_train))
    print('train precision: ', sum(precision_train)/len(precision_train))
    print('train recall: ', sum(recall_train)/len(recall_train))
    print('train f1: ', sum(f1_train)/len(f1_train))
    print('train roc: ', sum(roc_train)/len(roc_train))

    print('test accuracy: ', sum(acc_test)/len(acc_test))
    print('test precision: ', sum(precision_test)/len(precision_test))
    print('test recall: ', sum(recall_test)/len(recall_test))
    print('test f1: ', sum(f1_test)/len(f1_test))
    print('test roc: ', sum(roc_test)/len(roc_test))
    print('------------------')
    


if __name__ == "__main__":
    data = DataLoader('data.csv', True)
    train_X, train_y, test_X, test_y = data.train_test_split(.7,0)
    train_y_Man_Con, test_y_Man_Con = data.clean_y(0, train_y, test_y,  feature_1 = 'conservative')
    train_y_Man_Pri, test_y_Man_Pri = data.clean_y(0, train_y, test_y,  feature_1 = 'primary surgical')
    train_y_Man_Sec, test_y_Man_Sec = data.clean_y(0, train_y, test_y,  feature_1 = 'secondary surgical')
    train_y_Sev, test_y_Sev = data.clean_y(1, train_y, test_y)
    train_y_Dia, test_y_Dia = data.clean_y(2, train_y, test_y)

    gb_man_con = GradientBoosting({"tree_depth": 3, "learning_rate": .01}, log_path= 'src/logs/class_Man_Con_vs_rest_gb_log.csv')
    gb_man_pri = GradientBoosting({"tree_depth": 3, "learning_rate": .01}, log_path= 'src/logs/class_Man_Pri_vs_rest_gb_log.csv')
    gb_man_sec = GradientBoosting({"tree_depth": 3, "learning_rate": .01}, log_path= 'src/logs/class_Man_Sec_vs_rest_gb_log.csv')
    gb_sev = GradientBoosting({"tree_depth": 3, "learning_rate": .01}, log_path= 'src/logs/class_Sev_gb_log.csv')
    gb_dia = GradientBoosting({"tree_depth": 3, "learning_rate": .01}, log_path= 'src/logs/class_Dia_gb_log.csv')
    
    print('RUNNING ON ClASS FEATURE: MANAGEMENT, Conservative vs Rest')
    print('===============================')
    run_model(gb_man_con, train_X, train_y_Man_Con, test_X, test_y_Man_Con)
    print('===============================')
    print('RUNNING ON ClASS FEATURE: MANAGEMENT, Primary Surgery vs Rest')
    print('===============================')
    run_model(gb_man_pri, train_X, train_y_Man_Pri, test_X, test_y_Man_Pri)
    print('===============================')
    print('RUNNING ON ClASS FEATURE: Sevirity')
    print('===============================')
    run_model(gb_sev, train_X, train_y_Sev, test_X, test_y_Sev)
    print('===============================')
    print('RUNNING ON ClASS FEATURE: Diagnosis')
    print('===============================')
    run_model(gb_dia, train_X, train_y_Dia, test_X, test_y_Dia)