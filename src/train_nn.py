import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ucimlrepo import fetch_ucirepo # Make sure to import fetch_ucirepo in this file, as it's used to load the dataset. This was missing in the original code and would cause a NameError when trying to run the script.
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             classification_report, confusion_matrix)
from sklearn.utils.class_weight import compute_class_weight

# LOAD DATA
dataset = fetch_ucirepo(id=938)
X_raw = dataset.data.features
targets = dataset.data.targets

# TARGET DEFINITIONS
# Target 1: Diagnosis    — binary
# Target 2: Severity     — binary
# Target 3: Management   — multiclass (3 classes)
target_configs = { # Define each target's column, mapping, number of classes, and whether to use class weights
    "Diagnosis": {
        "col": "Diagnosis",
        "map": {"no appendicitis": 0, "appendicitis": 1},
        "num_classes": 1,   # binary → sigmoid output
        "use_weights": False
    },
    "Severity": {
        "col": "Severity", # this column has some NaNs, but we'll handle that in the code
        "map": {"uncomplicated": 0, "complicated": 1},
        "num_classes": 1,
        "use_weights": True
    },
    "Management": {
        "col": "Management",
        "map": None,        # label-encoded below
        "num_classes": 3,
        "use_weights": True
    }
}

# MODEL DEFINITION
class MLP(nn.Module): # Add num_classes to constructor to handle both binary and multiclass
    def __init__(self, input_dim, num_classes=1):
        super().__init__()
        out_dim = num_classes if num_classes > 1 else 1
        activation = nn.Softmax(dim=1) if num_classes > 1 else nn.Sigmoid()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32),        nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, out_dim),   activation
        )
    def forward(self, x): # The forward method remains the same, as the output layer and activation are defined in __init__
        return self.net(x)

# TRAIN + EVALUATE ONE TARGET
def run_target(target_name, config, X_raw, targets): # This function encapsulates the entire process for one target, making it easy to loop through all targets later
    print(f"\n{'='*50}")
    print(f"TARGET: {target_name}")
    print(f"{'='*50}")

    col = config["col"]
    num_classes = config["num_classes"]

    # Prepare target
    y = targets[col].copy()

    if config["map"] is not None: # If a mapping is provided, use it to convert string labels to numeric
        y = y.map(config["map"])
    else: # Drop rare classes with fewer than 2 samples before encoding
        class_counts = y.value_counts()
        valid_classes = class_counts[class_counts >= 2].index
        y = y[y.isin(valid_classes)]
        print(f"Dropped rare classes: {set(class_counts.index) - set(valid_classes)}")

        valid_idx = y.dropna().index       # Label encode Management — drop NaN first, then encode
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y.loc[valid_idx]), index=valid_idx)
        print(f"Classes: {le.classes_}")

    y = y.astype(float)
    mask = y.notna()
    # Align both X and y to the same index before resetting
    X = X_raw.loc[mask.index[mask]].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    # Encode categorical features
    X = pd.get_dummies(X)

    # Holdout test set (withheld from all CV)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # K-Fold CV
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accuracies, cv_f1s = [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_trainval, y_trainval)): # Note: split on trainval, not the original X and y
        X_tr, X_val = X_trainval.iloc[train_idx], X_trainval.iloc[val_idx]
        y_tr, y_val = y_trainval.iloc[train_idx], y_trainval.iloc[val_idx]

        # Preprocessing (fit on train only)
        imputer = SimpleImputer(strategy="median")
        X_tr  = imputer.fit_transform(X_tr)
        X_val = imputer.transform(X_val)

        scaler = StandardScaler()
        X_tr  = scaler.fit_transform(X_tr)
        X_val = scaler.transform(X_val)

        # Tensors
        X_tr_t  = torch.tensor(X_tr,  dtype=torch.float32)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)

        if num_classes == 1: # For binary classification, use float targets for BCELoss
            y_tr_t  = torch.tensor(y_tr.values,  dtype=torch.float32)
        else:
            y_tr_t  = torch.tensor(y_tr.values.astype(int), dtype=torch.long)

        # Class weights
        if config["use_weights"]: # Only compute weights for the training set in this fold
            classes = np.unique(y_tr.values)
            weights = compute_class_weight("balanced", classes=classes, y=y_tr.values)
            weight_tensor = torch.tensor(weights, dtype=torch.float32)
        else:
            weight_tensor = None

        # Loss function
        if num_classes == 1:
            criterion = nn.BCELoss(weight=None)  # binary: apply sample weights manually if needed
        else:
            criterion = nn.CrossEntropyLoss(weight=weight_tensor)

        # Model
        model = MLP(X_tr_t.shape[1], num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Train
        for _ in range(100):
            model.train()
            out = model(X_tr_t)
            if num_classes == 1:
                loss = criterion(out.squeeze(), y_tr_t)
            else:
                loss = criterion(out, y_tr_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            out_val = model(X_val_t)
            if num_classes == 1:
                preds_bin = (out_val.squeeze().numpy() > 0.5).astype(float)
            else:
                preds_bin = out_val.argmax(dim=1).numpy()

        acc = accuracy_score(y_val.values, preds_bin)
        f1  = f1_score(y_val.values, preds_bin, average="weighted")
        cv_accuracies.append(acc)
        cv_f1s.append(f1)

    print(f"\nCV Results (5-Fold):")
    print(f"  Mean Accuracy : {np.mean(cv_accuracies)*100:.2f}% ± {np.std(cv_accuracies)*100:.2f}%")
    print(f"  Mean F1       : {np.mean(cv_f1s):.4f} ± {np.std(cv_f1s):.4f}")

    # Final evaluation on holdout test set
    imputer = SimpleImputer(strategy="median")
    X_tv = imputer.fit_transform(X_trainval)
    X_te = imputer.transform(X_test)

    scaler = StandardScaler() # Fit on trainval, transform both trainval and test
    X_tv = scaler.fit_transform(X_tv)
    X_te = scaler.transform(X_te)

    X_tv_t = torch.tensor(X_tv, dtype=torch.float32)
    X_te_t = torch.tensor(X_te, dtype=torch.float32)

    if num_classes == 1: # For binary classification, use float targets for BCELoss
        y_tv_t = torch.tensor(y_trainval.values, dtype=torch.float32)
    else:
        y_tv_t = torch.tensor(y_trainval.values.astype(int), dtype=torch.long)

    if config["use_weights"]: # Compute class weights on the entire trainval set for the final model
        classes = np.unique(y_trainval.values)
        weights = compute_class_weight("balanced", classes=classes, y=y_trainval.values)
        weight_tensor = torch.tensor(weights, dtype=torch.float32)
    else:
        weight_tensor = None

    if num_classes == 1: # For binary classification, use BCELoss without built-in weights (we could apply sample weights manually if needed, but for simplicity we'll skip that here)
        criterion = nn.BCELoss()
    else:
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    final_model = MLP(X_tv_t.shape[1], num_classes)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=0.001)

    for _ in range(100): # Train on the entire trainval set before evaluating on the holdout test set
        final_model.train()
        out = final_model(X_tv_t)
        if num_classes == 1: # For binary classification, compute loss with BCELoss
            loss = criterion(out.squeeze(), y_tv_t)
        else:
            loss = criterion(out, y_tv_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_model.eval()
    with torch.no_grad(): # Get predictions on the holdout test set
        out_test = final_model(X_te_t)
        if num_classes == 1: # For binary classification, apply threshold to get binary predictions
            preds_test = (out_test.squeeze().numpy() > 0.5).astype(float)
        else:
            preds_test = out_test.argmax(dim=1).numpy()

    print(f"\nHoldout Test Accuracy: {accuracy_score(y_test.values, preds_test)*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test.values, preds_test, digits=2))

# RUN ALL THREE TARGETS
for name, config in target_configs.items():
    run_target(name, config, X_raw, targets)