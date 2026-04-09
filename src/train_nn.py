import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

# =========================
# LOAD DATA
# =========================
dataset = fetch_ucirepo(id=938)

X = dataset.data.features
y = dataset.data.targets["Diagnosis"]

# Convert string labels to numeric
y = y.map({
    "no appendicitis": 0,
    "appendicitis": 1
}).astype(float)

# Drop rows where target is missing
mask = y.notna()
X = X[mask].reset_index(drop=True)
y = y[mask].reset_index(drop=True)

# Encode categorical features
X = pd.get_dummies(X)

# =========================
# MODEL DEFINITION
# =========================
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# =========================
# K-FOLD CROSS VALIDATION
# =========================
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []
f1_scores = []
roc_aucs = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Impute missing values (fit on train only)
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Scale features (fit on train only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test.values, dtype=torch.float32)

    # Initialize model fresh each fold
    model = MLP(X_train_t.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    # Train
    for epoch in range(100):
        model.train()
        outputs = model(X_train_t).squeeze()
        loss = criterion(outputs, y_train_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        preds = model(X_test_t).squeeze()
        preds_binary = (preds > 0.5).float()

    y_test_np = y_test_t.numpy()
    preds_np = preds.numpy()
    preds_binary_np = preds_binary.numpy()

    acc = accuracy_score(y_test_np, preds_binary_np)
    f1 = f1_score(y_test_np, preds_binary_np)
    auc = roc_auc_score(y_test_np, preds_np)

    accuracies.append(acc)
    f1_scores.append(f1)
    roc_aucs.append(auc)

    print(f"--- Fold {fold + 1} ---")
    print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, ROC-AUC: {auc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_np, preds_binary_np))
    print()

# =========================
# FINAL RESULTS
# =========================
print("=== FINAL RESULTS ===")
print(f"Accuracy:  {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
print(f"F1 Score:  {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
print(f"ROC-AUC:   {np.mean(roc_aucs):.4f} ± {np.std(roc_aucs):.4f}")