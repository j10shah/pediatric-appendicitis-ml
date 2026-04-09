import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

# =========================
# LOAD DATA
# =========================
dataset = fetch_ucirepo(id=938)
X = dataset.data.features
y = dataset.data.targets["Diagnosis"]

y = y.map({"no appendicitis": 0, "appendicitis": 1}).astype(float)
mask = y.notna()
X = X[mask].reset_index(drop=True)
y = y[mask].reset_index(drop=True)
X = pd.get_dummies(X)

# =========================
# MODEL DEFINITION
# =========================
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32),        nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1),         nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# =========================
# K-FOLD + COLLECT RESULTS
# =========================
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

all_fpr, all_tpr, all_auc = [], [], []
all_preds_binary, all_labels = [], []

for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test  = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32)

    model = MLP(X_train_t.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    for _ in range(100):
        model.train()
        loss = criterion(model(X_train_t).squeeze(), y_train_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds     = model(X_test_t).squeeze().numpy()
        preds_bin = (preds > 0.5).astype(float)

    fpr, tpr, _ = roc_curve(y_test.values, preds)
    roc_auc     = auc(fpr, tpr)

    all_fpr.append(fpr)
    all_tpr.append(tpr)
    all_auc.append(roc_auc)
    all_preds_binary.extend(preds_bin)
    all_labels.extend(y_test.values)

# =========================
# PLOT 1: ROC CURVES
# =========================
plt.figure(figsize=(8, 6))
colors = ["steelblue", "darkorange", "green", "red", "purple"]

for i in range(5):
    plt.plot(all_fpr[i], all_tpr[i], color=colors[i], lw=1.5,
             label=f"Fold {i+1} (AUC = {all_auc[i]:.3f})")

plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — MLP (5-Fold Cross Validation)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
plt.show()
print("Saved: roc_curve.png")

# =========================
# PLOT 2: CONFUSION MATRIX (aggregated across all folds)
# =========================
cm = confusion_matrix(all_labels, all_preds_binary)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["No Appendicitis", "Appendicitis"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix — MLP (Aggregated, 5-Fold)")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("Saved: confusion_matrix.png")

# =========================
# PLOT 3: METRICS BAR CHART PER FOLD
# =========================
from sklearn.metrics import accuracy_score, f1_score

fold_accuracies, fold_f1s = [], []
kf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for i, (train_idx, test_idx) in enumerate(kf2.split(X, y)):
    fold_accuracies.append(accuracy_score(all_labels[i*len(test_idx):(i+1)*len(test_idx)],
                                          all_preds_binary[i*len(test_idx):(i+1)*len(test_idx)]))
    fold_f1s.append(f1_score(all_labels[i*len(test_idx):(i+1)*len(test_idx)],
                             all_preds_binary[i*len(test_idx):(i+1)*len(test_idx)]))

x = np.arange(5)
width = 0.25
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - width, fold_accuracies, width, label="Accuracy",  color="steelblue")
ax.bar(x,         fold_f1s,        width, label="F1 Score",  color="darkorange")
ax.bar(x + width, all_auc,         width, label="ROC-AUC",   color="green")

ax.set_xticks(x)
ax.set_xticklabels([f"Fold {i+1}" for i in range(5)])
ax.set_ylim(0.7, 1.0)
ax.set_ylabel("Score")
ax.set_title("MLP Performance Per Fold")
ax.legend()
plt.tight_layout()
plt.savefig("metrics_per_fold.png", dpi=150)
plt.show()
print("Saved: metrics_per_fold.png")