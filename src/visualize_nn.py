import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo # We can reuse the same data loading code as in train_nn.py
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

# LOAD DATA
dataset = fetch_ucirepo(id=938)
X = dataset.data.features
y = dataset.data.targets["Diagnosis"]

y = y.map({"no appendicitis": 0, "appendicitis": 1}).astype(float)
mask = y.notna()
X = X[mask].reset_index(drop=True)
y = y[mask].reset_index(drop=True)
X = pd.get_dummies(X)

# MODEL DEFINITION
class MLP(nn.Module):
    def __init__(self, input_dim): # The model architecture remains the same as in train_nn.py, but we can simplify it since we're only doing binary classification here
        super().__init__()
        self.net = nn.Sequential( # The output layer and activation are fixed for binary classification
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32),        nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1),         nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# K-FOLD + COLLECT RESULTS
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

all_fpr, all_tpr, all_auc = [], [], []
all_preds_binary, all_labels = [], []

for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)): # We can reuse the same data splitting and preprocessing code as in train_nn.py, but we will collect the false positive rates, true positive rates, and AUCs for each fold to plot the ROC curves later
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    imputer = SimpleImputer(strategy="median") # Fit on train, transform both train and test (we could also fit on trainval if we wanted to do a final model later, but for visualization this is fine)
    X_train = imputer.fit_transform(X_train)
    X_test  = imputer.transform(X_test)

    scaler = StandardScaler() # Fit on train, transform both train and test (same reasoning as above)
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32)

    model = MLP(X_train_t.shape[1]) # Initialize the model with the number of input features
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    for _ in range(100): # Train the model on the training set for this fold (we could also do early stopping or more epochs, but this is just for visualization)
        model.train()
        loss = criterion(model(X_train_t).squeeze(), y_train_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad(): # Get predicted probabilities on the test set for this fold, and compute the binary predictions using a 0.5 threshold (we will use the probabilities to compute the ROC curve, and the binary predictions to compute the confusion matrix later)
        preds     = model(X_test_t).squeeze().numpy()
        preds_bin = (preds > 0.5).astype(float)

    fpr, tpr, _ = roc_curve(y_test.values, preds)
    roc_auc     = auc(fpr, tpr)

    all_fpr.append(fpr)
    all_tpr.append(tpr)
    all_auc.append(roc_auc)
    all_preds_binary.extend(preds_bin)
    all_labels.extend(y_test.values)

# PLOT 1: ROC CURVES
plt.figure(figsize=(8, 6))
colors = ["steelblue", "darkorange", "green", "red", "purple"]

for i in range(5): # Plot each fold's ROC curve with its AUC in the legend
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

# PLOT 2: CONFUSION MATRIX (aggregated across all folds)
cm = confusion_matrix(all_labels, all_preds_binary) # We can use sklearn's ConfusionMatrixDisplay to plot the confusion matrix with nice labels and a color map
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["No Appendicitis", "Appendicitis"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix — MLP (Aggregated, 5-Fold)")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("Saved: confusion_matrix.png")

# PLOT 3: METRICS BAR CHART PER FOLD
from sklearn.metrics import accuracy_score, f1_score # We can compute the accuracy and F1 score for each fold using the binary predictions and true labels we collected, and then plot them in a bar chart to compare performance across folds. We can also include the AUC for each fold in the same chart for a comprehensive comparison.

fold_accuracies, fold_f1s = [], []
kf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for i, (train_idx, test_idx) in enumerate(kf2.split(X, y)): # We need to loop through the folds again to compute the metrics for each fold separately, since we only collected the binary predictions and true labels in a single list. We can use the same splitting logic to get the indices for each fold and then compute the metrics using those indices.
    fold_accuracies.append(accuracy_score(all_labels[i*len(test_idx):(i+1)*len(test_idx)],
                                          all_preds_binary[i*len(test_idx):(i+1)*len(test_idx)]))
    fold_f1s.append(f1_score(all_labels[i*len(test_idx):(i+1)*len(test_idx)], 
                             all_preds_binary[i*len(test_idx):(i+1)*len(test_idx)])) # We can compute the accuracy and F1 score for each fold using the corresponding slice of the all_labels and all_preds_binary lists, which contain the true labels and binary predictions for all folds concatenated together. The length of each test set is len(test_idx), so we can use that to slice the lists correctly for each fold.

x = np.arange(5)
width = 0.25
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - width, fold_accuracies, width, label="Accuracy",  color="steelblue")
ax.bar(x,         fold_f1s,        width, label="F1 Score",  color="darkorange")
ax.bar(x + width, all_auc,         width, label="ROC-AUC",   color="green")

# We can set the x-ticks to be the fold numbers, and add labels, title, and legend for clarity. We can also set the y-axis limits to focus on the range of metric values we have (e.g., 0.7 to 1.0) to make the differences more visually apparent.
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