import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix
)

# load data
train_data = np.load("features/train_features.npy")
val_data = np.load("features/val_features.npy")

X_train = train_data[:, :-1]
y_train = train_data[:, -1]

X_val = val_data[:, :-1]
y_val = val_data[:, -1]

# train
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
print("Model trained.\n")

# eval
val_preds = model.predict(X_val)

print("Validation Accuracy:", accuracy_score(y_val, val_preds))
print("\nClassification Report:")
print(classification_report(y_val, val_preds))

# Confusion Matrix
cm = confusion_matrix(y_val, val_preds)
print("Confusion Matrix:\n", cm)

# Heatmap Confusion Matrix
label_names = ["REP", "INS", "DEL", "PAU", "SUB"]

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_names,
    yticklabels=label_names
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Validation Confusion Matrix")
plt.tight_layout()

# save confusion matrix
os.makedirs("models", exist_ok=True)
plt.savefig("models/val_confusion_matrix.png")
plt.show()

# save model
joblib.dump(model, "models/dysfluency_rf.pkl")
print("\nModel saved to models/dysfluency_rf.pkl")