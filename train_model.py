import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix
)
from sklearn.utils import resample

train_data = np.load("features/train_features.npy")
val_data = np.load("features/val_features.npy")

X_train = train_data[:, :-1]
y_train = train_data[:, -1].astype(int)

X_val = val_data[:, :-1]
y_val = val_data[:, -1].astype(int)

print("Before balancing:", dict(zip(*np.unique(y_train, return_counts=True))))

Xy = np.column_stack((X_train, y_train))
classes = np.unique(y_train)

max_count = max([np.sum(y_train == c) for c in classes])

target_size = int(max_count * 0.7)

balanced_data = []

for c in classes:
    class_data = Xy[Xy[:, -1] == c]
    
    balanced_class = resample(
        class_data,
        replace=True,
        n_samples=target_size,
        random_state=42
    )
    
    balanced_data.append(balanced_class)

balanced_data = np.vstack(balanced_data)
np.random.shuffle(balanced_data)

X_train = balanced_data[:, :-1]
y_train = balanced_data[:, -1].astype(int)

print("After balancing:", dict(zip(*np.unique(y_train, return_counts=True))))

class_weights_manual = {
    0: 1.0,
    1: 1.0,
    2: 1.8,
    3: 1.3,
    4: 0.85
}

sample_weights = np.array([class_weights_manual[y] for y in y_train])

print("Manual class weights:", class_weights_manual)

model = XGBClassifier(
    n_estimators=1000,
    max_depth=8,            
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.5,               
    min_child_weight=3,      
    objective="multi:softprob",
    num_class=len(classes),
    random_state=42,
    n_jobs=-1,
    eval_metric="merror",     
    early_stopping_rounds=50
)

model.fit(
    X_train,
    y_train,
    sample_weight=sample_weights,
    eval_set=[(X_val, y_val)],
    verbose=True
)

print("\nModel trained.")

val_probs = model.predict_proba(X_val)
val_preds = np.argmax(val_probs, axis=1)

print("\nValidation Accuracy:", accuracy_score(y_val, val_preds))
print("\nClassification Report:")
print(classification_report(y_val, val_preds, digits=4))

cm = confusion_matrix(y_val, val_preds)
print("\nConfusion Matrix:\n", cm)

label_names_full = ["REP", "INS", "DEL", "PAU", "SUB"]
label_map = dict(zip(classes, label_names_full))
label_names = [label_map[c] for c in classes]

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
plt.title("Validation Confusion Matrix (Improved XGBoost)")
plt.tight_layout()

os.makedirs("models", exist_ok=True)
plt.savefig("models/val_confusion_matrix_improved.png")
plt.show()

joblib.dump(
    {
        "model": model,
        "classes": classes
    },
    "models/dysfluency_xgb_improved.pkl"
)

print("\nModel saved to models/dysfluency_xgb_improved.pkl")