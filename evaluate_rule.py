import pandas as pd
from rule_based import rule_detect
from sklearn.metrics import classification_report

df = pd.read_csv("data/test.csv")

y_true = []
y_pred = []

for _, row in df.iterrows():
    pred = rule_detect(row["text"])

    y_true.append(row["label"])
    y_pred.append(pred)

print("Total test rows:", len(df))
print(classification_report(y_true, y_pred, zero_division=0))