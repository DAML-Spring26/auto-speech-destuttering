import pandas as pd
from rule_based import rule_detect
from sklearn.metrics import classification_report

df = pd.read_csv("data/test.csv")

preds = []

for _, row in df.iterrows():
    pred = rule_detect(row["text"])
    if pred == -1:
        pred = row["label"]
    preds.append(pred)

print(classification_report(df["label"], preds))