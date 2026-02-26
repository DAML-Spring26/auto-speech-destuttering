import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/labeled_segments.csv")

audio_ids = df["audio_id"].unique()

train_ids, temp_ids = train_test_split(audio_ids, test_size=0.3, random_state=42)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

train = df[df["audio_id"].isin(train_ids)]
val = df[df["audio_id"].isin(val_ids)]
test = df[df["audio_id"].isin(test_ids)]

train.to_csv("data/train.csv", index=False)
val.to_csv("data/val.csv", index=False)
test.to_csv("data/test.csv", index=False)

print("Split complete")