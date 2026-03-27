import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("data/labeled_segments.csv")

# -----------------------------
# STEP 1: Audio-level label distribution
# -----------------------------
audio_stats = (
    df.groupby("audio_id")["label"]
    .value_counts(normalize=True)
    .unstack(fill_value=0)
)

# Ensure consistent column order
audio_stats = audio_stats.sort_index(axis=1)

# -----------------------------
# STEP 2: Dominant label per audio
# -----------------------------
audio_stats["dominant_label"] = audio_stats.idxmax(axis=1)

# -----------------------------
# STEP 3: Stratified split
# -----------------------------
train_ids, temp_ids = train_test_split(
    audio_stats.index,
    test_size=0.4,
    stratify=audio_stats["dominant_label"],
    random_state=42
)

val_ids, test_ids = train_test_split(
    temp_ids,
    test_size=0.5,
    stratify=audio_stats.loc[temp_ids, "dominant_label"],
    random_state=42
)

# -----------------------------
# STEP 4: Build datasets
# -----------------------------
train = df[df["audio_id"].isin(train_ids)]
val = df[df["audio_id"].isin(val_ids)]
test = df[df["audio_id"].isin(test_ids)]

# -----------------------------
# STEP 5: Debug distributions
# -----------------------------
print("\nClass distribution:")
print("Train:", Counter(train["label"]))
print("Val:", Counter(val["label"]))
print("Test:", Counter(test["label"]))

print("\nUnique audio files:")
print("Train:", len(train_ids))
print("Val:", len(val_ids))
print("Test:", len(test_ids))

# -----------------------------
# STEP 6: Save splits
# -----------------------------
train.to_csv("data/train.csv", index=False)
val.to_csv("data/val.csv", index=False)
test.to_csv("data/test.csv", index=False)

print("\nSplit complete ✅")