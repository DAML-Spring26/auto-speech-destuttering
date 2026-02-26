import librosa
import numpy as np
import pandas as pd
import os
from config import LABEL_NAMES

AUDIO_ROOT = "data/word_level"
FEATURE_DIR = "features"
os.makedirs(FEATURE_DIR, exist_ok=True)

# load train/val/test splits
train_df = pd.read_csv("data/train.csv")
val_df   = pd.read_csv("data/val.csv")
test_df  = pd.read_csv("data/test.csv")  # optional

folder_map = {
    "REP": "word_rep",
    "INS": "word_ins",
    "DEL": "word_del",
    "PAU": "word_pau",
    "SUB": "word_sub"
}

# extract MFCC
def extract_features(df_split):
    features = []
    for _, row in df_split.iterrows():
        label_name = LABEL_NAMES[row["label"]]
        folder = folder_map[label_name]

        path = os.path.join(AUDIO_ROOT, folder, f"{row['audio_id']}.wav")
        if not os.path.exists(path):
            continue

        y, sr = librosa.load(path, sr=None)
        segment = y[int(row["start"] * sr):int(row["end"] * sr)]

        if len(segment) < sr * 0.1:
            continue

        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)

        features.append(np.append(mfcc_mean, row["label"]))
    return np.array(features)

# extract/save features
print("Extracting train features:")
train_features = extract_features(train_df)
np.save(os.path.join(FEATURE_DIR, "train_features.npy"), train_features)

print("Extracting validation features:")
val_features = extract_features(val_df)
np.save(os.path.join(FEATURE_DIR, "val_features.npy"), val_features)

print("Feature extraction complete")