import librosa
import numpy as np
import pandas as pd
import os
from config import LABEL_NAMES

AUDIO_ROOT = "data/word_level"
FEATURE_DIR = "features"
os.makedirs(FEATURE_DIR, exist_ok=True)

# load splits
train_df = pd.read_csv("data/train.csv")
val_df   = pd.read_csv("data/val.csv")
test_df  = pd.read_csv("data/test.csv")

folder_map = {
    "REP": "word_rep",
    "INS": "word_ins",
    "DEL": "word_del",
    "PAU": "word_pau",
    "SUB": "word_sub"
}

WINDOW = 0.2

def extract_features(df_split):
    features = []

    for _, row in df_split.iterrows():
        label_name = LABEL_NAMES[row["label"]]
        folder = folder_map[label_name]

        path = os.path.join(AUDIO_ROOT, folder, f"{row['audio_id']}.wav")
        if not os.path.exists(path):
            continue

        try:
            y, sr = librosa.load(path, sr=None)

            start = max(0, int((row["start"] - WINDOW) * sr))
            end   = min(len(y), int((row["end"] + WINDOW) * sr))

            segment = y[start:end]

            if len(segment) < sr * 0.05:
                continue

            # MFCC
            mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
            delta = librosa.feature.delta(mfcc)
            delta2 = librosa.feature.delta(mfcc, order=2)

            feat = np.concatenate([
                np.mean(mfcc, axis=1),
                np.std(mfcc, axis=1),
                np.mean(delta, axis=1),
                np.mean(delta2, axis=1),
            ])

            # additional features
            zcr = librosa.feature.zero_crossing_rate(segment)
            energy = np.mean(segment ** 2)

            feat = np.concatenate([feat, [np.mean(zcr), energy]])

            features.append(np.append(feat, row["label"]))

        except Exception as e:
            print(f"Error processing {row['audio_id']}: {e}")

    return np.array(features)


print("Extracting train features:")
train_features = extract_features(train_df)
np.save(os.path.join(FEATURE_DIR, "train_features.npy"), train_features)

print("Extracting validation features:")
val_features = extract_features(val_df)
np.save(os.path.join(FEATURE_DIR, "val_features.npy"), val_features)

print("Extracting test features:")
test_features = extract_features(test_df)
np.save(os.path.join(FEATURE_DIR, "test_features.npy"), test_features)

print("Feature extraction complete.")