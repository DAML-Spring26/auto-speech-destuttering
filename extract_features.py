import librosa
import numpy as np
import pandas as pd
import os
from config import LABEL_NAMES

AUDIO_ROOT = "data/word_level"

df = pd.read_csv("data/train.csv")

features = []

for _, row in df.iterrows():

    # Recover folder name from label
    label_name = LABEL_NAMES[row["label"]]

    # Map back to original audio folder name
    folder_map = {
        "REP": "word_rep",
        "INS": "word_ins",
        "DEL": "word_del",
        "PAU": "word_pau",
        "SUB": "word_sub"
    }

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

features = np.array(features)

os.makedirs("features", exist_ok=True)
np.save("features/train_features.npy", features)

print("Features extracted")