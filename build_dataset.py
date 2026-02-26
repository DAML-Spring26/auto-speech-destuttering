import os
import json
import pandas as pd
from config import LABEL_MAP

TRANSCRIPT_ROOT = "transcripts"
rows = []

for folder in os.listdir(TRANSCRIPT_ROOT):
    if folder not in LABEL_MAP:
        continue

    label = LABEL_MAP[folder]
    folder_path = os.path.join(TRANSCRIPT_ROOT, folder)

    for file in os.listdir(folder_path):
        if not file.endswith(".json"):
            continue

        file_path = os.path.join(folder_path, file)

        with open(file_path, "r") as f:
            data = json.load(f)

        audio_id = file.replace(".json", "")

        for seg in data["segments"]:
            rows.append({
                "audio_id": audio_id,
                "start": seg["start"],
                "end": seg["end"],
                "duration": seg["end"] - seg["start"],
                "text": seg["text"].strip(),
                "label": label
            })

df = pd.DataFrame(rows)

os.makedirs("data", exist_ok=True)
df.to_csv("data/labeled_segments.csv", index=False)

print("Dataset built successfully.")
