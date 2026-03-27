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

        for seg in data.get("segments", []):
            seg_text = seg.get("text", "").strip()

            for w in seg.get("words", []):
                if "start" not in w or "end" not in w:
                    continue

                rows.append({
                    "audio_id": audio_id,
                    "word": w.get("word", "").strip(),
                    "start": float(w["start"]),
                    "end": float(w["end"]),
                    "duration": float(w["end"] - w["start"]),
                    "text": seg_text,   # context
                    "label": label
                })

df = pd.DataFrame(rows)

os.makedirs("data", exist_ok=True)
df.to_csv("data/labeled_segments.csv", index=False)

print(f"Dataset built successfully. Total rows: {len(df)}")