import os
import json
import pandas as pd
from jiwer import wer

GOLD_ROOT = "transcripts/gold_fluent"
CLEAN_ROOT = "cleaned_transcripts"
ORIG_ROOT = "transcripts"

LABELS = [
    "word_rep",
    "word_ins",
    "word_del",
    "word_pau",
    "word_sub"
]

rows = []

for label in LABELS:
    print(f"Processing {label}...")

    gold_dir = os.path.join(GOLD_ROOT, label)
    clean_dir = os.path.join(CLEAN_ROOT, label)
    orig_dir = os.path.join(ORIG_ROOT, f"{label}_transcripts_json")

    for fname in os.listdir(clean_dir):
        if not fname.endswith(".json"):
            continue

        id = fname.replace("_clean.json", "").replace(".json", "")

        gold_path = os.path.join(gold_dir, id + ".fluent.json")
        clean_path = os.path.join(clean_dir, fname)
        orig_path = os.path.join(orig_dir, id + ".json")

        if not (os.path.exists(gold_path) and os.path.exists(orig_path)):
            continue

        try:
            with open(gold_path) as f:
                gold = json.load(f)["fluent_text"]

            with open(clean_path) as f:
                clean = json.load(f)["text"]

            with open(orig_path) as f:
                orig = json.load(f)["text"]

            wer_clean = wer(gold, clean)
            wer_orig = wer(gold, orig)

            rows.append({
                "id": id,
                "label_type": label,
                "WER_clean": wer_clean,
                "WER_baseline": wer_orig
            })

        except Exception as e:
            print(f"Error on {id}: {e}")

df = pd.DataFrame(rows)

# Save CSV
df.to_csv("wer_results.csv", index=False)

print("\nSaved wer_results.csv")

# ===== Stats =====
print("\n=== OVERALL WER ===")
print("Baseline:", df["WER_baseline"].mean())
print("Cleaned :", df["WER_clean"].mean())

print("\n=== WER BY LABEL ===")
print(df.groupby("label_type")[["WER_baseline", "WER_clean"]].mean())