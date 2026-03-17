import os
import json

TRANSCRIPT_ROOT = "transcripts"
OUTPUT_ROOT = "transcripts/gold_fluent"

LABEL_FOLDERS = [
    "word_rep_transcripts_json",
    "word_ins_transcripts_json",
    "word_del_transcripts_json",
    "word_pau_transcripts_json",
    "word_sub_transcripts_json"
]

FILLERS = {"uh", "um", "erm", "ah", "eh", "mm", "hmm"}

def normalize_word(w):
    return w.strip().lower().strip(".,!?")

def is_filler(word):
    return normalize_word(word) in FILLERS

def remove_repetitions(words):
    cleaned = []
    prev = None
    for w in words:
        norm = normalize_word(w["word"])
        if prev is None or norm != prev:
            cleaned.append(w)
        prev = norm
    return cleaned

def make_fluent(data):
    fluent_words = []

    for segment in data.get("segments", []):
        words = segment.get("words", [])

        # Remove fillers
        words = [w for w in words if not is_filler(w["word"])]

        # Collapse repetitions
        words = remove_repetitions(words)

        fluent_words.extend(words)

    fluent_text = " ".join(w["word"].strip() for w in fluent_words)
    return fluent_text


for folder in LABEL_FOLDERS:
    input_dir = os.path.join(TRANSCRIPT_ROOT, folder)
    label_name = folder.replace("_transcripts_json", "")
    output_dir = os.path.join(OUTPUT_ROOT, label_name)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing {folder}...")

    for fname in os.listdir(input_dir):
        if not fname.endswith(".json"):
            continue

        input_path = os.path.join(input_dir, fname)

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        fluent_text = make_fluent(data)

        out = {
            "audio_id": fname.replace(".json", ""),
            "fluent_text": fluent_text
        }

        out_path = os.path.join(
            output_dir,
            fname.replace(".json", ".fluent.json")
        )

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=4)

print("Gold transcripts generated for all labels.")