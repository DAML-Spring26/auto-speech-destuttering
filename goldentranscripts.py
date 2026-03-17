import os
import json

INPUT_DIR = "transcripts/word_sub_transcripts_json"
OUTPUT_DIR = "transcripts/word_sub_fluent_json"

os.makedirs(OUTPUT_DIR, exist_ok=True)

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

    for segment in data["segments"]:
        words = segment.get("words", [])

        # Remove fillers
        words = [w for w in words if not is_filler(w["word"])]

        # Collapse repetitions
        words = remove_repetitions(words)

        fluent_words.extend(words)

    fluent_text = " ".join(w["word"].strip() for w in fluent_words)

    return fluent_text, fluent_words

for fname in os.listdir(INPUT_DIR):
    if not fname.endswith(".json"):
        continue

    with open(os.path.join(INPUT_DIR, fname), "r", encoding="utf-8") as f:
        data = json.load(f)

    fluent_text, fluent_words = make_fluent(data)

    out = {
        "audio_id": fname.replace(".json", ""),
        "verbatim_text": data["text"].strip(),
        "fluent_text": fluent_text,
        "fluent_words": fluent_words
    }

    out_path = os.path.join(
        OUTPUT_DIR,
        fname.replace(".json", ".fluent.json")
    )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4)