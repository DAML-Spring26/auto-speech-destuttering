import os
import whisper
import json
from tqdm import tqdm

CLEAN_ROOT = "outputs/clean_wavs"
OUT_ROOT = "cleaned_transcripts"

model = whisper.load_model("base")

LABELS = [
    "word_rep",
    "word_ins",
    "word_del",
    "word_pau",
    "word_sub"
]

for label in LABELS:
    input_dir = os.path.join(CLEAN_ROOT, label)
    output_dir = os.path.join(OUT_ROOT, label)

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nTranscribing {label}...")

    for fname in tqdm(os.listdir(input_dir)):
        if not fname.endswith(".wav"):
            continue

        input_path = os.path.join(input_dir, fname)
        out_name = fname.replace(".wav", ".json")
        output_path = os.path.join(output_dir, out_name)

        if os.path.exists(output_path):
            continue

        result = model.transcribe(input_path)

        with open(output_path, "w") as f:
            json.dump(result, f, indent=4)

print("Done transcribing cleaned audio.")