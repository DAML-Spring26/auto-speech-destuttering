import os
import random
import torch
import whisper
from tqdm import tqdm
import json

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = whisper.load_model("base", device=device)

base_input_dir = "data/word_level"
base_output_dir = "transcripts/word_level_transcripts_json"

os.makedirs(base_output_dir, exist_ok=True)

classes = ["word_del", "word_ins", "word_pau", "word_rep", "word_sub"]

for cls in classes:
    input_dir = os.path.join(base_input_dir, cls)
    output_dir = os.path.join(base_output_dir, cls)

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_dir):
        print(f"⚠️ Skipping missing folder: {input_dir}")
        continue

    all_audio_files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]

    random.shuffle(all_audio_files)

    audio_files = all_audio_files[:500]

    print(f"{cls}: selected {len(audio_files)} files (out of {len(all_audio_files)})")

    for filename in tqdm(audio_files, desc=f"Transcribing {cls}"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.replace(".wav", ".json"))

        if os.path.exists(output_path):
            continue

        result = model.transcribe(
            input_path,
            word_timestamps=True,
            fp16=(device == "cuda") 
        )
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)