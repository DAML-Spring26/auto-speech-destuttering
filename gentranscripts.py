import os
import torch
import whisper
from tqdm import tqdm

# 1. Check if GPU is actually being used
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 2. Load model once
model = whisper.load_model("base", device=device)

input_dir = "data/word_level/word_sub"
output_dir = "transcripts/word_sub_transcripts_json"
os.makedirs(output_dir, exist_ok=True)

# 3. Get list of files
all_audio_files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]
audio_files = all_audio_files[0:100]

# 4. Process loop
for filename in tqdm(audio_files, desc="Transcribing"):
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename.replace(".wav", ".json"))
    
    # Skip if already done (useful if the script crashes halfway)
    if os.path.exists(output_path):
        continue

    # Transcribe with word-level timestamps
    # fp16=True
    result = model.transcribe(input_path, word_timestamps=True, fp16=True)

    # Save ONLY the JSON (no clutter)
    import json
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)