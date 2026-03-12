import os
import sys
import json
import argparse
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import LABEL_MAP, LABEL_NAMES

LABEL_TO_FOLDER = {v: k for k, v in LABEL_MAP.items()}

TRANSCRIPT_TO_AUDIO = {
    "word_rep_transcripts_json": "word_rep",
    "word_ins_transcripts_json": "word_ins",
    "word_del_transcripts_json": "word_del",
    "word_pau_transcripts_json": "word_pau",
    "word_sub_transcripts_json": "word_sub",
}

def load_whisper_meta(transcript_root: str, transcript_folder: str, audio_id: str) -> dict:
    json_path = os.path.join(transcript_root, transcript_folder, f"{audio_id}.json")
    if not os.path.exists(json_path):
        return {}

    with open(json_path) as f:
        data = json.load(f)

    # Aggregate word-level confidences
    word_probs = []
    for seg in data.get("segments", []):
        for w in seg.get("words", []):
            word_probs.append(round(w.get("probability", 0.0), 4))

    # Use first segment for overall quality signals
    first_seg = data["segments"][0] if data.get("segments") else {}

    return {
        "avg_logprob":     round(first_seg.get("avg_logprob", 0.0), 4),
        "no_speech_prob":  round(first_seg.get("no_speech_prob", 0.0), 4),
        "compression_ratio": round(first_seg.get("compression_ratio", 1.0), 4),
        "word_confidences": word_probs,
    }

def build_manifest(
    csv_path: str,
    transcript_root: str,
    audio_root: str,
    output_path: str,
    split_name: str,
) -> int:
    df = pd.read_csv(csv_path)
    written = 0

    with open(output_path, "w") as fout:
        for _, row in df.iterrows():
            label      = int(row["label"])
            label_name = LABEL_NAMES[label]
            t_folder   = LABEL_TO_FOLDER[label]
            a_folder   = TRANSCRIPT_TO_AUDIO[t_folder]
            audio_id   = str(row["audio_id"])

            # Audio path (may not exist — that's OK)
            audio_path = os.path.abspath(
                os.path.join(audio_root, a_folder, f"{audio_id}.wav")
            )

            # Pull Whisper metadata from existing transcripts
            whisper_meta = load_whisper_meta(transcript_root, t_folder, audio_id)

            record = {
                "audio_filepath":   audio_path,
                "duration":         float(row["duration"]),
                "offset":           float(row["start"]),
                "text":             str(row["text"]).strip().lower(),
                "dysfluency_label": label,
                "dysfluency_name":  label_name,
            }
            record.update(whisper_meta)   # add Whisper confidence fields

            fout.write(json.dumps(record) + "\n")
            written += 1

    print(f"  [{split_name}] {written} records → {output_path}")
    return written


def main():
    parser = argparse.ArgumentParser(description="Build NeMo manifest from existing project data")
    parser.add_argument(
        "--project_root",
        default=os.path.join(os.path.dirname(__file__), ".."),
        help="Root of auto-speech-destuttering project (default: parent dir)",
    )
    parser.add_argument(
        "--audio_root",
        default=None,
        help="Path to word-level audio folders (default: <project_root>/data/word_level)",
    )
    args = parser.parse_args()

    root           = os.path.abspath(args.project_root)
    data_dir       = os.path.join(root, "data")
    transcript_dir = os.path.join(root, "transcripts")
    audio_root     = args.audio_root or os.path.join(data_dir, "word_level")
    manifest_dir   = os.path.join(os.path.dirname(__file__), "manifests")

    os.makedirs(manifest_dir, exist_ok=True)

    print("Building NeMo manifest from existing project data")
    print(f"  Project root  : {root}")
    print(f"  Transcripts   : {transcript_dir}")
    print(f"  Audio root    : {audio_root} (may not exist)")
    print()

    splits = {
        "train": os.path.join(data_dir, "train.csv"),
        "val":   os.path.join(data_dir, "val.csv"),
        "test":  os.path.join(data_dir, "test.csv"),
    }

    for split, csv_path in splits.items():
        if not os.path.exists(csv_path):
            print(f"  [skip] {split} CSV not found: {csv_path}")
            continue
        out_path = os.path.join(manifest_dir, f"{split}_manifest.json")
        build_manifest(csv_path, transcript_dir, audio_root, out_path, split)

    print("\ncomplete")

if __name__ == "__main__":
    main()
