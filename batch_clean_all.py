import os
import argparse

from nemo.nemo_model import NeMoSEDTextModel
from batch_clean_audio import clean_one  # reuse your existing function

LABEL_FOLDERS = [
    ("word_del", "word_del_transcripts_json"),
    ("word_ins", "word_ins_transcripts_json"),
    ("word_pau", "word_pau_transcripts_json"),
    ("word_rep", "word_rep_transcripts_json"),
    ("word_sub", "word_sub_transcripts_json"),
]

def iter_wavs(audio_dir):
    for name in os.listdir(audio_dir):
        if name.lower().endswith(".wav"):
            yield os.path.join(audio_dir, name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--audio_root", default="data/word_level")
    ap.add_argument("--json_root", default="transcripts")
    ap.add_argument("--out_root", default="outputs/clean_wavs")

    ap.add_argument("--window_k", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--remove_labels", default="PAU")
    ap.add_argument("--pad_ms", type=float, default=30.0)
    ap.add_argument("--crossfade_ms", type=float, default=20.0)

    args = ap.parse_args()

    remove_labels = {x.strip().upper() for x in args.remove_labels.split(",") if x.strip()}

    model = NeMoSEDTextModel.load_from_checkpoint(args.checkpoint)
    model.eval()

    os.makedirs(args.out_root, exist_ok=True)

    total = 0
    ok = 0
    skipped = 0

    for audio_folder, json_folder in LABEL_FOLDERS:
        audio_dir = os.path.join(args.audio_root, audio_folder)
        json_dir = os.path.join(args.json_root, json_folder)
        out_dir = os.path.join(args.out_root, audio_folder)
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.isdir(audio_dir):
            print(f"[SKIP FOLDER] missing audio dir: {audio_dir}")
            continue
        if not os.path.isdir(json_dir):
            print(f"[SKIP FOLDER] missing json dir: {json_dir}")
            continue

        for wav_path in iter_wavs(audio_dir):
            total += 1
            stem = os.path.splitext(os.path.basename(wav_path))[0]
            json_path = os.path.join(json_dir, f"{stem}.json")
            if not os.path.exists(json_path):
                skipped += 1
                continue

            out_wav = os.path.join(out_dir, f"{stem}_clean.wav")
            try:
                clean_one(
                    model=model,
                    wav_path=wav_path,
                    json_path=json_path,
                    out_wav=out_wav,
                    window_k=args.window_k,
                    threshold=args.threshold,
                    remove_labels=remove_labels,
                    pad_ms=args.pad_ms,
                    crossfade_ms=args.crossfade_ms,
                )
                ok += 1
            except Exception as e:
                print(f"[FAIL] {audio_folder}/{stem}: {e}")

        print(f"[DONE FOLDER] {audio_folder} -> {out_dir}")

    print(f"\nDone. total_wavs={total} cleaned={ok} skipped_no_json={skipped}")

if __name__ == "__main__":
    main()