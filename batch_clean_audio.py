import os
import json
import argparse
from glob import glob
import soundfile as sf

from nemo.nemo_model import NeMoSEDTextModel
from audio_stitch import merge_spans, complement_spans, stitch_with_crossfade


def extract_words(whisper_json: dict):
    words = []
    for seg in whisper_json.get("segments", []):
        for w in seg.get("words", []):
            if "start" in w and "end" in w:
                words.append({
                    "word": w.get("word", ""),
                    "start": float(w["start"]),
                    "end": float(w["end"]),
                    "prob": float(w.get("probability", 0.9)),
                })
    return words


def build_window_text(words, i, k):
    a = max(0, i - k)
    b = min(len(words), i + k + 1)
    toks = [words[j]["word"].strip().lower() for j in range(a, b)]
    return " ".join([t for t in toks if t]).strip()


def clean_one(model, wav_path, json_path, out_wav,
              window_k, threshold, remove_labels,
              pad_ms, crossfade_ms):

    y, sr = sf.read(wav_path, always_2d=False)
    total_dur_s = len(y) / sr

    with open(json_path) as f:
        tj = json.load(f)

    words = extract_words(tj)
    if not words:
        raise ValueError("No word timestamps found")

    seg0 = tj.get("segments", [{}])[0]
    avg_logprob = float(seg0.get("avg_logprob", -0.3))
    no_speech_prob = float(seg0.get("no_speech_prob", 0.0))
    compression_ratio = float(seg0.get("compression_ratio", 1.0))

    pad = pad_ms / 1000.0
    remove_spans = []

    for i in range(len(words)):
        window_text = build_window_text(words, i, window_k)
        if not window_text:
            continue

        s_center = max(0.0, words[i]["start"] - pad)
        e_center = min(total_dur_s, words[i]["end"] + pad)

        record = {
            "text": window_text,
            "duration": max(0.01, e_center - s_center),
            "avg_logprob": avg_logprob,
            "no_speech_prob": no_speech_prob,
            "compression_ratio": compression_ratio,
            "word_confidences": [words[i]["prob"]],
        }

        pred = model.predict_from_manifest_record(record)
        lab = str(pred["dysfluency_name"]).upper()
        conf = float(pred["confidence"])

        if lab in remove_labels and conf >= threshold:
            remove_spans.append((s_center, e_center))

    remove_spans = merge_spans(remove_spans)
    keep = complement_spans(remove_spans, total_dur_s)

    y_clean = stitch_with_crossfade(y, sr, keep, crossfade_ms=crossfade_ms)
    sf.write(out_wav, y_clean, sr)

    return len(remove_spans)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--whisper_json_dir", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--window_k", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--remove_labels", default="PAU")
    ap.add_argument("--pad_ms", type=float, default=30.0)
    ap.add_argument("--crossfade_ms", type=float, default=20.0)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    model = NeMoSEDTextModel.load_from_checkpoint(args.checkpoint)
    model.eval()

    remove_labels = {x.strip().upper() for x in args.remove_labels.split(",")}

    wavs = sorted(glob(os.path.join(args.audio_dir, "*.wav")))

    for wav_path in wavs:
        stem = os.path.splitext(os.path.basename(wav_path))[0]
        json_path = os.path.join(args.whisper_json_dir, f"{stem}.json")

        if not os.path.exists(json_path):
            print(f"[SKIP] No JSON for {stem}")
            continue

        out_wav = os.path.join(args.out_dir, f"{stem}_clean.wav")

        n_removed = clean_one(
            model,
            wav_path,
            json_path,
            out_wav,
            args.window_k,
            args.threshold,
            remove_labels,
            args.pad_ms,
            args.crossfade_ms,
        )

        print(f"[OK] {stem}: removed_spans={n_removed}")

    print("\nBatch cleaning complete.")


if __name__ == "__main__":
    main()