import os
import sys
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nemo.nemo_model import NeMoSEDTextModel, LABEL_NAMES

TRANSCRIPT_ROOT = os.path.join(os.path.dirname(__file__), "..", "transcripts")
FOLDER_MAP = {
    "REP": "word_rep_transcripts_json",
    "INS": "word_ins_transcripts_json",
    "DEL": "word_del_transcripts_json",
    "PAU": "word_pau_transcripts_json",
    "SUB": "word_sub_transcripts_json",
}


def load_from_transcript(audio_id: str, label: str) -> dict:
    folder = FOLDER_MAP.get(label.upper())
    if not folder:
        raise ValueError(f"Unknown label: {label}. Choose from REP/INS/DEL/PAU/SUB")
    path = os.path.join(TRANSCRIPT_ROOT, folder, f"{audio_id}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Transcript not found: {path}")
    with open(path) as f:
        data = json.load(f)
    seg  = data["segments"][0]
    wc   = [w["probability"] for w in seg.get("words", [])]
    return {
        "text":              data["text"].strip().lower(),
        "duration":          seg["end"] - seg["start"],
        "avg_logprob":       seg.get("avg_logprob", -0.3),
        "no_speech_prob":    seg.get("no_speech_prob", 0.0),
        "compression_ratio": seg.get("compression_ratio", 1.0),
        "word_confidences":  wc,
    }


def main():
    parser = argparse.ArgumentParser(description="NeMo SED inference")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text",       default=None, help="Raw transcript text")
    parser.add_argument("--audio_id",   default=None, help="Audio ID (needs --label)")
    parser.add_argument("--label",      default=None, help="Label folder e.g. DEL")
    args = parser.parse_args()

    #load model
    model = NeMoSEDTextModel.load_from_checkpoint(args.checkpoint)
    model.eval()
    print("Model loaded.\n")

    #build record
    if args.text:
        record = {
            "text": args.text.strip().lower(),
            "duration": len(args.text.split()) * 0.3,
            "avg_logprob": -0.3,
            "no_speech_prob": 0.0,
            "compression_ratio": 1.0,
            "word_confidences": [0.9] * len(args.text.split()),
        }
    elif args.audio_id and args.label:
        record = load_from_transcript(args.audio_id, args.label)
        print(f"Transcript: \"{record['text']}\"")
    else:
        parser.error("Provide either --text or both --audio_id and --label")

    #prediction
    result = model.predict_from_manifest_record(record)

    print(f"\nPrediction")
    print(f"  Dysfluency : {result['dysfluency_name']}  (label {result['dysfluency_label']})")
    print(f"  Confidence : {result['confidence']:.2%}")
    print(f"\n  Class probabilities:")
    for name, prob in result["all_probs"].items():
        bar = "█" * int(prob * 40)
        print(f"    {name:>3s}  {prob:.4f}  {bar}")


if __name__ == "__main__":
    main()
