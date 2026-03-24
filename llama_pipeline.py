import os
import sys
import json
import csv
import argparse
import time
from typing import Optional

import requests

sys.path.insert(0, os.path.dirname(__file__))
from config import LABEL_NAMES

LABEL_LIST = [LABEL_NAMES[i] for i in range(len(LABEL_NAMES))]

TRANSCRIPT_ROOT = os.path.join(os.path.dirname(__file__), "transcripts")
DATA_DIR        = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR      = os.path.join(os.path.dirname(__file__), "outputs")

FOLDER_MAP = {
    "REP": "word_rep_transcripts_json",
    "INS": "word_ins_transcripts_json",
    "DEL": "word_del_transcripts_json",
    "PAU": "word_pau_transcripts_json",
    "SUB": "word_sub_transcripts_json",
}

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3"

#call API
def call_llama(prompt: str, model: str = DEFAULT_MODEL, temperature: float = 0.1) -> str:
    """
    Send a prompt to a locally running Ollama Llama model.
    Returns the response text, or raises if Ollama is not running.
    """
    payload = {
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["response"].strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot reach Ollama. Make sure it is running:\n"
            "  ollama serve\n"
            "And that you have pulled a model:\n"
            f"  ollama pull {model}"
        )

def build_classify_prompt(transcript: str) -> str:
    return f"""You are a speech-language pathology assistant specialising in dysfluency detection.

Given the following speech transcript, classify it into exactly ONE of these dysfluency types:
  REP - Word Repetition: a word or phrase is repeated (e.g. "the the cat")
  INS - Word Insertion: an extra word is inserted that should not be there
  DEL - Word Deletion: a word is missing that should be present
  PAU - Pause/Filler: an unnatural pause or filler word (e.g. "um", "uh", very short utterance)
  SUB - Word Substitution: a word is replaced with a wrong word

Transcript: "{transcript}"

Respond in this exact JSON format with no extra text:
{{"label": "<REP|INS|DEL|PAU|SUB>", "confidence": <0.0-1.0>, "reason": "<one sentence>"}}"""


def build_clean_prompt(transcript: str, dysfluency_type: str) -> str:
    descriptions = {
        "REP": "contains a word repetition",
        "INS": "contains an incorrectly inserted word",
        "DEL": "is missing a word",
        "PAU": "contains an unnatural pause or filler",
        "SUB": "contains a word substituted with the wrong word",
    }
    desc = descriptions.get(dysfluency_type, "contains a dysfluency")
    return f"""You are a speech correction assistant.

The following transcript {desc}. Rewrite it as a natural, fluent sentence with the dysfluency corrected.
Only return the corrected sentence — no explanation, no quotes, no extra text.

Original: "{transcript}"

Corrected:"""


def build_explain_prompt(transcript: str, dysfluency_type: str, cleaned: str) -> str:
    return f"""You are a speech-language pathology assistant.

Original transcript:  "{transcript}"
Dysfluency detected:  {dysfluency_type}
Corrected version:    "{cleaned}"

In 2-3 sentences, explain to a clinician:
1. What specific dysfluency was detected and where in the sentence
2. What the likely correction is and why
Keep it concise and clinical."""

#pipeline
def run_pipeline(
    transcript: str,
    true_label: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    verbose: bool = True,
) -> dict:
    """
    Run all three Llama tasks on a single transcript string.

    Returns:
        {
            "transcript":    original text,
            "true_label":    ground truth label (if known),
            "pred_label":    Llama predicted label,
            "confidence":    Llama confidence 0-1,
            "classify_reason": one-sentence reason,
            "cleaned":       fluent corrected transcript,
            "explanation":   clinical explanation,
        }
    """
    result = {"transcript": transcript, "true_label": true_label or "unknown"}

    #classify
    if verbose:
        print("[1/3] Classifying dysfluency...")
    classify_raw = call_llama(build_classify_prompt(transcript), model=model)

    try:
        clean_json = classify_raw.replace("```json", "").replace("```", "").strip()
        classify_data = json.loads(clean_json)
        pred_label  = classify_data.get("label", "UNK").upper()
        confidence  = float(classify_data.get("confidence", 0.0))
        reason      = classify_data.get("reason", "")
    except (json.JSONDecodeError, ValueError):
        # Fallback to try to find a label keyword in response
        pred_label = "UNK"
        confidence = 0.0
        reason     = classify_raw[:200]
        for lbl in LABEL_LIST:
            if lbl in classify_raw.upper():
                pred_label = lbl
                break

    result["pred_label"]       = pred_label
    result["confidence"]       = round(confidence, 3)
    result["classify_reason"]  = reason

    #clean
    if verbose:
        print("[2/3] Generating clean transcript...")
    cleaned = call_llama(build_clean_prompt(transcript, pred_label), model=model)
    # Strip any accidental quotes
    cleaned = cleaned.strip().strip('"').strip("'")
    result["cleaned"] = cleaned

    #explain
    if verbose:
        print("[3/3] Generating explanation...")
    explanation = call_llama(build_explain_prompt(transcript, pred_label, cleaned), model=model)
    result["explanation"] = explanation

    return result

#load transcript from Whisper
def load_transcript_from_json(audio_id: str, label: str) -> dict:
    folder = FOLDER_MAP.get(label.upper())
    if not folder:
        raise ValueError(f"Unknown label '{label}'. Choose from: {list(FOLDER_MAP.keys())}")
    path = os.path.join(TRANSCRIPT_ROOT, folder, f"{audio_id}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Transcript not found: {path}")
    with open(path) as f:
        data = json.load(f)
    return {"text": data["text"].strip(), "audio_id": audio_id, "label": label}

#print final results
def print_result(result: dict):
    print("\n" + "=" * 65)
    print("Llama Dsyfluency Pipeline — Result")
    print("=" * 65)
    print(f"  Original   : {result['transcript']}")
    if result.get("true_label") and result["true_label"] != "unknown":
        correct = "✓" if result["true_label"] == result["pred_label"] else "✗"
        print(f"  True label : {result['true_label']}  {correct}")
    print(f"  Predicted  : {result['pred_label']}  (confidence {result['confidence']:.0%})")
    print(f"  Reason     : {result['classify_reason']}")
    print(f"\n  Cleaned    : {result['cleaned']}")
    print(f"\n  Explanation:\n  {result['explanation']}")
    print("=" * 65)

def run_batch(split: str, model: str, output_csv: str, limit: Optional[int] = None):
    import pandas as pd

    csv_path = os.path.join(DATA_DIR, f"{split}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Split CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if limit:
        df = df.head(limit)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    results = []
    correct = 0
    total   = 0

    print(f"\nRunning Llama pipeline on {len(df)} {split} samples...")
    print(f"Model: {model}\n")

    for i, row in df.iterrows():
        true_label = LABEL_NAMES[int(row["label"])]
        transcript = str(row["text"]).strip()

        print(f"[{total+1}/{len(df)}] {true_label} | {transcript[:60]}...")

        try:
            result = run_pipeline(transcript, true_label=true_label, model=model, verbose=True)
            results.append(result)

            if result["pred_label"] == true_label:
                correct += 1
            total += 1

            # Print accuracy for every 10 samples
            if total % 10 == 0:
                print(f"\n  Running accuracy: {correct}/{total} = {correct/total:.1%}\n")

        except Exception as e:
            print(f"  ERROR on row {i}: {e}")
            results.append({
                "transcript": transcript, "true_label": true_label,
                "pred_label": "ERROR", "confidence": 0.0,
                "classify_reason": str(e), "cleaned": "", "explanation": "",
            })
            total += 1

        time.sleep(0.1)  #small delay

    #save the csv
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["transcript", "true_label", "pred_label", "confidence",
                      "classify_reason", "cleaned", "explanation"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    #final summary
    accuracy = correct / total if total > 0 else 0
    print(f"\n{'='*50}")
    print(f"  BATCH COMPLETE")
    print(f"  Accuracy : {correct}/{total} = {accuracy:.1%}")
    print(f"  Results  → {output_csv}")
    print(f"{'='*50}")

    #break down per class
    from collections import defaultdict
    class_correct = defaultdict(int)
    class_total   = defaultdict(int)
    for r in results:
        class_total[r["true_label"]] += 1
        if r["pred_label"] == r["true_label"]:
            class_correct[r["true_label"]] += 1
    print("\n  Per-class accuracy:")
    for lbl in LABEL_LIST:
        n = class_total[lbl]
        c = class_correct[lbl]
        bar = "█" * int((c / n * 20) if n > 0 else 0)
        print(f"    {lbl:>3s}  {c:>2}/{n:<2}  {bar}")


#CLI
def main():
    parser = argparse.ArgumentParser(description="Whisper → Llama dysfluency pipeline")
    parser.add_argument("--model",      default=DEFAULT_MODEL,
                        help=f"Ollama model name (default: {DEFAULT_MODEL})")

    #single mode
    single = parser.add_argument_group("Single transcript")
    single.add_argument("--text",     default=None, help="Raw transcript text")
    single.add_argument("--audio_id", default=None, help="Audio ID (needs --label)")
    single.add_argument("--label",    default=None, help="Label e.g. DEL (needs --audio_id)")

    #batch mode
    batch = parser.add_argument_group("Batch mode")
    batch.add_argument("--batch",      action="store_true")
    batch.add_argument("--split",      default="test", choices=["train", "val", "test"])
    batch.add_argument("--output_csv", default=None)
    batch.add_argument("--limit",      type=int, default=None,
                       help="Only process first N rows (for quick testing)")

    args = parser.parse_args()

    if args.batch:
        out_csv = args.output_csv or os.path.join(OUTPUT_DIR, f"llama_{args.split}_results.csv")
        run_batch(args.split, args.model, out_csv, limit=args.limit)

    elif args.text:
        result = run_pipeline(args.text, model=args.model)
        print_result(result)

    elif args.audio_id and args.label:
        record = load_transcript_from_json(args.audio_id, args.label)
        print(f"Transcript: \"{record['text']}\"")
        result = run_pipeline(record["text"], true_label=args.label, model=args.model)
        print_result(result)

    else:
        parser.print_help()
        print("\nExample:")
        print('  python llama_pipeline.py --text "the the quick brown fox"')
        print('  python llama_pipeline.py --audio_id 1003 --label DEL')
        print('  python llama_pipeline.py --batch --split test --limit 20')


if __name__ == "__main__":
    main()