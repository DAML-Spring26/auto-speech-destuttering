import os
import sys
import json
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import LABEL_NAMES
from nemo.nemo_model import NeMoSEDTextModel, ManifestDataset, collate_fn

LABEL_LIST   = [LABEL_NAMES[i] for i in range(len(LABEL_NAMES))]
MANIFEST_DIR = os.path.join(os.path.dirname(__file__), "manifests")
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), "outputs")

def evaluate_nemo(model: NeMoSEDTextModel, manifest_path: str):
    """Return (preds, labels, texts) lists for the manifest."""
    ds = ManifestDataset(manifest_path)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate_fn
    )

    all_preds, all_labels, all_texts = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits = model(batch)
            preds  = logits.argmax(dim=-1).cpu().tolist()
            labels = batch["labels"].tolist()
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_texts.extend(batch["texts"])

    return all_preds, all_labels, all_texts

def evaluate_rf(val_features_path: str, test_manifest_path: str):
    """
    Run the existing RF model on val features.
    Returns (preds, labels) from the precomputed val_features.npy.
    """
    import joblib

    rf_path = os.path.join(os.path.dirname(__file__), "..", "models", "dysfluency_rf.pkl")
    feat_path = val_features_path

    if not os.path.exists(rf_path) or not os.path.exists(feat_path):
        return None, None

    model = joblib.load(rf_path)
    data  = np.load(feat_path)
    X, y  = data[:, :-1], data[:, -1].astype(int)
    preds = model.predict(X).astype(int)
    return preds.tolist(), y.tolist()

def plot_confusion_matrix(labels, preds, title: str, save_path: str):
    cm = confusion_matrix(labels, preds, labels=list(range(len(LABEL_LIST))))
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_LIST, yticklabels=LABEL_LIST)
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved → {save_path}")

def print_report(labels, preds, title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(classification_report(labels, preds, target_names=LABEL_LIST, digits=4))
    macro_f1 = f1_score(labels, preds, average="macro")
    print(f"  Macro F1: {macro_f1:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  required=True,
                        help="Path to NeMo SED checkpoint (.ckpt)")
    parser.add_argument("--manifest",    default=None,
                        help="Manifest to evaluate (default: nemo/manifests/test_manifest.json)")
    parser.add_argument("--compare_rf",  action="store_true",
                        help="Also evaluate original RF model for comparison")
    args = parser.parse_args()

    manifest = args.manifest or os.path.join(MANIFEST_DIR, "test_manifest.json")
    if not os.path.exists(manifest):
        manifest = os.path.join(MANIFEST_DIR, "val_manifest.json")
        print(f"  test manifest not found, using: {manifest}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    #load NeMo model
    print(f"Loading NeMo checkpoint: {args.checkpoint}")
    model = NeMoSEDTextModel.load_from_checkpoint(args.checkpoint)
    model.eval()

    #evaluate NeMo
    nemo_preds, nemo_labels, _ = evaluate_nemo(model, manifest)
    print_report(nemo_labels, nemo_preds, "NeMo SED Model — Test Results")
    plot_confusion_matrix(
        nemo_labels, nemo_preds,
        title="NeMo SED — Confusion Matrix",
        save_path=os.path.join(OUTPUT_DIR, "nemo_confusion_matrix.png"),
    )

    #compare with RF
    if args.compare_rf:
        val_feat_path = os.path.join(os.path.dirname(__file__), "..", "features", "val_features.npy")
        rf_preds, rf_labels = evaluate_rf(val_feat_path, manifest)

        if rf_preds is not None:
            print_report(rf_labels, rf_preds, "Original RF Model — Val Results")

            #summary
            nemo_f1 = f1_score(nemo_labels, nemo_preds, average="macro")
            rf_f1   = f1_score(rf_labels,   rf_preds,   average="macro")
            print(f"\n{'─'*40}")
            print(f"  Model Comparison (Macro F1)")
            print(f"{'─'*40}")
            print(f"  NeMo SED  : {nemo_f1:.4f}")
            print(f"  RF (MFCC) : {rf_f1:.4f}")
            delta = nemo_f1 - rf_f1
            print(f"  Delta     : {delta:+.4f} ({'NeMo better' if delta > 0 else 'RF better'})")
            print(f"{'─'*40}")
        else:
            print("  RF model or features not found — skipping RF comparison.")


if __name__ == "__main__":
    main()
