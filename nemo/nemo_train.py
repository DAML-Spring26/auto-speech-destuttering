import os
import sys
import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nemo.nemo_model import NeMoSEDTextModel

MANIFEST_DIR = os.path.join(os.path.dirname(__file__), "manifests")
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), "outputs")


def main():
    parser = argparse.ArgumentParser(description="Train NeMo SED model")
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--hidden_dim", type=int,   default=128)
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--fast_dev",   action="store_true",
                        help="Run 1 train + 1 val batch only (smoke test)")
    args = parser.parse_args()

    #different paths
    train_manifest = os.path.join(MANIFEST_DIR, "train_manifest.json")
    val_manifest   = os.path.join(MANIFEST_DIR, "val_manifest.json")
    test_manifest  = os.path.join(MANIFEST_DIR, "test_manifest.json")

    for p in [train_manifest, val_manifest]:
        if not os.path.exists(p):
            print(f"ERROR: manifest not found: {p}")
            print("Run  python nemo/build_nemo_manifest.py  first.")
            sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ckpt_dir = os.path.join(OUTPUT_DIR, "checkpoints")

    #model
    model = NeMoSEDTextModel(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        lr=args.lr,
    )
    model.setup_training_data(train_manifest,  batch_size=args.batch_size)
    model.setup_validation_data(val_manifest,  batch_size=args.batch_size)

    print(f"\nNeMo SED Text Model")
    print(f"  Train manifest : {train_manifest}")
    print(f"  Val manifest   : {val_manifest}")
    print(f"  Hidden dim     : {args.hidden_dim}")
    print(f"  LR             : {args.lr}")
    print(f"  Epochs         : {args.epochs}\n")

    #train
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="nemo_sed_{epoch:02d}-{val_f1:.4f}",
            monitor="val_f1",
            mode="max",
            save_top_k=3,
            save_last=True,
            verbose=True,
        ),
        EarlyStopping(monitor="val_f1", patience=10, mode="max", verbose=True),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=16 if torch.cuda.is_available() else 32,
        gradient_clip_val=1.0,
        log_every_n_steps=5,
        callbacks=callbacks,
        logger=TensorBoardLogger(OUTPUT_DIR, name="nemo_sed"),
        fast_dev_run=args.fast_dev,
        enable_progress_bar=True,
    )

    trainer.fit(model)

    #test
    if os.path.exists(test_manifest) and not args.fast_dev:
        print("\nRunning test evaluation...")
        model.setup_test_data(test_manifest, batch_size=args.batch_size)
        trainer.test(model)

    print(f"\nTraining complete.")
    print(f"  Best checkpoint : {callbacks[0].best_model_path}")
    print(f"  TensorBoard     : tensorboard --logdir {OUTPUT_DIR}/nemo_sed")


if __name__ == "__main__":
    main()
