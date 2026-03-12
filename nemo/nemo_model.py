import os
import sys
import json
import math
from typing import List, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import F1Score, Accuracy, ConfusionMatrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import LABEL_NAMES   # {0:"REP", 1:"INS", 2:"DEL", 3:"PAU", 4:"SUB"}

NUM_CLASSES = len(LABEL_NAMES)
LABEL_LIST  = [LABEL_NAMES[i] for i in range(NUM_CLASSES)]


class TextFeatureExtractor(nn.Module):
    OUT_DIM = 32

    def forward(
        self,
        texts:            List[str],
        avg_logprobs:     torch.Tensor,   # (B,)
        no_speech_probs:  torch.Tensor,   # (B,)
        comp_ratios:      torch.Tensor,   # (B,)
        word_conf_mean:   torch.Tensor,   # (B,)
        word_conf_std:    torch.Tensor,   # (B,)
        word_conf_min:    torch.Tensor,   # (B,)
        word_conf_max:    torch.Tensor,   # (B,)
        word_conf_low:    torch.Tensor,   # (B,)  fraction < 0.8
        durations:        torch.Tensor,   # (B,)

    ) -> torch.Tensor:
        batch_feats = []
        for i, text in enumerate(texts):
            words     = text.lower().split()
            n_words   = len(words)
            avg_wlen  = sum(len(w) for w in words) / max(n_words, 1)
            ttr       = len(set(words)) / max(n_words, 1)
            rep_rate  = sum(1 for j in range(1, len(words)) if words[j] == words[j-1]) / max(n_words, 1)
            pau_proxy = float(n_words <= 2)

            #char bigram hash (16 buckets)
            bigrams = [text[k:k+2] for k in range(len(text) - 1)]
            hash_vec = [0.0] * 16
            for bg in bigrams:
                hash_vec[hash(bg) % 16] += 1.0
            total = max(sum(hash_vec), 1.0)
            hash_vec = [v / total for v in hash_vec]

            lexical = [
                n_words / 20.0, # normalized word count
                avg_wlen / 10.0,
                ttr,
                rep_rate,
                pau_proxy,
            ]
            acoustic = [
                float(avg_logprobs[i]),
                float(no_speech_probs[i]),
                float(comp_ratios[i]),
            ]
            wconf = [
                float(word_conf_mean[i]),
                float(word_conf_std[i]),
                float(word_conf_min[i]),
                float(word_conf_max[i]),
                float(word_conf_low[i]),
            ]
            spk_rate = [
                float(durations[i]) / 10.0,
                n_words / max(float(durations[i]), 0.1) / 10.0,
                math.log1p(float(durations[i])) / 5.0,
            ]

            feat = hash_vec + lexical + acoustic + wconf + spk_rate #16+5+3+5+3 = 32
            batch_feats.append(feat)

        return torch.tensor(batch_feats, dtype=torch.float32)

class SEDHead(nn.Module):

    def __init__(self, feat_in: int, num_classes: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ManifestDataset(torch.utils.data.Dataset):

    def __init__(self, manifest_path: str):
        self.records = []
        with open(manifest_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        wc = r.get("word_confidences", [1.0])
        wc_t = torch.tensor(wc, dtype=torch.float32)
        return {
            "text":           r["text"],
            "label":          torch.tensor(r["dysfluency_label"], dtype=torch.long),
            "duration":       torch.tensor(r.get("duration", 1.0), dtype=torch.float32),
            "avg_logprob":    torch.tensor(r.get("avg_logprob", -0.3), dtype=torch.float32),
            "no_speech_prob": torch.tensor(r.get("no_speech_prob", 0.0), dtype=torch.float32),
            "comp_ratio":     torch.tensor(r.get("compression_ratio", 1.0), dtype=torch.float32),
            "wc_mean":        wc_t.mean(),
            "wc_std":         wc_t.std() if len(wc_t) > 1 else torch.tensor(0.0),
            "wc_min":         wc_t.min(),
            "wc_max":         wc_t.max(),
            "wc_low":         (wc_t < 0.8).float().mean(),
        }


def collate_fn(batch):
    texts = [b["text"] for b in batch]
    return {
        "texts":          texts,
        "labels":         torch.stack([b["label"]          for b in batch]),
        "durations":      torch.stack([b["duration"]        for b in batch]),
        "avg_logprobs":   torch.stack([b["avg_logprob"]     for b in batch]),
        "no_speech_probs":torch.stack([b["no_speech_prob"]  for b in batch]),
        "comp_ratios":    torch.stack([b["comp_ratio"]       for b in batch]),
        "wc_mean":        torch.stack([b["wc_mean"]          for b in batch]),
        "wc_std":         torch.stack([b["wc_std"]           for b in batch]),
        "wc_min":         torch.stack([b["wc_min"]           for b in batch]),
        "wc_max":         torch.stack([b["wc_max"]           for b in batch]),
        "wc_low":         torch.stack([b["wc_low"]           for b in batch]),
    }

class NeMoSEDTextModel(pl.LightningModule):
    def __init__(
        self,
        hidden_dim:  int   = 128,
        dropout:     float = 0.3,
        lr:          float = 1e-3,
        weight_decay:float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.feature_extractor = TextFeatureExtractor()
        self.sed_head = SEDHead(
            feat_in=TextFeatureExtractor.OUT_DIM,
            num_classes=NUM_CLASSES,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.loss_fn = nn.CrossEntropyLoss()

        #metrics
        kw = dict(task="multiclass", num_classes=NUM_CLASSES)
        self.val_f1   = F1Score(**kw, average="macro")
        self.val_acc  = Accuracy(**kw)
        self.test_f1  = F1Score(**kw, average="macro")
        self.test_acc = Accuracy(**kw)
        self.test_cm  = ConfusionMatrix(**kw)

    def setup_training_data(self, manifest_path: str, batch_size: int = 32):
        ds = ManifestDataset(manifest_path)
        self._train_dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=True,
            num_workers=2, collate_fn=collate_fn
        )

    def setup_validation_data(self, manifest_path: str, batch_size: int = 32):
        ds = ManifestDataset(manifest_path)
        self._val_dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=2, collate_fn=collate_fn
        )

    def setup_test_data(self, manifest_path: str, batch_size: int = 32):
        ds = ManifestDataset(manifest_path)
        self._test_dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=2, collate_fn=collate_fn
        )

    def train_dataloader(self): return self._train_dl
    def val_dataloader(self):   return self._val_dl
    def test_dataloader(self):  return self._test_dl

    def forward(self, batch: dict) -> torch.Tensor:
        feats = self.feature_extractor(
            texts           = batch["texts"],
            avg_logprobs    = batch["avg_logprobs"],
            no_speech_probs = batch["no_speech_probs"],
            comp_ratios     = batch["comp_ratios"],
            word_conf_mean  = batch["wc_mean"],
            word_conf_std   = batch["wc_std"],
            word_conf_min   = batch["wc_min"],
            word_conf_max   = batch["wc_max"],
            word_conf_low   = batch["wc_low"],
            durations       = batch["durations"],
        ).to(self.device)
        return self.sed_head(feats)

    def training_step(self, batch, _):
        logits = self(batch)
        loss   = self.loss_fn(logits, batch["labels"].to(self.device))
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        logits = self(batch)
        labels = batch["labels"].to(self.device)
        loss   = self.loss_fn(logits, labels)
        preds  = logits.argmax(dim=-1)
        self.val_f1(preds, labels)
        self.val_acc(preds, labels)
        self.log("val_loss", loss, prog_bar=True)

    def on_validation_epoch_end(self):
        f1  = self.val_f1.compute()
        acc = self.val_acc.compute()
        self.log("val_f1",  f1,  prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.val_f1.reset(); self.val_acc.reset()

    def test_step(self, batch, _):
        logits = self(batch)
        labels = batch["labels"].to(self.device)
        preds  = logits.argmax(dim=-1)
        self.test_f1(preds, labels)
        self.test_acc(preds, labels)
        self.test_cm(preds, labels)

    def on_test_epoch_end(self):
        self.log("test_f1",  self.test_f1.compute())
        self.log("test_acc", self.test_acc.compute())
        self.test_f1.reset(); self.test_acc.reset(); self.test_cm.reset()

    #optimizer
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50, eta_min=1e-6)
        return [opt], [{"scheduler": sched, "interval": "epoch"}]

    # inference
    @torch.no_grad()
    def predict_from_manifest_record(self, record: dict) -> dict:
        """Pass a single manifest record dict and get a prediction back."""
        self.eval()
        wc   = record.get("word_confidences", [1.0])
        wc_t = torch.tensor(wc, dtype=torch.float32)
        batch = {
            "texts":          [record["text"]],
            "durations":      torch.tensor([record.get("duration", 1.0)]),
            "avg_logprobs":   torch.tensor([record.get("avg_logprob", -0.3)]),
            "no_speech_probs":torch.tensor([record.get("no_speech_prob", 0.0)]),
            "comp_ratios":    torch.tensor([record.get("compression_ratio", 1.0)]),
            "wc_mean":        wc_t.mean().unsqueeze(0),
            "wc_std":         (wc_t.std() if len(wc_t) > 1 else torch.tensor(0.0)).unsqueeze(0),
            "wc_min":         wc_t.min().unsqueeze(0),
            "wc_max":         wc_t.max().unsqueeze(0),
            "wc_low":         (wc_t < 0.8).float().mean().unsqueeze(0),
        }
        logits = self(batch)
        probs  = F.softmax(logits, dim=-1)[0]
        pred   = probs.argmax().item()
        return {
            "dysfluency_label": pred,
            "dysfluency_name":  LABEL_NAMES[pred],
            "confidence":       round(probs[pred].item(), 4),
            "all_probs":        {LABEL_NAMES[i]: round(probs[i].item(), 4) for i in range(NUM_CLASSES)},
        }
