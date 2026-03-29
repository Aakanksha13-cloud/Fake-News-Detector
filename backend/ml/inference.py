"""Inference utilities for fake-news detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = ["FAKE", "UNCERTAIN", "REAL"]


@dataclass
class InferenceResult:
    label: str
    score: float
    sentence_scores: List[Dict[str, object]]


class FakeNewsInferenceService:
    def __init__(self, model_dir: str = "backend/ml/model", device: str | None = None) -> None:
        model_path = Path(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self._nlp = None

    def _get_sentence_splitter(self):
        if self._nlp is not None:
            return self._nlp

        import spacy

        self._nlp = spacy.blank("en")
        self._nlp.add_pipe("sentencizer")
        return self._nlp

    def _predict_text(self, text: str) -> Dict[str, object]:
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        pred_idx = int(probs.argmax())
        return {"label": LABELS[pred_idx], "score": float(probs[pred_idx]), "probs": probs.tolist()}

    def predict(self, text: str) -> InferenceResult:
        overall = self._predict_text(text)
        nlp = self._get_sentence_splitter()
        doc = nlp(text)

        sentence_scores: List[Dict[str, object]] = []
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue
            sent_pred = self._predict_text(sent_text)
            sentence_scores.append(
                {
                    "text": sent_text,
                    "score": sent_pred["score"],
                    "label": sent_pred["label"],
                }
            )

        return InferenceResult(
            label=overall["label"],
            score=overall["score"],
            sentence_scores=sentence_scores,
        )
