"""Fine-tune RoBERTa on LIAR-style fake-news labels.

Usage:
    python backend/ml/train.py --output-dir backend/ml/model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from datasets import DatasetDict, load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "roberta-base"
LABEL_TO_ID = {"FAKE": 0, "UNCERTAIN": 1, "REAL": 2}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
LIAR_PARQUET_FILES = {
    "train": "https://huggingface.co/datasets/ucsbnlp/liar/resolve/main/default/liar-train.parquet",
    "validation": "https://huggingface.co/datasets/ucsbnlp/liar/resolve/main/default/liar-validation.parquet",
    "test": "https://huggingface.co/datasets/ucsbnlp/liar/resolve/main/default/liar-test.parquet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune RoBERTa for fake-news detection.")
    parser.add_argument(
        "--dataset-name",
        default="ucsbnlp/liar",
        help="HuggingFace dataset id. Use ucsbnlp/liar (Parquet) on datasets>=4; legacy id 'liar' is script-based.",
    )
    parser.add_argument("--dataset-config", default=None, help="Optional dataset config.")
    parser.add_argument("--text-column", default="statement", help="Source text column.")
    parser.add_argument("--label-column", default="label", help="Source label column.")
    parser.add_argument("--max-length", type=int, default=512, help="Max token length.")
    parser.add_argument("--output-dir", default="backend/ml/model", help="Output model directory.")
    parser.add_argument("--epochs", type=float, default=2.0, help="Training epochs.")
    parser.add_argument("--train-batch-size", type=int, default=8, help="Train batch size per device.")
    parser.add_argument("--eval-batch-size", type=int, default=16, help="Eval batch size per device.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Initial learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--test-size", type=float, default=0.1, help="Validation split if no split exists.")
    return parser.parse_args()


def map_liar_label(raw_label: str) -> int:
    normalized = str(raw_label).strip().lower()
    if normalized in {"false", "pants-fire", "barely-true"}:
        return LABEL_TO_ID["FAKE"]
    if normalized in {"half-true"}:
        return LABEL_TO_ID["UNCERTAIN"]
    if normalized in {"mostly-true", "true"}:
        return LABEL_TO_ID["REAL"]
    return LABEL_TO_ID["UNCERTAIN"]


def normalize_label_column(dataset_dict: DatasetDict, label_column: str) -> DatasetDict:
    def _map_item(item: Dict[str, object]) -> Dict[str, int]:
        value = item.get(label_column)
        if isinstance(value, (int, np.integer)):
            mapped = int(value)
            # LIAR integer convention often maps:
            # 0 pants-fire, 1 false, 2 barely-true, 3 half-true, 4 mostly-true, 5 true
            if mapped in {0, 1, 2}:
                return {"labels": LABEL_TO_ID["FAKE"]}
            if mapped == 3:
                return {"labels": LABEL_TO_ID["UNCERTAIN"]}
            if mapped in {4, 5}:
                return {"labels": LABEL_TO_ID["REAL"]}
            return {"labels": LABEL_TO_ID["UNCERTAIN"]}
        return {"labels": map_liar_label(str(value))}

    updated = dataset_dict.map(_map_item)
    return updated


def infer_text_column(split, preferred_column: str) -> str:
    if preferred_column in split.column_names:
        return preferred_column
    for candidate in ("statement", "text", "content", "headline"):
        if candidate in split.column_names:
            return candidate
    raise ValueError(f"No suitable text column found in dataset columns: {split.column_names}")


def load_dataset_safe(dataset_name: str, dataset_config: str | None) -> DatasetDict:
    """Load HF dataset; fall back to Parquet LIAR if script-based loading is disabled."""
    if dataset_name in {"liar", "LIAR", "ucsbnlp/liar", "UKPLab/liar"}:
        return load_dataset("parquet", data_files=LIAR_PARQUET_FILES)

    try:
        if dataset_config:
            return load_dataset(dataset_name, dataset_config)
        return load_dataset(dataset_name)
    except RuntimeError as exc:
        msg = str(exc)
        if "Dataset scripts are no longer supported" in msg or "liar.py" in msg:
            if dataset_name in {"liar", "LIAR"} or dataset_name.endswith("/liar"):
                return load_dataset("parquet", data_files=LIAR_PARQUET_FILES)
        raise


def ensure_train_eval_splits(ds: DatasetDict, test_size: float, seed: int) -> DatasetDict:
    if "train" in ds and "validation" in ds:
        return ds
    if "train" in ds and "test" in ds:
        return DatasetDict({"train": ds["train"], "validation": ds["test"]})
    if "train" in ds:
        split = ds["train"].train_test_split(test_size=test_size, seed=seed)
        return DatasetDict({"train": split["train"], "validation": split["test"]})
    first_key = list(ds.keys())[0]
    split = ds[first_key].train_test_split(test_size=test_size, seed=seed)
    return DatasetDict({"train": split["train"], "validation": split["test"]})


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )
    macro_f1 = f1_score(labels, predictions, average="macro")
    accuracy = accuracy_score(labels, predictions)
    return {
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "f1_macro": macro_f1,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset_safe(args.dataset_name, args.dataset_config)
    dataset = ensure_train_eval_splits(dataset, test_size=args.test_size, seed=args.seed)
    dataset = normalize_label_column(dataset, label_column=args.label_column)

    text_column = infer_text_column(dataset["train"], args.text_column)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_batch(examples: Dict[str, List[object]]) -> Dict[str, object]:
        return tokenizer(
            [str(x) for x in examples[text_column]],
            truncation=True,
            max_length=args.max_length,
        )

    tokenized = dataset.map(tokenize_batch, batched=True)
    keep_cols = {"input_ids", "attention_mask", "labels"}
    remove_cols = [col for col in tokenized["train"].column_names if col not in keep_cols]
    tokenized = tokenized.remove_columns(remove_cols)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        logging_dir=str(output_dir / "logs"),
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        seed=args.seed,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    with (output_dir / "training_metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    print("Training complete. Saved model and metrics to:", output_dir)


if __name__ == "__main__":
    main()
