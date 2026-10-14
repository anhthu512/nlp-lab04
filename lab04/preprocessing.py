"""Stateless text processing and reproducible IMDb acquisition for W4."""
from __future__ import annotations

import hashlib
import html
import re
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_REVISION = "e6281661ce1c48d982bc483cf8a173c1bbeb5d31"
DATASET_URL = "https://huggingface.co/datasets/stanfordnlp/imdb"
FILES = {
    "train": ("train-00000-of-00001.parquet", "db47d16b5c297cc0dd625e519c81319c24c9149e70e8496de5475f6fa928342c"),
    "test": ("test-00000-of-00001.parquet", "b52e26e2f872d282ffac460bf9770b25ac6f102cda0e6ca7158df98c94e8b3da"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_imdb(data_dir: str | Path) -> pd.DataFrame:
    """Load labeled official splits; download pinned parquet files if needed."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for split, (filename, checksum) in FILES.items():
        path = data_dir / filename
        if not path.exists():
            url = f"{DATASET_URL}/resolve/{DATASET_REVISION}/plain_text/{filename}"
            temporary = path.with_suffix(".part")
            request = urllib.request.Request(url, headers={"User-Agent": "NLP-LAB04/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
            if sha256_file(temporary) != checksum:
                raise ValueError(f"Checksum mismatch for {filename}")
            temporary.replace(path)
        if sha256_file(path) != checksum:
            raise ValueError(f"Checksum mismatch for cached {filename}")
        frame = pd.read_parquet(path)
        frame["official_split"] = split
        frame["sample_id"] = [f"{split}:{i}" for i in range(len(frame))]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def normalize_text(text: str) -> str:
    """Remove HTML and normalize whitespace/case without learning parameters."""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\u2019", "'").lower()
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    """Shared tokenizer: retain negation, contractions and single characters."""
    return re.findall(r"[a-z]+(?:'[a-z]+)?|\d+", normalize_text(text))


def document_key(text: str) -> str:
    return hashlib.sha256(" ".join(tokenize(text)).encode("utf-8")).hexdigest()


def inspect_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return mechanical audit numbers; no interpretation of the dataset."""
    text = frame["text"].fillna("").astype(str)
    keys = text.map(document_key)
    summary = {
        "samples": len(frame),
        "classes": frame["label"].nunique(dropna=True),
        "missing_text": int(frame["text"].isna().sum()),
        "blank_text": int(text.str.strip().eq("").sum()),
        "missing_labels": int(frame["label"].isna().sum()),
        "invalid_labels": int((frame["label"].notna() & ~frame["label"].isin([0, 1])).sum()),
        "raw_duplicate_extra_rows": int(text.duplicated().sum()),
        "token_duplicate_extra_rows": int(keys.duplicated().sum()),
        "token_duplicate_groups": int(keys[keys.duplicated(keep=False)].nunique()),
    }
    counts = frame.groupby(["official_split", "label"], dropna=False).size().rename("count").reset_index()
    counts["class"] = counts["label"].map({0: "negative", 1: "positive"})
    counts["percentage"] = counts["count"] / counts.groupby("official_split")["count"].transform("sum") * 100
    lengths = pd.DataFrame({"sample_id": frame["sample_id"], "official_split": frame["official_split"],
                            "characters": text.str.len(), "tokens": text.map(lambda s: len(tokenize(s)))})
    return pd.DataFrame([summary]), counts, lengths


def clean_official_splits(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deduplicate before splitting; keep training data independent of test labels.

    Within each official split, remove conflicting-label duplicate groups, then
    retain the first row per shared-tokenizer key. Remove cross-split duplicates
    from TEST only, without inspecting their labels to modify training data.
    """
    cleaned = {}
    audits = []
    for name in ("train", "test"):
        part = frame.loc[frame.official_split.eq(name)].copy()
        original = len(part)
        valid = part.text.notna() & part.label.isin([0, 1])
        part = part.loc[valid].copy()
        part["clean_text"] = part.text.map(normalize_text)
        part["tokens"] = part.text.map(tokenize)
        part = part.loc[part.tokens.map(len).gt(0)].copy()
        part["document_key"] = part.text.map(document_key)
        valid_count = len(part)
        conflicts = part.groupby("document_key").label.nunique()
        conflicting_keys = set(conflicts[conflicts.gt(1)].index)
        part = part.loc[~part.document_key.isin(conflicting_keys)].copy()
        conflict_rows = valid_count - len(part)
        before_dedup = len(part)
        part = part.drop_duplicates("document_key", keep="first").copy()
        audits.append({"official_split": name, "original": original,
                       "invalid_or_empty_removed": original - valid_count,
                       "conflicting_label_groups": len(conflicting_keys),
                       "conflicting_label_rows_removed": conflict_rows,
                       "duplicate_extra_rows_removed": before_dedup - len(part),
                       "cross_split_rows_removed": 0})
        cleaned[name] = part
    train_keys = set(cleaned["train"].document_key)
    overlap = cleaned["test"].document_key.isin(train_keys)
    audits[1]["cross_split_rows_removed"] = int(overlap.sum())
    cleaned["test"] = cleaned["test"].loc[~overlap].copy()
    for item in audits:
        item["retained"] = len(cleaned[item["official_split"]])
    return pd.concat(cleaned.values(), ignore_index=True), pd.DataFrame(audits)


def average_embeddings(token_lists, keyed_vectors) -> tuple[np.ndarray, pd.DataFrame]:
    """Average in-vocabulary vectors, with zero fallback and coverage audit."""
    vectors = np.zeros((len(token_lists), keyed_vectors.vector_size), dtype=np.float32)
    records = []
    for index, tokens in enumerate(token_lists):
        known = [word for word in tokens if word in keyed_vectors]
        if known:
            vectors[index] = np.mean(keyed_vectors[known], axis=0)
        records.append({"total_tokens": len(tokens), "known_tokens": len(known),
                        "oov_tokens": len(tokens) - len(known),
                        "coverage": len(known) / len(tokens) if tokens else 0.0})
    return vectors, pd.DataFrame(records)
