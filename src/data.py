"""Φόρτωση ESCI, merge examples-products και train/val/test splits.

Χρήση:
    python -m src.data --config configs/data.yaml
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["product_id", "product_locale"]
LABELS = ["E", "S", "C", "I"]


def load_config(path):
    with open(ROOT / path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_esci(cfg):
    """Examples (φιλτραρισμένα σε version/locale) με left join στα products."""
    raw = ROOT / cfg["esci"]["raw_dir"]
    version, locale = cfg["filters"]["version"], cfg["filters"]["locale"]

    examples = pd.read_parquet(
        raw / cfg["esci"]["examples_file"],
        filters=[(version, "==", 1), ("product_locale", "==", locale)],
    )
    examples = examples.drop(columns=["__index_level_0__"], errors="ignore")

    products = pd.read_parquet(
        raw / cfg["esci"]["products_file"],
        filters=[("product_locale", "==", locale)],
    )
    products = products[products["product_id"].isin(examples["product_id"])]

    df = examples.merge(products, on=KEYS, how="left", validate="many_to_one")
    assert len(df) == len(examples)

    sources = pd.read_csv(raw / cfg["esci"]["sources_file"])
    df = df.merge(sources, on="query_id", how="left", validate="many_to_one")
    return df


def split_train_val(df, val_fraction, seed):
    """Χωρίζει το official train σε train/val ανά query_id (κανένα query και στα δύο)."""
    query_ids = np.sort(df["query_id"].unique())
    rng = np.random.default_rng(seed)
    n_val = int(round(len(query_ids) * val_fraction))
    val_ids = set(rng.choice(query_ids, size=n_val, replace=False))
    is_val = df["query_id"].isin(val_ids)
    return df[~is_val], df[is_val]


def make_splits(df, cfg):
    official_train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    train, val = split_train_val(
        official_train, cfg["split"]["val_fraction"], cfg["split"]["seed"]
    )
    splits = {"train": train, "val": val, "test": test}

    ids = {name: set(s["query_id"]) for name, s in splits.items()}
    assert not ids["train"] & ids["val"]
    assert not ids["train"] & ids["test"]
    assert not ids["val"] & ids["test"]
    return {name: s.reset_index(drop=True) for name, s in splits.items()}


def summarize(splits):
    summary = {}
    for name, s in splits.items():
        dist = s["esci_label"].value_counts(normalize=True).reindex(LABELS, fill_value=0)
        summary[name] = {
            "rows": len(s),
            "queries": int(s["query_id"].nunique()),
            "products": int(s["product_id"].nunique()),
            "label_dist": {k: round(float(v), 4) for k, v in dist.items()},
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    splits = make_splits(load_esci(cfg), cfg)

    out = ROOT / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    for name, s in splits.items():
        s.to_parquet(out / f"{name}.parquet", index=False)

    summary = {"config": cfg, "splits": summarize(splits)}
    with open(out / "split_info.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary["splits"], indent=2))


if __name__ == "__main__":
    main()
