"""Συγκεντρώνει τα results/<exp>/metrics_<split>.json σε έναν πίνακα.

Χρήση:
    python -m src.report --split test
Έξοδος: results/summary_<split>.csv (και markdown στην οθόνη).
"""
import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLS = ["ndcg@10", "ndcg@5", "mrr", "recall@5", "recall@10", "accuracy", "macro_f1"]


def collect(split):
    rows = []
    for path in sorted((ROOT / "results").glob(f"*/metrics_{split}.json")):
        res = json.loads(path.read_text(encoding="utf-8"))
        row = {"experiment": path.parent.name, **res["ranking"]}
        if "classification" in res:
            row["accuracy"] = res["classification"]["accuracy"]
            row["macro_f1"] = res["classification"]["macro_f1"]
        rows.append(row)
    df = pd.DataFrame(rows).set_index("experiment")
    return df.reindex(columns=[c for c in COLS if c in df]).sort_values("ndcg@10", ascending=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    df = collect(args.split)
    df.to_csv(ROOT / "results" / f"summary_{args.split}.csv")
    print(df.to_markdown(floatfmt=".4f") if _has_tabulate() else df.to_string())


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
