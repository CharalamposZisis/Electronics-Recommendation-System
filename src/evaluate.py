"""Κοινός κώδικας αξιολόγησης για όλα τα μοντέλα (ranking + ταξινόμηση E/S/C/I).

Κάθε μοντέλο γράφει ένα αρχείο predictions (parquet/csv) με στήλες:
    query_id, product_id, score            (υποχρεωτικές — μεγαλύτερο score = πιο σχετικό)
    pred_label                             (προαιρετική — E/S/C/I, για μετρικές ταξινόμησης)
Οι ετικέτες (esci_label) διαβάζονται από το split, ΟΧΙ από το αρχείο predictions.

Ορισμοί:
- nDCG@k: graded gains E=1.0, S=0.1, C=0.01, I=0.0 (Reddy et al., 2022). Το ideal DCG
  υπολογίζεται από τις ετικέτες των υποψηφίων του ίδιου query. Queries με ideal DCG = 0
  (όλα I) εξαιρούνται.
- MRR, Recall@k: σχετικό = E. Queries χωρίς κανένα E εξαιρούνται.
- Ισοβαθμίες στο score σπάνε με seed (τυχαία αλλά αναπαράξιμα).

Χρήση:
    python -m src.evaluate --predictions results/bm25_title/predictions_test.parquet --split test
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["E", "S", "C", "I"]
GAINS = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}
DEFAULT_KS = (1, 5, 10)


def load_split(split, data_dir="data/processed/esci_small_us"):
    cols = ["query_id", "product_id", "esci_label", "source"]
    return pd.read_parquet(ROOT / data_dir / f"{split}.parquet", columns=cols)


def load_predictions(path):
    path = ROOT / path
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def attach_labels(preds, gold):
    """Ενώνει predictions με gold labels· κάθε ζεύγος του split πρέπει να έχει πρόβλεψη."""
    keys = ["query_id", "product_id"]
    if preds.duplicated(keys).any():
        raise ValueError("Διπλές γραμμές (query_id, product_id) στα predictions")
    df = gold.merge(preds, on=keys, how="left", validate="one_to_one")
    missing = df["score"].isna().sum()
    if missing:
        raise ValueError(f"{missing} ζεύγη του split δεν έχουν score στα predictions")
    return df


def rank_within_query(df, seed=0):
    """Προσθέτει στήλη rank (1 = πρώτο) ανά query, με αναπαράξιμο τυχαίο tie-break."""
    rng = np.random.default_rng(seed)
    df = df.assign(_tie=rng.random(len(df)))
    df = df.sort_values(["query_id", "score", "_tie"], ascending=[True, False, True])
    df["rank"] = df.groupby("query_id").cumcount() + 1
    return df.drop(columns="_tie")


def per_query_ranking(df, ks=DEFAULT_KS):
    """Μετρικές ranking ανά query. Το df πρέπει να έχει rank και esci_label."""
    df = df.assign(
        gain=df["esci_label"].map(GAINS),
        is_e=(df["esci_label"] == "E").astype(float),
    )
    df["disc"] = 1.0 / np.log2(df["rank"] + 1)
    out = {}
    g = df.groupby("query_id", sort=True)

    # ideal ordering: gains σε φθίνουσα σειρά μέσα σε κάθε query
    ideal = df.sort_values(["query_id", "gain"], ascending=[True, False])
    ideal_rank = ideal.groupby("query_id").cumcount() + 1
    ideal = ideal.assign(disc=1.0 / np.log2(ideal_rank + 1), rank=ideal_rank)

    for k in ks:
        top = df["rank"] <= k
        dcg = (df["gain"] * df["disc"]).where(top, 0.0).groupby(df["query_id"]).sum()
        itop = ideal["rank"] <= k
        idcg = (ideal["gain"] * ideal["disc"]).where(itop, 0.0).groupby(ideal["query_id"]).sum()
        out[f"ndcg@{k}"] = (dcg / idcg).where(idcg > 0)

        n_e = g["is_e"].sum()
        hits = df["is_e"].where(top, 0.0).groupby(df["query_id"]).sum()
        out[f"recall@{k}"] = (hits / n_e).where(n_e > 0)

    first_e = df[df["is_e"] == 1].groupby("query_id")["rank"].min()
    out["mrr"] = (1.0 / first_e).reindex(g.size().index)
    return pd.DataFrame(out)


def summarize_ranking(per_query):
    summary = {}
    for col in per_query.columns:
        vals = per_query[col].dropna()
        summary[col] = round(float(vals.mean()), 4)
    summary["n_queries_ndcg"] = int(per_query["ndcg@10"].notna().sum()) if "ndcg@10" in per_query else None
    summary["n_queries_with_E"] = int(per_query["mrr"].notna().sum())
    return summary


def classification_metrics(df):
    y_true, y_pred = df["esci_label"], df["pred_label"]
    f1_per = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)), 4),
        "f1_per_class": {lab: round(float(v), 4) for lab, v in zip(LABELS, f1_per)},
        "confusion_matrix": {
            "labels": LABELS,
            "matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        },
    }


def evaluate(preds, gold, ks=DEFAULT_KS, seed=0, by_source=True):
    """Επιστρέφει dict με όλες τις μετρικές· ίδια συνάρτηση για κάθε μοντέλο."""
    df = rank_within_query(attach_labels(preds, gold), seed=seed)
    pq = per_query_ranking(df, ks=ks)
    result = {"n_rows": len(df), "n_queries": int(df["query_id"].nunique()),
              "ranking": summarize_ranking(pq)}

    if by_source and "source" in df:
        src = df.groupby("query_id")["source"].first()
        result["ranking_by_source"] = {
            s: summarize_ranking(pq.loc[src[src == s].index]) for s in sorted(src.dropna().unique())
        }

    if "pred_label" in df and df["pred_label"].notna().all():
        result["classification"] = classification_metrics(df)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--data-dir", default="data/processed/esci_small_us")
    parser.add_argument("--out", default=None, help="json εξόδου (default: δίπλα στα predictions)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    res = evaluate(load_predictions(args.predictions), load_split(args.split, args.data_dir), seed=args.seed)
    out = Path(args.out) if args.out else (ROOT / args.predictions).with_name(f"metrics_{args.split}.json")
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res["ranking"], indent=2))


if __name__ == "__main__":
    main()
