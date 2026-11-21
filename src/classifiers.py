"""Baselines ταξινόμησης E/S/C/I: majority class και Logistic Regression σε lexical features.

Features ανά ζεύγος (query, product):
- TF-IDF του query (q), του τίτλου (t) και των κοινών όρων (q ⊙ t, elementwise)
- αριθμητικά: BM25(query, title), cosine(q, t), ποσοστό όρων του query στον τίτλο,
  μήκος query/τίτλου (standardised)
Ο vectorizer και ο scaler γίνονται fit ΜΟΝΟ στο train. Το C επιλέγεται με macro-F1 στο val.
Ranking score = expected gain = P(E) + 0.1 P(S) + 0.01 P(C) (ίδια gains με το nDCG).

Χρήση:
    python -m src.classifiers --config configs/logreg_lexical.yaml
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.baselines import BM25, ROOT, TOKEN_PATTERN, build_catalogue, load_config, load_splits, pair_scores
from src.evaluate import GAINS, LABELS, evaluate

GOLD_COLS = ["query_id", "product_id", "esci_label", "source"]


class LexicalFeatures:
    def __init__(self, min_df=2, bm25=None):
        self.min_df, self.bm25_cfg = min_df, bm25 or {}

    def fit(self, train, catalogue):
        self.vec = TfidfVectorizer(lowercase=True, token_pattern=TOKEN_PATTERN,
                                   sublinear_tf=True, min_df=self.min_df)
        self.vec.fit(pd.concat([train["query"].drop_duplicates(),
                                train["product_title"].fillna("").drop_duplicates()]))
        self.bm25 = BM25(**self.bm25_cfg).fit(catalogue.values)  # unsupervised, χωρίς ετικέτες
        self.catalogue_index = catalogue.index
        self.scaler = StandardScaler().fit(self._dense(train))
        return self

    def _dense(self, df):
        q_tok = df["query"].str.lower().str.findall(TOKEN_PATTERN)
        t_tok = df["product_title"].fillna("").str.lower().str.findall(TOKEN_PATTERN)
        overlap = [len(set(q) & set(t)) / max(len(set(q)), 1) for q, t in zip(q_tok, t_tok)]
        bm25 = pair_scores(self.bm25, self.catalogue_index, df)
        return np.column_stack([bm25, overlap, q_tok.str.len(), t_tok.str.len()])

    def transform(self, df):
        q = self.vec.transform(df["query"])
        t = self.vec.transform(df["product_title"].fillna(""))
        qt = q.multiply(t).tocsr()
        cos = np.asarray(qt.sum(axis=1))
        dense = np.hstack([self.scaler.transform(self._dense(df)), cos])
        return sparse.hstack([q, t, qt, sparse.csr_matrix(dense)]).tocsr()


def macro_f1_of(preds, gold, seed):
    return evaluate(preds, gold, seed=seed, by_source=False)["classification"]["macro_f1"]


def to_predictions(df, proba, classes):
    p = pd.DataFrame(proba, columns=classes)
    score = sum(GAINS[c] * p[c] for c in classes)
    return df[["query_id", "product_id"]].assign(
        score=score.values, pred_label=np.asarray(classes)[proba.argmax(axis=1)])


def run(cfg):
    seed = cfg["seed"]
    splits = load_splits(cfg["data_dir"], ["train", "val", "test"])
    out_dir = ROOT / "results" / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {"experiment": cfg["name"]}

    if cfg["method"] == "majority":
        majority = splits["train"]["esci_label"].value_counts().idxmax()
        predict = lambda df: df[["query_id", "product_id"]].assign(score=0.0, pred_label=majority)
        info["majority_label"] = majority
    elif cfg["method"] == "logreg":
        t0 = time.time()
        feats = LexicalFeatures(**cfg.get("features", {})).fit(
            splits["train"], build_catalogue(splits, ["product_title"]))
        X_tr = feats.transform(splits["train"])
        X_val = feats.transform(splits["val"])
        y_tr = splits["train"]["esci_label"].values
        print(f"features: {X_tr.shape} ({time.time() - t0:.0f}s)")

        val_scores, models = {}, {}
        for C in cfg["model"]["C_grid"]:
            t0 = time.time()
            m = LogisticRegression(C=C, max_iter=cfg["model"]["max_iter"],
                                   class_weight=cfg["model"].get("class_weight"),
                                   random_state=seed)
            m.fit(X_tr, y_tr)
            pv = to_predictions(splits["val"], m.predict_proba(X_val), list(m.classes_))
            val_scores[C] = macro_f1_of(pv, splits["val"][GOLD_COLS], seed)
            models[C] = m
            print(f"C={C}: val macro-F1 {val_scores[C]:.4f} ({time.time() - t0:.0f}s)")
        best_C = max(val_scores, key=val_scores.get)
        model = models[best_C]
        info.update(val_macro_f1_by_C={str(k): v for k, v in val_scores.items()}, best_C=best_C)
        predict = lambda df: to_predictions(df, model.predict_proba(feats.transform(df)),
                                            list(model.classes_))
    else:
        raise ValueError(cfg["method"])

    for name in cfg["eval_splits"]:
        df = splits[name]
        preds = predict(df)
        preds.to_parquet(out_dir / f"predictions_{name}.parquet", index=False)
        res = {**info, **evaluate(preds, df[GOLD_COLS], seed=seed)}
        (out_dir / f"metrics_{name}.json").write_text(
            json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        c = res["classification"]
        print(name, f"acc {c['accuracy']} macro-F1 {c['macro_f1']} F1 {c['f1_per_class']}",
              f"nDCG@10 {res['ranking']['ndcg@10']}")

    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
