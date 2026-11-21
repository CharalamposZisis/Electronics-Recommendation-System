"""Lexical baselines για re-ranking των υποψηφίων κάθε query: BM25, TF-IDF, random.

Το task: για κάθε query του ESCI ταξινομούμε τα 16-40 υποψήφια products του (όχι full-corpus
retrieval). Οι στατιστικές λέξεων (IDF, avgdl) υπολογίζονται σε όλο τον κατάλογο products
του small-us (train+val+test), χωρίς χρήση ετικετών.

Χρήση:
    python -m src.baselines --config configs/bm25_title.yaml
Έξοδος: results/<name>/{predictions_<split>.parquet, metrics_<split>.json, config.yaml}
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.evaluate import evaluate

ROOT = Path(__file__).resolve().parents[1]
TOKEN_PATTERN = r"(?u)\b\w+\b"
TEXT_FIELDS = ["product_title", "product_brand", "product_color",
               "product_bullet_point", "product_description"]
HTML_TAG = re.compile(r"<[^>]+>")


def load_config(path):
    with open(ROOT / path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_splits(data_dir, names):
    cols = ["query_id", "query", "product_id", "esci_label", "source"] + TEXT_FIELDS
    return {n: pd.read_parquet(ROOT / data_dir / f"{n}.parquet", columns=cols)
            for n in names}


def product_text(df, fields):
    parts = [df[f].fillna("").astype(str) for f in fields]
    text = parts[0]
    for p in parts[1:]:
        text = text + " " + p
    return text.str.replace(HTML_TAG, " ", regex=True)


def build_catalogue(splits, fields):
    """Μοναδικά products όλων των splits, με το κείμενο που θα δει ο ranker."""
    allp = pd.concat([s[["product_id"] + TEXT_FIELDS] for s in splits.values()])
    allp = allp.drop_duplicates("product_id").reset_index(drop=True)
    return pd.Series(product_text(allp, fields).values, index=allp["product_id"])


class BM25:
    def __init__(self, k1=1.2, b=0.75):
        self.k1, self.b = k1, b

    def fit(self, docs):
        self.vec = CountVectorizer(lowercase=True, token_pattern=TOKEN_PATTERN)
        tf = self.vec.fit_transform(docs).tocsr().astype(np.float64)
        n_docs = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        dl = np.asarray(tf.sum(axis=1)).ravel()
        norm = self.k1 * (1 - self.b + self.b * dl / dl.mean())
        # βάρος BM25 κάθε (doc, term): idf * tf*(k1+1) / (tf + k1*norm_doc)
        rows = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
        tf.data = idf[tf.indices] * tf.data * (self.k1 + 1) / (tf.data + norm[rows])
        self.doc_matrix = tf
        return self

    def query_matrix(self, queries):
        q = self.vec.transform(queries)
        q.data[:] = 1.0  # κάθε όρος του query μετράει μία φορά
        return q


class TfIdf:
    def fit(self, docs):
        self.vec = TfidfVectorizer(lowercase=True, token_pattern=TOKEN_PATTERN,
                                   sublinear_tf=True)
        self.doc_matrix = self.vec.fit_transform(docs).tocsr()
        return self

    def query_matrix(self, queries):
        return self.vec.transform(queries)  # l2-normalised → γινόμενο = cosine


def pair_scores(model, catalogue_index, df):
    """score(query, product) = εσωτερικό γινόμενο query-vector και doc-vector, ανά γραμμή."""
    doc_pos = catalogue_index.get_indexer(df["product_id"])
    assert (doc_pos >= 0).all()
    uq, q_pos = np.unique(df["query"].values, return_inverse=True)
    Q = model.query_matrix(uq)
    prod = Q[q_pos].multiply(model.doc_matrix[doc_pos])
    return np.asarray(sparse.csr_matrix(prod).sum(axis=1)).ravel()


def run(cfg):
    method = cfg["method"]
    split_names = cfg["eval_splits"]
    splits = load_splits(cfg["data_dir"], ["train", "val", "test"])
    out_dir = ROOT / "results" / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    if method in ("bm25", "tfidf"):
        catalogue = build_catalogue(splits, cfg["fields"])
        model = BM25(**cfg.get("bm25", {})) if method == "bm25" else TfIdf()
        model.fit(catalogue.values)
        print(f"{method}: {len(catalogue)} products, vocab {len(model.vec.vocabulary_)}")

    all_metrics = {}
    for name in split_names:
        df = splits[name]
        if method == "random":
            score = np.random.default_rng(cfg["seed"]).random(len(df))
        else:
            score = pair_scores(model, catalogue.index, df)
        preds = df[["query_id", "product_id"]].assign(score=score)
        preds.to_parquet(out_dir / f"predictions_{name}.parquet", index=False)

        res = evaluate(preds, df[["query_id", "product_id", "esci_label", "source"]],
                       seed=cfg["seed"])
        res["experiment"] = cfg["name"]
        (out_dir / f"metrics_{name}.json").write_text(
            json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        all_metrics[name] = res["ranking"]
        print(name, json.dumps(res["ranking"]))

    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return all_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
