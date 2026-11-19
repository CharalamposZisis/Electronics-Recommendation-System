"""Φίλτρο electronics και EDA: ESCI product_id ↔ Amazon 2023 meta_Electronics parent_asin.

1. Coverage: πόσα products / ζεύγη / queries του ESCI (small, us) αντιστοιχίζονται
   σε προϊόν του meta_Electronics, με διάφορους υποψήφιους ορισμούς «electronics query».
2. Sanity check: σε queries που είναι προφανώς electronics (λέξεις-κλειδιά), τι
   ποσοστό των products βρίσκει το join (εκτίμηση recall του φίλτρου).
3. EDA στο υποσύνολο electronics (ορισμός: >=50% των E products του query στο meta).

Δεν αλλάζει τα splits· μόνο μετράει.

Χρήση:
    python -m src.electronics --config configs/data.yaml
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import LABELS, ROOT, load_config
from src.eda import INK, LABEL_COLORS, MUTED, style

SPLITS = ["train", "val", "test"]
ELEC_DEF = ">=50% of E products"
# Λέξεις-κλειδιά για queries που είναι αναμφίβολα electronics (μόνο για sanity check).
KEYWORDS = (r"\b(?:laptop|headphones?|earbuds|monitor|hdmi|keyboard|router|ssd|usb|"
            r"webcam|camera|speaker|tv)\b")


def load(cfg):
    proc = ROOT / cfg["output_dir"]
    esci = pd.concat(
        [pd.read_parquet(proc / f"{s}.parquet",
                         columns=["query_id", "query", "product_id", "esci_label",
                                  "product_title", "source"])
         .assign(split=s) for s in SPLITS],
        ignore_index=True,
    )
    meta = pd.read_parquet(ROOT / cfg["amazon_meta"]["out_file"])
    meta = meta.drop_duplicates("parent_asin").rename(columns={"title": "meta_title"})
    return esci, meta


def coverage(esci, meta):
    m = esci.merge(meta, left_on="product_id", right_on="parent_asin", how="left")
    m["is_elec"] = m["parent_asin"].notna()

    prods = m.drop_duplicates("product_id")
    per_q = m.groupby("query_id").agg(n=("is_elec", "size"), n_elec=("is_elec", "sum"),
                                      split=("split", "first"))
    per_q["frac"] = per_q["n_elec"] / per_q["n"]
    # E products ανά query: πόσα από τα «σωστά» αποτελέσματα είναι electronics
    e = m[m["esci_label"] == "E"].groupby("query_id")["is_elec"].mean()
    per_q["frac_E_elec"] = e.reindex(per_q.index)

    out = {
        "meta_products": len(meta),
        "esci_products": len(prods),
        "products_matched": int(prods["is_elec"].sum()),
        "products_matched_share": round(float(prods["is_elec"].mean()), 4),
        "rows_matched_share": round(float(m["is_elec"].mean()), 4),
        "matched_share_by_label": {
            lab: round(float(m.loc[m["esci_label"] == lab, "is_elec"].mean()), 4) for lab in LABELS},
        "queries": len(per_q),
        "queries_by_threshold": {},
    }
    defs = {
        ">=1 product": per_q["n_elec"] >= 1,
        ">=50% products": per_q["frac"] >= 0.5,
        ELEC_DEF: per_q["frac_E_elec"] >= 0.5,
        "all products": per_q["frac"] == 1,
    }
    for name, mask in defs.items():
        out["queries_by_threshold"][name] = {
            "total": int(mask.sum()),
            **{s: int((mask & (per_q["split"] == s)).sum()) for s in SPLITS},
        }
    per_q["is_elec_query"] = defs[ELEC_DEF]
    return m, per_q, out


def keyword_check(m):
    """Recall proxy: σε προφανώς-electronics queries, ποσοστό products που βρήκε το join."""
    kq = m[m["query"].str.lower().str.contains(KEYWORDS, regex=True)]
    miss = kq[(kq["esci_label"] == "E") & ~kq["is_elec"]]
    return {
        "keywords": KEYWORDS,
        "queries": int(kq["query_id"].nunique()),
        "matched_share_by_label": {
            lab: round(float(kq.loc[kq["esci_label"] == lab, "is_elec"].mean()), 4) for lab in LABELS},
    }, miss.sample(min(30, len(miss)), random_state=42)[["query", "product_id", "product_title"]]


def elec_eda(m, per_q):
    """Στατιστικά του υποσυνόλου electronics σε σύγκριση με όλο το small-us."""
    elec_q = per_q.index[per_q["is_elec_query"]]
    sub = m[m["query_id"].isin(elec_q)]
    qtext = m.drop_duplicates("query_id").set_index("query_id")["query"]
    words = qtext.str.split().str.len()

    def dist(s):
        return {k: round(float(v), 4) for k, v in
                s.value_counts(normalize=True).reindex(LABELS, fill_value=0).items()}

    matched = sub[sub["is_elec"]].drop_duplicates("product_id")
    out = {
        "definition": ELEC_DEF,
        "queries": len(elec_q),
        "rows": len(sub),
        "label_dist": {"all_small_us": dist(m["esci_label"]), "electronics": dist(sub["esci_label"])},
        "queries_with_C": {
            "all_small_us": round(float((m.groupby("query_id")["esci_label"]
                                         .apply(lambda s: (s == "C").any())).mean()), 4),
            "electronics": round(float((sub.groupby("query_id")["esci_label"]
                                        .apply(lambda s: (s == "C").any())).mean()), 4)},
        "candidates_per_query_median": {
            "all_small_us": float(per_q["n"].median()),
            "electronics": float(per_q.loc[elec_q, "n"].median())},
        "query_words_median": {
            "all_small_us": float(words.median()), "electronics": float(words.loc[elec_q].median())},
        "source_dist": {
            "all_small_us": m.drop_duplicates("query_id")["source"].value_counts(normalize=True).round(4).to_dict(),
            "electronics": sub.drop_duplicates("query_id")["source"].value_counts(normalize=True).round(4).to_dict()},
        "matched_products": len(matched),
        "price": {"available_share": round(float(matched["price"].notna().mean()), 4),
                  **matched["price"].describe(percentiles=[.25, .5, .75, .95]).round(2).to_dict()},
        "average_rating": matched["average_rating"].describe().round(2).to_dict(),
        "rating_number_median": float(matched["rating_number"].median()),
    }
    cats = m[m["is_elec"]].drop_duplicates("product_id")
    out["top_main_category"] = cats["main_category"].value_counts().head(15).to_dict()
    out["top_category_level2"] = cats["categories"].str.split(" > ").str[1] \
        .value_counts().head(20).to_dict()
    return sub, elec_q, words, out


# ---------------------------------------------------------------- figures

def fig_frac(per_q, out_dir):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(per_q["frac"], bins=np.linspace(0, 1, 21), color=LABEL_COLORS["E"],
            edgecolor="white", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("ποσοστό products του query που βρέθηκαν στο meta_Electronics")
    ax.set_ylabel("queries (log)")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.grid(axis="x", visible=False)
    ax.set_title("Electronics ανά query (ESCI small, us)", loc="left", color=INK)
    fig.savefig(out_dir / "electronics_fraction_per_query.png")
    plt.close(fig)


def fig_label_compare(eda_out, out_dir):
    d = pd.DataFrame(eda_out["label_dist"]).T.loc[["all_small_us", "electronics"], LABELS]
    d.index = ["όλο το small-us", "electronics"]
    fig, ax = plt.subplots(figsize=(7, 1.9))
    left = np.zeros(len(d))
    for lab in LABELS:
        ax.barh(d.index, d[lab], left=left, color=LABEL_COLORS[lab], label=lab,
                height=0.6, edgecolor="white", linewidth=2)
        for i, v in enumerate(d[lab]):
            if v > 0.06:
                ax.text(left[i] + v / 2, i, f"{v:.0%}", ha="center", va="center",
                        fontsize=8, color="white")
        left += d[lab].to_numpy()
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0))
    ax.set_title("Κατανομή ετικετών: electronics vs όλο το small-us", loc="left", pad=22)
    fig.savefig(out_dir / "labels_electronics_vs_all.png")
    plt.close(fig)


def fig_categories(eda_out, out_dir):
    s = pd.Series(eda_out["top_category_level2"]).head(12)[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(s) + 1))
    ax.barh(s.index, s.values, color=LABEL_COLORS["E"], height=0.6)
    for i, v in enumerate(s.values):
        ax.text(v, i, f" {v:,}", va="center", fontsize=8, color=MUTED)
    ax.set_xlabel("ESCI products (small, us) που βρέθηκαν στο meta")
    ax.grid(axis="y", visible=False)
    ax.set_title("Υποκατηγορίες Electronics (2ο επίπεδο)", loc="left")
    fig.savefig(out_dir / "electronics_categories.png")
    plt.close(fig)


def fig_price(sub, out_dir):
    p = sub[sub["is_elec"]].drop_duplicates("product_id")["price"].dropna()
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(p, bins=np.logspace(0, np.log10(max(p.max(), 10)), 40),
            color=LABEL_COLORS["E"], edgecolor="white", linewidth=0.5)
    ax.set_xscale("log")
    ax.axvline(p.median(), color=INK, linestyle="--", linewidth=1)
    ax.set_xlabel("τιμή ($, log)")
    ax.set_ylabel("products")
    ax.grid(axis="x", visible=False)
    ax.set_title(f"Τιμές electronics products (median ${p.median():.0f}, n={len(p):,})",
                 loc="left")
    fig.savefig(out_dir / "electronics_price.png")
    plt.close(fig)


def fig_query_length(words, elec_q, out_dir):
    fig, ax = plt.subplots(figsize=(7, 3))
    bins = np.arange(0.5, 16.5, 1)
    ax.hist(words.clip(upper=15), bins=bins, density=True, color=LABEL_COLORS["E"],
            alpha=0.85, label="όλο το small-us", edgecolor="white", linewidth=0.5)
    ax.hist(words.loc[elec_q].clip(upper=15), bins=bins, density=True, histtype="step",
            color=LABEL_COLORS["S"], linewidth=2, label="electronics")
    ax.set_xlabel("λέξεις (15 = 15+)")
    ax.set_ylabel("ποσοστό queries")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    ax.set_title("Μήκος query: electronics vs όλο το small-us", loc="left")
    fig.savefig(out_dir / "electronics_query_length.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--out", default="results/eda/electronics")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    style()

    esci, meta = load(cfg)
    m, per_q, cov = coverage(esci, meta)
    kw, kw_missed = keyword_check(m)
    sub, elec_q, words, eda_out = elec_eda(m, per_q)

    summary = {"coverage": cov, "keyword_check": kw, "electronics_subset": eda_out}
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    kw_missed.to_csv(out_dir / "keyword_queries_unmatched_E_sample.csv", index=False)
    qtext = m.drop_duplicates("query_id").set_index("query_id")["query"]
    qtext.loc[elec_q].sample(min(50, len(elec_q)), random_state=42) \
        .to_csv(out_dir / "sample_elec_queries.csv")

    fig_frac(per_q, out_dir)
    fig_label_compare(eda_out, out_dir)
    fig_categories(eda_out, out_dir)
    fig_price(sub, out_dir)
    fig_query_length(words, elec_q, out_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
