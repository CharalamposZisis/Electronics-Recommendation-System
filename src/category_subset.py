"""Υποσύνολο του ESCI ανά κατηγορία προϊόντος και EDA σε αυτό.

Το ESCI δεν έχει κατηγορία. Ένα ESCI product ανήκει στην κατηγορία X αν το
product_id (ASIN) του υπάρχει ως parent_asin στο meta_<X> του Amazon Reviews 2023
(κατεβαίνει με src/meta.py).

1. Coverage: πόσα products / ζεύγη / queries του ESCI (small, us) αντιστοιχίζονται,
   με διάφορους υποψήφιους ορισμούς «query της κατηγορίας».
2. Sanity check (όπου υπάρχουν λέξεις-κλειδιά για την κατηγορία): σε queries που
   προφανώς ανήκουν στην κατηγορία, ποσοστό products που βρίσκει το join (≈ recall).
3. EDA στο υποσύνολο (ορισμός: >=50% των E products του query στην κατηγορία),
   σε σύγκριση με όλο το small-us.

Δεν αλλάζει τα splits· μόνο μετράει.

Χρήση:
    python -m src.category_subset --category Electronics
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
from src.meta import meta_path

SPLITS = ["train", "val", "test"]
SUBSET_DEF = ">=50% of E products"
ALL = "all_small_us"
# Λέξεις-κλειδιά για queries που αναμφίβολα ανήκουν στην κατηγορία (μόνο για sanity check).
KEYWORDS = {
    "Electronics": (r"\b(?:laptop|headphones?|earbuds|monitor|hdmi|keyboard|router|ssd|usb|"
                    r"webcam|camera|speaker|tv)\b"),
}


def slug(category):
    return category.lower()


def load(cfg, category):
    proc = ROOT / cfg["output_dir"]
    esci = pd.concat(
        [pd.read_parquet(proc / f"{s}.parquet",
                         columns=["query_id", "query", "product_id", "esci_label",
                                  "product_title", "source"])
         .assign(split=s) for s in SPLITS],
        ignore_index=True,
    )
    meta = pd.read_parquet(meta_path(cfg["amazon_meta"], category))
    meta = meta.drop_duplicates("parent_asin").rename(columns={"title": "meta_title"})
    return esci, meta


def coverage(esci, meta):
    m = esci.merge(meta, left_on="product_id", right_on="parent_asin", how="left")
    m["in_cat"] = m["parent_asin"].notna()

    prods = m.drop_duplicates("product_id")
    per_q = m.groupby("query_id").agg(n=("in_cat", "size"), n_in_cat=("in_cat", "sum"),
                                      split=("split", "first"))
    per_q["frac"] = per_q["n_in_cat"] / per_q["n"]
    # E products ανά query: πόσα από τα «σωστά» αποτελέσματα ανήκουν στην κατηγορία
    e = m[m["esci_label"] == "E"].groupby("query_id")["in_cat"].mean()
    per_q["frac_E_in_cat"] = e.reindex(per_q.index)

    out = {
        "meta_products": len(meta),
        "esci_products": len(prods),
        "products_matched": int(prods["in_cat"].sum()),
        "products_matched_share": round(float(prods["in_cat"].mean()), 4),
        "rows_matched_share": round(float(m["in_cat"].mean()), 4),
        "matched_share_by_label": {
            lab: round(float(m.loc[m["esci_label"] == lab, "in_cat"].mean()), 4) for lab in LABELS},
        "queries": len(per_q),
        "queries_by_threshold": {},
    }
    defs = {
        ">=1 product": per_q["n_in_cat"] >= 1,
        ">=50% products": per_q["frac"] >= 0.5,
        SUBSET_DEF: per_q["frac_E_in_cat"] >= 0.5,
        "all products": per_q["frac"] == 1,
    }
    for name, mask in defs.items():
        out["queries_by_threshold"][name] = {
            "total": int(mask.sum()),
            **{s: int((mask & (per_q["split"] == s)).sum()) for s in SPLITS},
        }
    per_q["in_subset"] = defs[SUBSET_DEF]
    return m, per_q, out


def keyword_check(m, category):
    """Recall proxy: σε προφανή queries της κατηγορίας, ποσοστό products που βρήκε το join."""
    if category not in KEYWORDS:
        return None, None
    kq = m[m["query"].str.lower().str.contains(KEYWORDS[category], regex=True)]
    miss = kq[(kq["esci_label"] == "E") & ~kq["in_cat"]]
    return {
        "keywords": KEYWORDS[category],
        "queries": int(kq["query_id"].nunique()),
        "matched_share_by_label": {
            lab: round(float(kq.loc[kq["esci_label"] == lab, "in_cat"].mean()), 4) for lab in LABELS},
    }, miss.sample(min(30, len(miss)), random_state=42)[["query", "product_id", "product_title"]]


def subset_eda(m, per_q, category):
    """Στατιστικά του υποσυνόλου σε σύγκριση με όλο το small-us."""
    key = slug(category)
    sub_q = per_q.index[per_q["in_subset"]]
    sub = m[m["query_id"].isin(sub_q)]
    qtext = m.drop_duplicates("query_id").set_index("query_id")["query"]
    words = qtext.str.split().str.len()

    def dist(s):
        return {k: round(float(v), 4) for k, v in
                s.value_counts(normalize=True).reindex(LABELS, fill_value=0).items()}

    def has_c(d):
        return round(float(d.groupby("query_id")["esci_label"].apply(lambda s: (s == "C").any()).mean()), 4)

    def sources(d):
        return d.drop_duplicates("query_id")["source"].value_counts(normalize=True).round(4).to_dict()

    matched = sub[sub["in_cat"]].drop_duplicates("product_id")
    out = {
        "category": category,
        "definition": SUBSET_DEF,
        "queries": len(sub_q),
        "rows": len(sub),
        "label_dist": {ALL: dist(m["esci_label"]), key: dist(sub["esci_label"])},
        "queries_with_C": {ALL: has_c(m), key: has_c(sub)},
        "candidates_per_query_median": {ALL: float(per_q["n"].median()),
                                        key: float(per_q.loc[sub_q, "n"].median())},
        "query_words_median": {ALL: float(words.median()), key: float(words.loc[sub_q].median())},
        "source_dist": {ALL: sources(m), key: sources(sub)},
        "matched_products": len(matched),
        "price": {"available_share": round(float(matched["price"].notna().mean()), 4),
                  **matched["price"].describe(percentiles=[.25, .5, .75, .95]).round(2).to_dict()},
        "average_rating": matched["average_rating"].describe().round(2).to_dict(),
        "rating_number_median": float(matched["rating_number"].median()),
    }
    cats = m[m["in_cat"]].drop_duplicates("product_id")
    out["top_main_category"] = cats["main_category"].value_counts().head(15).to_dict()
    out["top_category_level2"] = cats["categories"].str.split(" > ").str[1] \
        .value_counts().head(20).to_dict()
    return sub, sub_q, words, out


# ---------------------------------------------------------------- figures

def fig_frac(per_q, category, out_dir):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(per_q["frac"], bins=np.linspace(0, 1, 21), color=LABEL_COLORS["E"],
            edgecolor="white", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_xlabel(f"ποσοστό products του query που βρέθηκαν στο meta_{category}")
    ax.set_ylabel("queries (log)")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.grid(axis="x", visible=False)
    ax.set_title(f"{category} ανά query (ESCI small, us)", loc="left", color=INK)
    fig.savefig(out_dir / "fraction_per_query.png")
    plt.close(fig)


def fig_label_compare(eda_out, out_dir):
    key = slug(eda_out["category"])
    d = pd.DataFrame(eda_out["label_dist"]).T.loc[[ALL, key], LABELS]
    d.index = ["όλο το small-us", eda_out["category"]]
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
    ax.set_title(f"Κατανομή ετικετών: {eda_out['category']} vs όλο το small-us", loc="left", pad=22)
    fig.savefig(out_dir / "labels_subset_vs_all.png")
    plt.close(fig)


def fig_categories(eda_out, out_dir):
    s = pd.Series(eda_out["top_category_level2"]).head(12)[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(s) + 1))
    ax.barh(s.index, s.values, color=LABEL_COLORS["E"], height=0.6)
    for i, v in enumerate(s.values):
        ax.text(v, i, f" {v:,}", va="center", fontsize=8, color=MUTED)
    ax.set_xlabel("ESCI products (small, us) που βρέθηκαν στο meta")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Υποκατηγορίες {eda_out['category']} (2ο επίπεδο)", loc="left")
    fig.savefig(out_dir / "subcategories.png")
    plt.close(fig)


def fig_price(sub, category, out_dir):
    p = sub[sub["in_cat"]].drop_duplicates("product_id")["price"].dropna()
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(p, bins=np.logspace(0, np.log10(max(p.max(), 10)), 40),
            color=LABEL_COLORS["E"], edgecolor="white", linewidth=0.5)
    ax.set_xscale("log")
    ax.axvline(p.median(), color=INK, linestyle="--", linewidth=1)
    ax.set_xlabel("τιμή ($, log)")
    ax.set_ylabel("products")
    ax.grid(axis="x", visible=False)
    ax.set_title(f"Τιμές {category} products (median ${p.median():.0f}, n={len(p):,})",
                 loc="left")
    fig.savefig(out_dir / "price.png")
    plt.close(fig)


def fig_query_length(words, sub_q, category, out_dir):
    fig, ax = plt.subplots(figsize=(7, 3))
    bins = np.arange(0.5, 16.5, 1)
    ax.hist(words.clip(upper=15), bins=bins, density=True, color=LABEL_COLORS["E"],
            alpha=0.85, label="όλο το small-us", edgecolor="white", linewidth=0.5)
    ax.hist(words.loc[sub_q].clip(upper=15), bins=bins, density=True, histtype="step",
            color=LABEL_COLORS["S"], linewidth=2, label=category)
    ax.set_xlabel("λέξεις (15 = 15+)")
    ax.set_ylabel("ποσοστό queries")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    ax.set_title(f"Μήκος query: {category} vs όλο το small-us", loc="left")
    fig.savefig(out_dir / "query_length.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--category", default="Electronics",
                        help="κατηγορία Amazon 2023 (πρέπει να έχει κατέβει με src.meta)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    category = args.category
    out_dir = ROOT / "results/eda/categories" / slug(category)
    out_dir.mkdir(parents=True, exist_ok=True)
    style()

    esci, meta = load(cfg, category)
    m, per_q, cov = coverage(esci, meta)
    kw, kw_missed = keyword_check(m, category)
    sub, sub_q, words, eda_out = subset_eda(m, per_q, category)

    summary = {"category": category, "coverage": cov, "keyword_check": kw, "subset": eda_out}
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    if kw_missed is not None:
        kw_missed.to_csv(out_dir / "keyword_queries_unmatched_E_sample.csv", index=False)
    qtext = m.drop_duplicates("query_id").set_index("query_id")["query"]
    qtext.loc[sub_q].sample(min(50, len(sub_q)), random_state=42) \
        .to_csv(out_dir / "sample_subset_queries.csv")

    fig_frac(per_q, category, out_dir)
    fig_label_compare(eda_out, out_dir)
    fig_categories(eda_out, out_dir)
    fig_price(sub, category, out_dir)
    fig_query_length(words, sub_q, category, out_dir)
    print(json.dumps({k: summary[k] for k in ["coverage", "keyword_check"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
