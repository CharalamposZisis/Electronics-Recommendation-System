"""EDA στο πλήρες ESCI (όλα τα locales, small και large version).

Γράφει στατιστικά (json/csv) και γραφήματα (png) στο results/eda/esci_full/.
Τα products διαβάζονται ανά locale και κρατιούνται μόνο μήκη/missing
(όχι τα κείμενα), ώστε να χωράει σε ~6GB RAM.

Χρήση:
    python -m src.eda --config configs/data.yaml
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.data import KEYS, LABELS, ROOT, load_config

TEXT_COLS = ["product_title", "product_description", "product_bullet_point",
             "product_brand", "product_color"]
# Κατηγορικά χρώματα (σταθερή σειρά E, S, C, I) και ink για κείμενο/άξονες.
LABEL_COLORS = {"E": "#2a78d6", "S": "#eb6834", "C": "#1baf7a", "I": "#eda100"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"


def style():
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 150, "savefig.bbox": "tight",
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.labelsize": 9, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": MUTED,
        "ytick.color": MUTED, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.frameon": False, "legend.fontsize": 8, "font.size": 9,
    })


def label_dist(s):
    d = s.value_counts(normalize=True).reindex(LABELS, fill_value=0)
    return {k: round(float(v), 4) for k, v in d.items()}


def load_examples(raw, cfg):
    ex = pd.read_parquet(raw / cfg["esci"]["examples_file"])
    ex = ex.drop(columns=["__index_level_0__"], errors="ignore")
    src = pd.read_csv(raw / cfg["esci"]["sources_file"])
    return ex.merge(src, on="query_id", how="left", validate="many_to_one")


def product_stats(path, locales):
    """Ανά (product_id, locale): μήκος σε χαρακτήρες κάθε text πεδίου (NaN αν λείπει/κενό)."""
    parts = []
    for loc in locales:
        t = pq.read_table(path, columns=["product_id", "product_locale", *TEXT_COLS],
                          filters=[("product_locale", "==", loc)])
        cols = {"product_id": t["product_id"].to_numpy(),
                "product_locale": t["product_locale"].to_numpy()}
        for c in TEXT_COLS:
            n = pc.utf8_length(pc.utf8_trim_whitespace(t[c])).to_numpy(zero_copy_only=False)
            n = np.asarray(n, dtype="float64")
            n[n == 0] = np.nan  # κενό string = missing
            cols[c] = n
        brand = t["product_brand"].to_pandas() if loc == "us" else None
        parts.append(pd.DataFrame(cols))
        if brand is not None:
            us_brands = brand
        del t
    return pd.concat(parts, ignore_index=True), us_brands


def overview(ex):
    out = {"rows": len(ex), "queries": int(ex["query_id"].nunique()),
           "products": int(ex[KEYS].drop_duplicates().shape[0])}
    for version in ["small_version", "large_version"]:
        sub = ex[ex[version] == 1]
        out[version] = {}
        for loc, g in sub.groupby("product_locale"):
            out[version][loc] = {
                split: {"rows": len(gs), "queries": int(gs["query_id"].nunique()),
                        "label_dist": label_dist(gs["esci_label"])}
                for split, gs in g.groupby("split")
            }
    return out


def per_query(ex):
    """Στατιστικά ανά query (small_version, ανά locale)."""
    sub = ex[ex["small_version"] == 1]
    g = sub.groupby(["product_locale", "query_id"])
    q = g["esci_label"].value_counts().unstack(fill_value=0).reindex(columns=LABELS, fill_value=0)
    q["n"] = q.sum(axis=1)
    q = q.join(g["query"].first())
    q["q_words"] = q["query"].str.split().str.len()
    q["q_chars"] = q["query"].str.len()
    q = q.reset_index()

    out = {}
    for loc, d in q.groupby("product_locale"):
        out[loc] = {
            "queries": len(d),
            "candidates_per_query": d["n"].describe(percentiles=[.25, .5, .75, .95]).round(2).to_dict(),
            "query_words": d["q_words"].describe(percentiles=[.25, .5, .75, .95]).round(2).to_dict(),
            "share_queries_with_E": round(float((d["E"] > 0).mean()), 4),
            "share_queries_without_E": round(float((d["E"] == 0).mean()), 4),
            "share_queries_all_E": round(float((d["E"] == d["n"]).mean()), 4),
            "share_queries_with_C": round(float((d["C"] > 0).mean()), 4),
        }
    return q, out


def label_by_bucket(ex, q, locale="us"):
    """Κατανομή ετικετών ανά μήκος query και ανά πηγή query (small, us)."""
    sub = ex[(ex["small_version"] == 1) & (ex["product_locale"] == locale)]
    words = q[q["product_locale"] == locale].set_index("query_id")["q_words"]
    bucket = pd.cut(sub["query_id"].map(words), [0, 2, 4, 6, 10, 100],
                    labels=["1-2", "3-4", "5-6", "7-10", "11+"])
    by_len = pd.crosstab(bucket, sub["esci_label"], normalize="index")[LABELS].round(4)
    by_len["rows"] = bucket.value_counts().reindex(by_len.index)
    by_src = pd.crosstab(sub["source"], sub["esci_label"], normalize="index")[LABELS].round(4)
    by_src["queries"] = sub.groupby("source")["query_id"].nunique()
    return by_len, by_src


def missingness(ex, ps):
    """Missing και μήκη πεδίων, μετρημένα στα μοναδικά products του small_version."""
    used = ex.loc[ex["small_version"] == 1, KEYS].drop_duplicates()
    p = used.merge(ps, on=KEYS, how="left", validate="one_to_one")
    miss = p.groupby("product_locale")[TEXT_COLS].apply(lambda d: d.isna().mean()).round(4)
    lengths = p.groupby("product_locale")[TEXT_COLS].median().round(1)
    return miss, lengths


def overlap(ex, locale="us"):
    sub = ex[(ex["small_version"] == 1) & (ex["product_locale"] == locale)]
    tr, te = sub[sub["split"] == "train"], sub[sub["split"] == "test"]
    te_prod = set(te["product_id"])
    return {
        "query_id_overlap_train_test": len(set(tr["query_id"]) & set(te["query_id"])),
        "query_text_overlap_train_test": len(set(tr["query"]) & set(te["query"])),
        "test_products_seen_in_train": round(len(te_prod & set(tr["product_id"])) / len(te_prod), 4),
        "products_in_multiple_queries": round(
            float((sub.groupby("product_id")["query_id"].nunique() > 1).mean()), 4),
    }


# ---------------------------------------------------------------- figures

def fig_labels_by_locale(ov, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, version in zip(axes, ["small_version", "large_version"]):
        locs = sorted(ov[version])
        dist = pd.DataFrame({loc: pd.Series(_merge_splits(ov[version][loc])) for loc in locs}).T
        left = np.zeros(len(locs))
        for lab in LABELS:
            ax.barh(locs, dist[lab], left=left, color=LABEL_COLORS[lab], label=lab,
                    height=0.6, edgecolor="white", linewidth=2)
            for i, v in enumerate(dist[lab]):
                if v > 0.06:
                    ax.text(left[i] + v / 2, i, f"{v:.0%}", ha="center", va="center",
                            fontsize=8, color="white")
            left += dist[lab].to_numpy()
        ax.set_title(version.replace("_", " "))
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
        ax.grid(axis="y", visible=False)
    fig.suptitle("Κατανομή ετικετών ESCI ανά locale", x=0.01, ha="left", y=1.12)
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=4, loc="upper left",
               bbox_to_anchor=(0.01, 1.06))
    fig.savefig(out / "labels_by_locale.png")
    plt.close(fig)


def _merge_splits(loc_dict):
    rows = sum(v["rows"] for v in loc_dict.values())
    return {lab: sum(v["rows"] * v["label_dist"][lab] for v in loc_dict.values()) / rows
            for lab in LABELS}


def fig_hist(q, col, title, xlabel, out, name, clip):
    locs = sorted(q["product_locale"].unique())
    fig, axes = plt.subplots(1, len(locs), figsize=(9, 2.8), sharey=False)
    for ax, loc in zip(axes, locs):
        v = q.loc[q["product_locale"] == loc, col].clip(upper=clip)
        ax.hist(v, bins=np.arange(0.5, clip + 1.5, 1), color=LABEL_COLORS["E"],
                edgecolor="white", linewidth=0.5)
        ax.axvline(v.median(), color=INK, linewidth=1, linestyle="--")
        ax.set_title(f"{loc}  (median {v.median():.0f})")
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("queries")
    fig.suptitle(title, x=0.01, ha="left", y=1.04)
    fig.savefig(out / name)
    plt.close(fig)


def fig_stacked_table(df, title, ylabel, out, name):
    fig, ax = plt.subplots(figsize=(6.5, 0.45 * len(df) + 1.2))
    idx = [str(i) for i in df.index]
    left = np.zeros(len(df))
    for lab in LABELS:
        ax.barh(idx, df[lab], left=left, color=LABEL_COLORS[lab], label=lab,
                height=0.6, edgecolor="white", linewidth=2)
        for i, v in enumerate(df[lab]):
            if v > 0.06:
                ax.text(left[i] + v / 2, i, f"{v:.0%}", ha="center", va="center",
                        fontsize=8, color="white")
        left += df[lab].to_numpy()
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0))
    ax.set_title(title, loc="left", pad=22)
    fig.savefig(out / name)
    plt.close(fig)


def fig_missing(miss, out):
    fig, ax = plt.subplots(figsize=(7, 3))
    cols = [c.replace("product_", "") for c in TEXT_COLS]
    x = np.arange(len(cols))
    locs = list(miss.index)
    w = 0.8 / len(locs)
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    for i, loc in enumerate(locs):
        ax.bar(x + i * w - 0.4 + w / 2, miss.loc[loc], width=w - 0.02, color=colors[i], label=loc)
    ax.set_xticks(x, cols)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.set_ylabel("missing (products)")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 0.75)
    ax.legend(ncol=3, loc="upper right")
    ax.set_title("Ποσοστό products χωρίς τιμή ανά πεδίο (small version)", loc="left")
    fig.savefig(out / "missing_fields.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--out", default="results/eda/esci_full")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw = ROOT / cfg["esci"]["raw_dir"]
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    style()

    ex = load_examples(raw, cfg)
    locales = sorted(ex["product_locale"].unique())
    ps, us_brands = product_stats(raw / cfg["esci"]["products_file"], locales)

    ov = overview(ex)
    q, pq_stats = per_query(ex)
    by_len, by_src = label_by_bucket(ex, q)
    miss, lengths = missingness(ex, ps)
    ov_us = overlap(ex)

    used_us = set(ex.loc[(ex["small_version"] == 1) & (ex["product_locale"] == "us"), "product_id"])
    del ps
    top_brands = us_brands.value_counts().head(30)  # σε όλο το us catalogue

    summary = {"overview": ov, "per_query_small": pq_stats,
               "overlap_small_us": ov_us, "n_products_small_us": len(used_us)}
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    by_len.to_csv(out / "labels_by_query_length_small_us.csv")
    by_src.to_csv(out / "labels_by_query_source_small_us.csv")
    miss.to_csv(out / "missing_fields_small.csv")
    lengths.to_csv(out / "median_field_length_chars_small.csv")
    top_brands.to_csv(out / "top_brands_us_catalogue.csv", header=["products"])

    fig_labels_by_locale(ov, out)
    fig_hist(q, "n", "Υποψήφια products ανά query (small version)", "products",
             out, "candidates_per_query.png", clip=40)
    fig_hist(q, "q_words", "Μήκος query σε λέξεις (small version)", "λέξεις",
             out, "query_length.png", clip=15)
    fig_stacked_table(by_len, "Ετικέτες ανά μήκος query (small, us)", "λέξεις",
                      out, "labels_by_query_length_us.png")
    fig_stacked_table(by_src, "Ετικέτες ανά πηγή query (small, us)", "πηγή",
                      out, "labels_by_query_source_us.png")
    fig_missing(miss, out)

    print(json.dumps({k: summary[k] for k in ["per_query_small", "overlap_small_us"]}, indent=2))
    print(by_len, by_src, miss, lengths, top_brands.head(15), sep="\n\n")


if __name__ == "__main__":
    main()
