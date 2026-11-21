# Product-Search-Recommendation-System
Research project comparing fine-tuned LLMs and Retrieval-Augmented Generation for intelligent product search, ranking, and recommendations based on Amazon datasets.

Τα μοντέλα εκπαιδεύονται και αξιολογούνται σε **όλες** τις κατηγορίες προϊόντων του ESCI.
Επιπλέον, συγκεκριμένες κατηγορίες (αρχικά **Electronics**) μελετώνται ως case studies,
με ξεχωριστή αξιολόγηση στο αντίστοιχο υποσύνολο.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Πηγές δεδομένων

Η εργασία χρησιμοποιεί δύο δημόσια datasets της Amazon. Κανένα αρχείο δεδομένων δεν
μπαίνει στο git· κατεβαίνουν με τις εντολές παρακάτω.

### 1. Shopping Queries Dataset (ESCI) — Amazon Science
- **Τι είναι:** πραγματικές αναζητήσεις χρηστών του Amazon, η καθεμία με 16–40 υποψήφια
  products και ανθρώπινη ετικέτα σχετικότητας για κάθε ζεύγος:
  **E**xact, **S**ubstitute, **C**omplement, **I**rrelevant. Γλώσσες/χώρες: us, es, jp.
- **Πηγή:** https://github.com/amazon-science/esci-data (Apache-2.0)
- **Αναφορά:** Reddy et al., *Shopping Queries Dataset: A Large-Scale ESCI Benchmark for
  Improving Product Search*, 2022, arXiv:2206.06588
- **Αρχεία:** `shopping_queries_dataset_examples.parquet` (ζεύγη query–product + ετικέτα, split),
  `shopping_queries_dataset_products.parquet` (title, description, bullet points, brand, color),
  `shopping_queries_dataset_sources.csv` (πηγή κάθε query).
- **Χρήση στην εργασία:** ο πυρήνας του benchmark — εκπαίδευση και αξιολόγηση ranking/ταξινόμησης E/S/C/I.
  Δεν περιέχει κατηγορία ούτε τιμή προϊόντος.

### 2. Amazon Reviews 2023 — McAuley Lab (UCSD)
- **Τι είναι:** reviews και metadata προϊόντων Amazon, χωρισμένα σε 33 κατηγορίες.
  Τα metadata έχουν, μεταξύ άλλων, κατηγορίες, τιμή, store, βαθμολογίες.
- **Πηγή:** https://amazon-reviews-2023.github.io
  (αρχεία: `raw/meta_categories/meta_<Category>.jsonl.gz`· έχει κατέβει το `Electronics`, ~1.3GB συμπιεσμένο)
- **Αναφορά:** Hou et al., *Bridging Language and Items for Retrieval and Recommendation*, 2024, arXiv:2403.03952
- **Χρήση στην εργασία:** (α) υποσύνολα του ESCI ανά κατηγορία (case studies), με join
  `product_id` (ESCI) ↔ `parent_asin` (Amazon 2023)· (β) αργότερα, reviews ως evidence για το RAG.

## Δεδομένα (ESCI)

1. Κατέβασμα του [Shopping Queries Dataset](https://github.com/amazon-science/esci-data) μέσα στο `data/esci_data/`:
   ```bash
   cd data/esci_data
   git clone https://github.com/amazon-science/esci-data.git
   cd esci-data && git lfs install && git lfs pull
   ```
2. Δημιουργία των splits (merge examples–products, φίλτρο `small_version`/`us`, val 10% ανά `query_id`, seed 42):
   ```bash
   python -m src.data --config configs/data.yaml
   ```
   Έξοδος: `data/processed/esci_small_us/{train,val,test}.parquet` και `split_info.json` (μεγέθη και κατανομή ετικετών).

## Δεδομένα (Amazon Reviews 2023 — metadata ανά κατηγορία)

Streaming λήψη (το .jsonl.gz δεν γράφεται στον δίσκο) και αποθήκευση μόνο των στηλών
`parent_asin, title, main_category, categories, store, price, average_rating, rating_number`.
Οι κατηγορίες ορίζονται στο `configs/data.yaml` (`amazon_meta.categories`):
```bash
python -m src.meta --config configs/data.yaml               # όλες του config (τώρα: Electronics)
python -m src.meta --category Cell_Phones_and_Accessories   # ή μία συγκεκριμένη
```
Έξοδος: `data/amazon_reviews/meta_<Category>_slim.parquet`.

## EDA

EDA σε όλο το ESCI (όλα τα locales, small και large version):
```bash
python -m src.eda --config configs/data.yaml
```
Έξοδος στο `results/eda/esci_full/`: `summary.json`, πίνακες csv και γραφήματα png
(κατανομή ετικετών, υποψήφια ανά query, μήκος query, πηγή query, missing πεδία).
Χρειάζεται ~6GB RAM.

Η ίδια ανάλυση με επεξηγήσεις, παράδειγμα query και γραφήματα inline:
`notebooks/01_eda_esci_full.ipynb` (καλεί τις συναρτήσεις του `src/eda.py`).

### Υποσύνολα ανά κατηγορία (case studies)

Join ESCI `product_id` ↔ `meta_<Category>` `parent_asin`, coverage, sanity check
και EDA στα queries της κατηγορίας (απαιτεί `src.data` και `src.meta` για την κατηγορία):
```bash
python -m src.category_subset --category Electronics
```
Έξοδος στο `results/eda/categories/<category>/`· notebook: `notebooks/02_eda_category_subset.ipynb`
(η κατηγορία ορίζεται στη μεταβλητή `CATEGORY` στην αρχή).

> **Περιορισμός (Electronics):** το join μέσω ASIN βρίσκει μόνο ~2.5% των ESCI products και, σε
> προφανώς-electronics queries, μόνο ~30% των E products. Το υποσύνολο (856 queries με
> ορισμό «≥50% των E products στο meta») είναι προκαταρκτικό.

## Αξιολόγηση (κοινή για όλα τα μοντέλα)

Κάθε μοντέλο γράφει ένα αρχείο predictions με στήλες `query_id, product_id, score`
(και προαιρετικά `pred_label` ∈ {E,S,C,I}). Οι μετρικές υπολογίζονται **μόνο** από το `src/evaluate.py`:
```bash
python -m src.evaluate --predictions results/<exp>/predictions_test.parquet --split test
```
- **nDCG@k**: graded gains E=1, S=0.1, C=0.01, I=0 (όπως στο paper του ESCI). Τα queries με μόνο I εξαιρούνται.
- **MRR, Recall@k**: σχετικό = E.
- **Accuracy, macro-F1, F1 ανά κλάση, confusion matrix**: όταν υπάρχει `pred_label`.
- Οι ισοβαθμίες στο score σπάνε τυχαία με seed. Υπάρχει και ανάλυση ανά πηγή query (`ranking_by_source`).

## Baselines (re-ranking των υποψηφίων κάθε query)

BM25 (k1=1.2, b=0.75), TF-IDF (cosine) και random, είτε μόνο με τον τίτλο (`*_title`) είτε με
τίτλο+brand+color+bullets+description (`*_text`). Το IDF υπολογίζεται σε όλο τον κατάλογο small-us, χωρίς ετικέτες.
```bash
for c in random bm25_title bm25_text tfidf_title tfidf_text; do
  python -m src.baselines --config configs/$c.yaml
done
python -m src.report --split test     # → results/summary_test.csv
```
Baselines ταξινόμησης E/S/C/I (`src/classifiers.py`):
- `majority`: πάντα η πιο συχνή κλάση του train (E).
- `logreg_lexical` / `logreg_lexical_balanced`: Logistic Regression σε TF-IDF του query, του τίτλου και των
  κοινών τους όρων, μαζί με BM25, overlap και μήκη, χωρίς ή με `class_weight=balanced`. Vectorizer και scaler
  γίνονται fit μόνο στο train και το C επιλέγεται με macro-F1 στο val. Το ranking score είναι
  P(E) + 0.1·P(S) + 0.01·P(C).
```bash
for c in majority logreg_lexical logreg_lexical_balanced; do
  python -m src.classifiers --config configs/$c.yaml
done
```

Έξοδος ανά πείραμα: `results/<exp>/metrics_{val,test}.json`, `config.yaml` και τα predictions
(τα predictions δεν μπαίνουν στο git). Τα `*_text` θέλουν ~7GB RAM.

## Δομή

```
src/        κώδικας (data.py, eda.py, meta.py, category_subset.py, evaluate.py, baselines.py, classifiers.py, report.py)
configs/    ένα yaml ανά πείραμα
notebooks/  EDA και παρουσίαση αποτελεσμάτων
results/    metrics και γραφήματα
data/       δεδομένα (εκτός git, εκτός από τα scripts λήψης)
```
