# Electronics-Recommendation-System
Research project comparing fine-tuned LLMs and Retrieval-Augmented Generation for intelligent electronics product search, ranking, and recommendations based on Amazon Dataset.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

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

## Δομή

```
src/        κώδικας (data.py, ...)
configs/    ένα yaml ανά πείραμα
notebooks/  EDA και παρουσίαση αποτελεσμάτων
results/    metrics και γραφήματα
data/       δεδομένα (εκτός git, εκτός από τα scripts λήψης)
```
