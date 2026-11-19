"""Streaming λήψη του meta_Electronics (Amazon Reviews 2023) σε «λεπτό» parquet.

Το .jsonl.gz (~1.3GB) διαβάζεται απευθείας από το δίκτυο, χωρίς να γραφτεί
στον δίσκο, και κρατιούνται μόνο οι στήλες του config (amazon_meta.columns).

Χρήση:
    python -m src.meta --config configs/data.yaml
"""
import argparse
import gzip
import json
import time
import urllib.request

import pyarrow as pa
import pyarrow.parquet as pq

from src.data import ROOT, load_config

SCHEMA = {
    "parent_asin": pa.string(),
    "title": pa.string(),
    "main_category": pa.string(),
    "categories": pa.string(),     # λίστα ενωμένη με " > "
    "store": pa.string(),
    "price": pa.float64(),
    "average_rating": pa.float64(),
    "rating_number": pa.int64(),
}


def _float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None  # π.χ. "None", "from 12.99", κενό


def clean(rec, columns):
    out = {}
    for c in columns:
        v = rec.get(c)
        if c == "categories":
            v = " > ".join(v) if v else None
        elif c in ("price", "average_rating"):
            v = _float(v)
        elif c == "rating_number":
            v = _float(v)
            v = int(v) if v is not None else None
        out[c] = v
    return out


def stream_to_parquet(url, out_path, columns, batch_size=100_000):
    schema = pa.schema([(c, SCHEMA[c]) for c in columns])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".parquet.tmp")
    n, t0, batch = 0, time.time(), []
    with urllib.request.urlopen(url) as resp, gzip.GzipFile(fileobj=resp) as gz, \
            pq.ParquetWriter(tmp, schema, compression="zstd") as writer:
        for line in gz:
            batch.append(clean(json.loads(line), columns))
            if len(batch) == batch_size:
                writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                n += len(batch)
                batch = []
                print(f"{n:,} records  ({time.time() - t0:.0f}s)", flush=True)
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=schema))
            n += len(batch)
    tmp.rename(out_path)  # μόνο αν ολοκληρώθηκε χωρίς σφάλμα
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)["amazon_meta"]

    out = ROOT / cfg["out_file"]
    n = stream_to_parquet(cfg["url"], out, cfg["columns"])
    print(f"Έτοιμο: {n:,} products → {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
