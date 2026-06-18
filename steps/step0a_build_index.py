#!/usr/bin/env python3
"""
step0a_build_index.py
Crawl all summary.json files for a target and build a master index CSV.
Also selects reference structure (highest ranking_score) for downstream alignment.
Usage: python step0a_build_index.py --config targets/T2383/config.yaml
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
import yaml


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_index(config: dict) -> pd.DataFrame:
    runs_dir  = Path(config["runs_dir"])
    models    = config["models"]
    n_samples = config["n_samples"]

    records = []
    for model in models:
        model_dir = runs_dir / model / "ligand"
        if not model_dir.exists():
            print(f"  WARNING: {model_dir} not found, skipping")
            continue

        seed_dirs = [p for p in model_dir.glob("seed_*")
                     if p.is_dir() and p.name.split("_")[1].isdigit()]
        seed_dirs = sorted(seed_dirs, key=lambda p: int(p.name.split("_")[1]))

        n_found = 0
        for seed_dir in seed_dirs:
            seed = int(seed_dir.name.split("_")[1])
            for sample in range(n_samples):
                sample_dir   = seed_dir / f"sample_{sample}"
                summary_path = sample_dir / "summary.json"
                cif_path     = sample_dir / "model.cif"

                if not summary_path.exists() or not cif_path.exists():
                    continue

                try:
                    with open(summary_path) as f:
                        d = json.load(f)
                except Exception as e:
                    print(f"  WARNING: failed to read {summary_path}: {e}")
                    continue

                records.append({
                    "model":         model,
                    "seed":          seed,
                    "sample":        sample,
                    "ranking_score": d.get("ranking_score"),
                    "plddt":         d.get("plddt"),
                    "iptm":          d.get("iptm"),
                    "ptm":           d.get("ptm"),
                    "gpde":          d.get("gpde"),
                    "has_clash":     d.get("has_clash"),
                    "affinity":      d.get("affinity"),
                    "iptm_ligand":   (d.get("chain_iptm") or {}).get("B"),
                    "cif_path":      str(cif_path),
                })
                n_found += 1

        print(f"  {model}: {n_found} structures found")

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config  = load_config(args.config)
    out_dir = Path(config["results_dir"]) / "step0"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building index for {config['target_id']}...")
    df = build_index(config)

    # Save index
    out_csv = out_dir / "index.csv"
    df.to_csv(out_csv, index=False)

    # Select reference structure
    ref = df.loc[df["ranking_score"].idxmax()]
    ref_path = out_dir / "reference.txt"
    ref_path.write_text(ref["cif_path"])

    # Summary
    print(f"\nSummary:")
    for model, grp in df.groupby("model"):
        rs = grp["ranking_score"]
        il = grp["iptm_ligand"]
        print(f"  {model}: {len(grp)} structures, "
              f"ranking_score mean={rs.mean():.3f} "
              f"min={rs.min():.3f} max={rs.max():.3f} | "
              f"iptm_ligand mean={il.mean():.3f} "
              f"min={il.min():.3f} max={il.max():.3f}")

    print(f"\nReference: {ref['model']} seed={ref['seed']} "
          f"sample={ref['sample']} "
          f"ranking_score={ref['ranking_score']:.3f}")
    print(f"\nSaved: {out_csv} ({len(df)} rows)")
    print(f"Saved: {ref_path}")


if __name__ == "__main__":
    main()