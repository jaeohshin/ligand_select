#!/usr/bin/env python3
"""
step2_validate_pockets.py
Match Step1 pocket clusters to P2Rank predictions, rank by combined score.
Usage: python pipeline/steps/step2_validate_pockets.py --config targets/T2383/config.yaml
"""

from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_p2rank(prank_bin: Path, ref_cif: Path, out_dir: Path) -> Path:
    """Run P2Rank on reference structure, return predictions CSV path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(prank_bin), "predict",
        "-f", str(ref_cif),
        "-o", str(out_dir),
    ]
    print(f"Running P2Rank...")
    subprocess.run(cmd, check=True)

    csv_files = list(out_dir.glob("*_predictions.csv"))
    if not csv_files:
        csv_files = list(out_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No P2Rank predictions CSV found in {out_dir}")
    return csv_files[0]


def parse_p2rank(csv_path: Path) -> pd.DataFrame:
    """Parse P2Rank predictions CSV."""
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    # Keep essential columns
    keep = ["name", "rank", "score", "probability",
            "center_x", "center_y", "center_z",
            "sas_points", "surf_atoms"]
    df = df[[c for c in keep if c in df.columns]]
    df = df.rename(columns={
        "name": "p2rank_name",
        "rank": "p2rank_rank",
        "score": "p2rank_score",
        "probability": "p2rank_prob",
        "center_x": "p2rank_x",
        "center_y": "p2rank_y",
        "center_z": "p2rank_z",
    })
    return df


def match_pockets(step1: pd.DataFrame,
                  p2rank: pd.DataFrame,
                  dist_cutoff: float = 10.0) -> pd.DataFrame:
    """
    For each Step1 pocket, find closest P2Rank pocket.
    Annotate with P2Rank score and distance.
    """
    rows = []
    for _, s1 in step1.iterrows():
        cx, cy, cz = s1["centroid_x"], s1["centroid_y"], s1["centroid_z"]

        best_dist = np.inf
        best_p2 = None
        for _, p2 in p2rank.iterrows():
            dist = np.sqrt(
                (cx - p2["p2rank_x"]) ** 2 +
                (cy - p2["p2rank_y"]) ** 2 +
                (cz - p2["p2rank_z"]) ** 2
            )
            if dist < best_dist:
                best_dist = dist
                best_p2 = p2

        row = s1.to_dict()
        if best_dist <= dist_cutoff and best_p2 is not None:
            row["p2rank_name"]  = best_p2["p2rank_name"]
            row["p2rank_rank"]  = best_p2["p2rank_rank"]
            row["p2rank_score"] = best_p2["p2rank_score"]
            row["p2rank_prob"]  = best_p2["p2rank_prob"]
            row["p2rank_dist"]  = round(best_dist, 3)
        else:
            row["p2rank_name"]  = None
            row["p2rank_rank"]  = None
            row["p2rank_score"] = 0.0
            row["p2rank_prob"]  = 0.0
            row["p2rank_dist"]  = round(best_dist, 3)

        rows.append(row)

    return pd.DataFrame(rows)


def rank_pockets(df: pd.DataFrame, top_final: int) -> pd.DataFrame:
    """
    Combined ranking:
    - normalize cluster size (n) and p2rank_score to [0,1]
    - combined_score = 0.5 * norm_n + 0.5 * norm_p2rank
    """
    df = df.copy()

    # Normalize
    df["norm_n"] = (df["n"] - df["n"].min()) / (df["n"].max() - df["n"].min() + 1e-9)
    p2_max = df["p2rank_score"].max()
    p2_min = df["p2rank_score"].min()
    df["norm_p2rank"] = (df["p2rank_score"] - p2_min) / (p2_max - p2_min + 1e-9)

    df["combined_score"] = 0.5 * df["norm_n"] + 0.5 * df["norm_p2rank"]
    df = df.sort_values("combined_score", ascending=False).reset_index(drop=True)
    df["final_rank"] = df.index + 1
    df["selected"] = df["final_rank"] <= top_final

    return df


def write_pymol_script(df: pd.DataFrame, ref_cif: Path, out_pml: Path) -> None:
    """Write PyMOL script to visualize validated pockets."""
    selected = df[df["selected"]]
    lines = []

    # Load reference protein
    lines.append(f"load {ref_cif}, reference")
    lines.append("hide everything, reference")
    lines.append("show cartoon, reference and polymer")
    lines.append("set cartoon_transparency, 0.7, reference")
    lines.append("color gray80, reference")

    # Load representative ligand per selected pocket
    for _, row in selected.iterrows():
        pocket = row["pocket"]
        rep_cif = row["rep_cif"]
        lines.append(f"load {rep_cif}, {pocket}")
        lines.append(f"hide everything, {pocket}")
        lines.append(f"show sticks, {pocket} and organic")
        lines.append(f"align {pocket}, reference")

    lines.append("util.cbag organic")
    lines.append("zoom organic")
    out_pml.write_text("\n".join(lines))
    print(f"Saved: {out_pml}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dist-cutoff", type=float, default=10.0,
                        help="Max distance (Å) to match Step1 pocket to P2Rank pocket")
    args = parser.parse_args()

    config   = load_config(args.config)
    step0    = Path(config["results_dir"]) / "step0"
    step1    = Path(config["results_dir"]) / "step1"
    out_dir  = Path(config["results_dir"]) / "step2"
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_cif      = Path((step0 / "reference.txt").read_text().strip())
    pockets_df   = pd.read_csv(step1 / "pockets.csv")
    top_final    = config.get("pocket_top_final", 5)

    # P2Rank
    prank_bin = Path(config.get(
        "p2rank_bin",
        "/gpfs/deepfold/casp/casp17-ligand/tools/p2rank_2.4.2/prank"
    ))
    p2rank_out = out_dir / "p2rank"
    p2rank_csv = run_p2rank(prank_bin, ref_cif, p2rank_out)
    p2rank_df  = parse_p2rank(p2rank_csv)
    print(f"  P2Rank found {len(p2rank_df)} pockets")

    # Match + rank
    print(f"\nMatching Step1 pockets to P2Rank (cutoff={args.dist_cutoff}Å)...")
    matched = match_pockets(pockets_df, p2rank_df, args.dist_cutoff)
    ranked  = rank_pockets(matched, top_final)

    # Save
    ranked.to_csv(out_dir / "pockets_validated.csv", index=False)

    # PyMOL script
    write_pymol_script(ranked, ref_cif, out_dir / "load_pockets.pml")

    # Summary
    print(f"\nPocket validation summary:")
    cols = ["pocket", "n", "p2rank_rank", "p2rank_score",
            "p2rank_dist", "combined_score", "final_rank", "selected"]
    print(ranked[cols].to_string(index=False))
    print(f"\nSaved: {out_dir}/pockets_validated.csv")


if __name__ == "__main__":
    main()
