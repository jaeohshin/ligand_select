#!/usr/bin/env python3
"""
step2_validate_pockets.py
Match Step1 pocket clusters to P2Rank predictions, rank by combined score.
P2Rank is run on each pocket's rep_cif (most central structure) for consistency.
If split_by_conformation=true in config, ranking is done per conformation label.
Usage: python ligand_select/steps/step2_validate_pockets.py --config targets/T2383/config.yaml
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


def run_p2rank(prank_bin: Path, cif: Path, out_dir: Path) -> pd.DataFrame | None:
    """Run P2Rank on a single CIF, return parsed predictions DataFrame."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(prank_bin), "predict", "-f", str(cif), "-o", str(out_dir)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"  WARNING: P2Rank failed for {cif}: {e}")
        return None
    csv_files = list(out_dir.glob("*_predictions.csv")) or list(out_dir.glob("*.csv"))
    if not csv_files:
        print(f"  WARNING: No P2Rank predictions CSV found in {out_dir}")
        return None
    df = pd.read_csv(csv_files[0], skipinitialspace=True)
    df.columns = df.columns.str.strip()
    keep = ["name", "rank", "score", "probability",
            "center_x", "center_y", "center_z"]
    df = df[[c for c in keep if c in df.columns]]
    df = df.rename(columns={
        "name": "p2rank_name", "rank": "p2rank_rank",
        "score": "p2rank_score", "probability": "p2rank_prob",
        "center_x": "p2rank_x", "center_y": "p2rank_y", "center_z": "p2rank_z",
    })
    return df


def match_pocket_to_p2rank(centroid: np.ndarray,
                            p2rank_df: pd.DataFrame,
                            dist_cutoff: float) -> dict:
    """Find closest P2Rank pocket to a centroid."""
    if p2rank_df is None or len(p2rank_df) == 0:
        return {"p2rank_name": None, "p2rank_rank": None,
                "p2rank_score": 0.0, "p2rank_prob": 0.0, "p2rank_dist": np.nan}

    best_dist, best_p2 = np.inf, None
    for _, p2 in p2rank_df.iterrows():
        dist = np.linalg.norm(centroid - np.array([p2["p2rank_x"],
                                                    p2["p2rank_y"],
                                                    p2["p2rank_z"]]))
        if dist < best_dist:
            best_dist, best_p2 = dist, p2

    if best_dist <= dist_cutoff and best_p2 is not None:
        return {
            "p2rank_name":  best_p2["p2rank_name"],
            "p2rank_rank":  best_p2["p2rank_rank"],
            "p2rank_score": best_p2["p2rank_score"],
            "p2rank_prob":  best_p2["p2rank_prob"],
            "p2rank_dist":  round(best_dist, 3),
        }
    return {"p2rank_name": None, "p2rank_rank": None,
            "p2rank_score": 0.0, "p2rank_prob": 0.0,
            "p2rank_dist": round(best_dist, 3)}


def validate_pockets(pockets_df: pd.DataFrame,
                     prank_bin: Path,
                     p2rank_base: Path,
                     dist_cutoff: float) -> pd.DataFrame:
    """Run P2Rank on each pocket's rep_cif and match to pocket centroid."""
    rows = []
    for _, pocket_row in pockets_df.iterrows():
        pocket = pocket_row["pocket"]
        rep_cif = Path(pocket_row["rep_cif"])
        centroid = np.array([pocket_row["centroid_x"],
                             pocket_row["centroid_y"],
                             pocket_row["centroid_z"]])

        print(f"  {pocket}: running P2Rank on {rep_cif.parent.name}/{rep_cif.name}...")
        p2rank_out = p2rank_base / pocket
        p2rank_df = run_p2rank(prank_bin, rep_cif, p2rank_out)

        match = match_pocket_to_p2rank(centroid, p2rank_df, dist_cutoff)
        row = pocket_row.to_dict()
        row.update(match)
        rows.append(row)

    return pd.DataFrame(rows)


def rank_pockets_single(df: pd.DataFrame, top_final: int) -> pd.DataFrame:
    df = df.copy()
    df["norm_n"] = (df["n"] - df["n"].min()) / (df["n"].max() - df["n"].min() + 1e-9)
    p2_min, p2_max = df["p2rank_score"].min(), df["p2rank_score"].max()
    df["norm_p2rank"] = (df["p2rank_score"] - p2_min) / (p2_max - p2_min + 1e-9)
    df["combined_score"] = 0.5 * df["norm_n"] + 0.5 * df["norm_p2rank"]
    df = df.sort_values("combined_score", ascending=False).reset_index(drop=True)
    df["final_rank"] = df.index + 1
    df["selected"] = df["final_rank"] <= top_final
    df = df.round(4)
    return df


def rank_pockets(df: pd.DataFrame,
                 top_final: int,
                 split_by_conformation: bool) -> pd.DataFrame:
    if split_by_conformation and "conf_label" in df.columns:
        parts = []
        for conf_label in sorted(df["conf_label"].unique()):
            conf_df = df[df["conf_label"] == conf_label].copy()
            print(f"  Ranking conf{conf_label} ({len(conf_df)} pockets)...")
            parts.append(rank_pockets_single(conf_df, top_final))
        return pd.concat(parts).reset_index(drop=True)
    else:
        return rank_pockets_single(df, top_final)


def write_pymol_script(df: pd.DataFrame, ref_cif: Path, out_pml: Path) -> None:
    selected = df[df["selected"]]
    lines = [
        f"load {ref_cif}, reference",
        "hide everything, reference",
        "show cartoon, reference and polymer",
        "set cartoon_transparency, 0.7, reference",
        "color gray80, reference",
    ]
    for _, row in selected.iterrows():
        pocket = row["pocket"]
        rep_cif = row["rep_cif"]
        lines += [
            f"load {rep_cif}, {pocket}",
            f"hide everything, {pocket}",
            f"show sticks, {pocket} and organic",
            f"align {pocket}, reference",
        ]
    lines += ["util.cbag organic", "zoom organic"]
    out_pml.write_text("\n".join(lines))
    print(f"Saved: {out_pml}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dist-cutoff", type=float, default=10.0,
                        help="Max distance (Å) to match pocket centroid to P2Rank pocket")
    args = parser.parse_args()

    config  = load_config(args.config)
    step0   = Path(config["results_dir"]) / "step0"
    step1   = Path(config["results_dir"]) / "step1"
    out_dir = Path(config["results_dir"]) / "step2"
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_cif   = Path((step0 / "reference.txt").read_text().strip())
    pockets_df = pd.read_csv(step1 / "pockets.csv")
    top_final  = config.get("pocket_top_final", 5)
    split_by_conformation = config.get("split_by_conformation", False)
    prank_bin  = Path(config.get("p2rank_bin",
                      "/gpfs/deepfold/casp/casp17-ligand/tools/p2rank_2.4.2/prank"))

    print(f"split_by_conformation: {split_by_conformation}")
    print(f"\nRunning P2Rank on each pocket's rep_cif ({len(pockets_df)} pockets)...")
    p2rank_base = out_dir / "p2rank"
    matched = validate_pockets(pockets_df, prank_bin, p2rank_base, args.dist_cutoff)

    print(f"\nRanking pockets (top_final={top_final})...")
    ranked = rank_pockets(matched, top_final, split_by_conformation)

    ranked.to_csv(out_dir / "pockets_validated.csv", index=False)
    write_pymol_script(ranked, ref_cif, out_dir / "load_pockets.pml")

    print(f"\nPocket validation summary:")
    cols = ["pocket", "n", "p2rank_rank", "p2rank_score",
            "p2rank_dist", "combined_score", "final_rank", "selected"]
    print(ranked[cols].to_string(index=False))
    print(f"\nSaved: {out_dir}/pockets_validated.csv")


if __name__ == "__main__":
    main()