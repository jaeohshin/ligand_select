#!/usr/bin/env python3
"""
step5_select_final.py
Generate summary report for human-guided final selection of 5 poses.
Filters low-confidence clusters, ranks candidates, generates PyMOL script.
Usage: python ligand_select/steps/step5_select_final.py --config targets/T2383/config.yaml
"""

from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd
import yaml


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def write_pymol_script(df: pd.DataFrame, ref_cif: Path, out_pml: Path) -> None:
    lines = [
        f"load {ref_cif}, reference",
        "hide everything, reference",
        "show cartoon, reference and polymer",
        "set cartoon_transparency, 0.7, reference",
        "color gray80, reference",
        "",
    ]
    colors = ["blue", "red", "green", "orange", "purple",
              "cyan", "magenta", "yellow", "salmon", "lime"]
    for i, (_, row) in enumerate(df.iterrows()):
        name = f"{row['pocket']}_{row['pose_cluster']}"
        color = colors[i % len(colors)]
        lines += [
            f"# rank={row['candidate_rank']} n={row['n']} "
            f"cnn={row['gnina_cnn_score']} model={row['rep_model']} seed={int(row['rep_seed'])}",
            f"load {row['rep_cif']}, {name}",
            f"hide everything, {name}",
            f"show sticks, {name} and organic",
            f"align {name}, reference",
            f"color {color}, {name} and organic",
            "",
        ]
    lines += ["util.cbag organic", "zoom organic"]
    out_pml.write_text("\n".join(lines))
    print(f"Saved: {out_pml}")


def write_report(df: pd.DataFrame, candidates: pd.DataFrame,
                 target_id: str, out_txt: Path) -> None:
    lines = [
        f"=" * 60,
        f"CASP17 Ligand Selection Report: {target_id}",
        f"=" * 60,
        "",
        f"Pipeline summary:",
        f"  Total scored clusters: {len(df)}",
        f"  Candidates (n >= min_n): {len(candidates)}",
        "",
        f"Candidate poses (ranked by CNN score):",
        "",
    ]

    cols = ["candidate_rank", "pocket", "pose_cluster", "n",
            "conformation_label", "rep_model", "rep_seed", "rep_sample",
            "gnina_affinity", "gnina_cnn_score", "gnina_cnn_affinity", "gnina_rmsd"]
    available = [c for c in cols if c in candidates.columns]
    lines.append(candidates[available].to_string(index=False))
    lines += [
        "",
        "-" * 60,
        "Notes:",
        "- candidate_rank: global rank by gnina_cnn_score",
        "- n: number of structures in this pose cluster",
        "- conformation_label: 0=ConfA, 1=ConfB",
        "- gnina_cnn_score: CNN-based pose quality (higher=better)",
        "- gnina_affinity: Vina affinity kcal/mol (more negative=better)",
        "- gnina_rmsd: RMSD after local minimization (lower=better)",
        "",
        "Select 5 final poses considering:",
        "  1. High cnn_score",
        "  2. Large cluster size (n)",
        "  3. Diversity across pockets",
        "  4. Low gnina_rmsd (stable pose)",
        "-" * 60,
    ]
    out_txt.write_text("\n".join(lines))
    print(f"Saved: {out_txt}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--min-n", type=int, default=5,
                        help="Minimum cluster size to consider (default: 5)")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Top N candidates to show in report (default: 10)")
    args = parser.parse_args()

    config    = load_config(args.config)
    target_id = config["target_id"]
    step0     = Path(config["results_dir"]) / "step0"
    step4     = Path(config["results_dir"]) / "step4"
    out_dir   = Path(config["results_dir"]) / "step5"
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_cif = Path((step0 / "reference.txt").read_text().strip())
    df = pd.read_csv(step4 / "poses_scored.csv")

    print(f"[{target_id}] Total scored clusters: {len(df)}")
    print(f"  Filtering n < {args.min_n}...")

    # Filter low-n clusters
    candidates = df[df["n"] >= args.min_n].copy()
    print(f"  Candidates after filter: {len(candidates)}")

    # Rank by cnn_score
    candidates = candidates.sort_values("gnina_cnn_score", ascending=False).reset_index(drop=True)
    candidates["candidate_rank"] = candidates.index + 1
    candidates = candidates.head(args.top_n)

    # Save
    candidates.to_csv(out_dir / "candidates.csv", index=False)
    write_pymol_script(candidates, ref_cif, out_dir / "candidates.pml")
    write_report(df, candidates, target_id, out_dir / "selection_report.txt")

    # Print summary
    print(f"\n{'='*60}")
    print(f"TOP {args.top_n} CANDIDATES FOR MANUAL SELECTION")
    print(f"{'='*60}")
    cols = ["candidate_rank", "pocket", "pose_cluster", "n",
            "conformation_label", "rep_model", "rep_seed",
            "gnina_cnn_score", "gnina_affinity", "gnina_rmsd"]
    available = [c for c in cols if c in candidates.columns]
    print(candidates[available].to_string(index=False))
    print(f"\n→ Select 5 from above and note their rep_cif paths for submission.")
    print(f"Saved: {out_dir}/candidates.csv")
    print(f"Saved: {out_dir}/candidates.pml")
    print(f"Saved: {out_dir}/selection_report.txt")


if __name__ == "__main__":
    main()
