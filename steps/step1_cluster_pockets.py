#!/usr/bin/env python3
"""
step1_cluster_pockets.py
Align all structures to reference, extract ligand centroids, cluster into pockets.
Clusters are done per conformation_label separately.
Pocket names: conf{label}_pocket_{n}

Usage: python pipeline/steps/step1_cluster_pockets.py --config targets/T2383/config.yaml
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from sklearn.cluster import DBSCAN

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.cif_parser import get_ca_and_ligand
from utils.kabsch import kabsch, apply_transform


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def extract_centroids(df: pd.DataFrame,
                      ref_cif: Path,
                      chain_protein: str,
                      chain_ligand: str) -> pd.DataFrame:
    print(f"Loading reference: {ref_cif}")
    ref_ca, _ = get_ca_and_ligand(ref_cif,
                                   chain_protein=chain_protein,
                                   chain_ligand=None)
    n_ref = len(ref_ca)
    print(f"  Reference: {n_ref} Ca atoms")

    centroids = []
    failed = 0
    total = len(df)

    for i, row in df.iterrows():
        if i % 500 == 0:
            print(f"  Processing {i}/{total}...")
        try:
            ca, lig = get_ca_and_ligand(Path(row["cif_path"]),
                                         chain_protein=chain_protein,
                                         chain_ligand=None)
            if len(ca) != n_ref:
                failed += 1
                centroids.append([np.nan, np.nan, np.nan])
                continue

            rot, mob_center, ref_center = kabsch(ca, ref_ca)
            aligned_lig = apply_transform(lig, rot, mob_center, ref_center)
            centroid = aligned_lig.mean(axis=0)
            centroids.append(centroid.tolist())

        except Exception as e:
            failed += 1
            centroids.append([np.nan, np.nan, np.nan])

    if failed:
        print(f"  WARNING: {failed} structures failed")

    centroids = np.array(centroids)
    df = df.copy()
    df["centroid_x"] = centroids[:, 0]
    df["centroid_y"] = centroids[:, 1]
    df["centroid_z"] = centroids[:, 2]
    return df


def cluster_pockets_single(df: pd.DataFrame,
                            eps: float,
                            top_n: int,
                            conf_label: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster pockets for a single conformation group."""
    valid = df.dropna(subset=["centroid_x", "centroid_y", "centroid_z"])
    coords = valid[["centroid_x", "centroid_y", "centroid_z"]].to_numpy()

    labels = DBSCAN(eps=eps, min_samples=1).fit_predict(coords)

    df = df.copy()
    df["pocket"] = "noise"

    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    order = unique[np.argsort(-counts)]
    prefix = f"conf{conf_label}_pocket_" if conf_label != "" else "pocket_"
    label_to_pocket = {label: f"{prefix}{i+1}" for i, label in enumerate(order)}
    pocket_labels = [label_to_pocket.get(l, "noise") for l in labels]
    df.loc[valid.index, "pocket"] = pocket_labels

    # Pocket summary
    summary_rows = []
    for pocket, grp in df.groupby("pocket"):
        if pocket == "noise":
            continue
        summary_rows.append({
            "pocket":      pocket,
            "conf_label":  conf_label if conf_label != "" else 0,
            "n":           len(grp),
            "centroid_x":  grp["centroid_x"].mean(),
            "centroid_y":  grp["centroid_y"].mean(),
            "centroid_z":  grp["centroid_z"].mean(),
        })

    summary = pd.DataFrame(summary_rows)
    if len(summary):
        summary["pocket_num"] = summary["pocket"].str.split("_").str[-1].astype(int)
        summary = summary.sort_values("pocket_num").drop(columns="pocket_num")
        summary = summary.head(top_n).reset_index(drop=True)

    top_pockets = set(summary["pocket"]) if len(summary) else set()
    df.loc[~df["pocket"].isin(top_pockets), "pocket"] = "other"

    return df, summary


def cluster_pockets(df: pd.DataFrame,
                    eps: float,
                    top_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster pockets per conformation_label."""
    conf_labels = sorted(df["conformation_label"].unique())
    print(f"  Conformation labels: {conf_labels}")

    all_dfs = []
    all_summaries = []

    for conf_label in conf_labels:
        conf_df = df[df["conformation_label"] == conf_label].copy()
        print(f"\n  conf{conf_label}: {len(conf_df)} structures")
        conf_df, summary = cluster_pockets_single(conf_df, eps, top_n, conf_label)
        all_dfs.append(conf_df)
        all_summaries.append(summary)
        if len(summary):
            print(f"  Pockets found: {len(summary)}")
            for _, r in summary.iterrows():
                print(f"    {r['pocket']}: n={r['n']}")

    df_out = pd.concat(all_dfs).sort_index()
    summary_out = pd.concat(all_summaries).reset_index(drop=True)
    return df_out, summary_out


def add_representative_cif(df: pd.DataFrame,
                            summary: pd.DataFrame) -> pd.DataFrame:
    rep_cifs = []
    for _, row in summary.iterrows():
        pocket = row["pocket"]
        cx, cy, cz = row["centroid_x"], row["centroid_y"], row["centroid_z"]
        members = df[df["pocket"] == pocket].copy()
        members["dist"] = np.sqrt(
            (members["centroid_x"] - cx) ** 2 +
            (members["centroid_y"] - cy) ** 2 +
            (members["centroid_z"] - cz) ** 2
        )
        rep = members.loc[members["dist"].idxmin()]
        rep_cifs.append(rep["cif_path"])
    summary = summary.copy()
    summary["rep_cif"] = rep_cifs
    return summary


def plot_centroids(df: pd.DataFrame, out_png: Path) -> None:
    def pocket_sort_key(p):
        # handles both "pocket_1" and "conf0_pocket_1"
        return int(p.split("_")[-1])
    pockets = sorted([p for p in df["pocket"].unique()
                      if p not in ("other", "noise")],
                     key=pocket_sort_key)
    cmap = plt.get_cmap("tab10")
    colors = {p: cmap(i % 10) for i, p in enumerate(pockets)}
    colors["other"] = (0.8, 0.8, 0.8, 0.3)
    colors["noise"] = (0.8, 0.8, 0.8, 0.3)

    projections = [
        ("centroid_x", "centroid_y", "X", "Y"),
        ("centroid_x", "centroid_z", "X", "Z"),
        ("centroid_y", "centroid_z", "Y", "Z"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (xc, yc, xl, yl) in zip(axes, projections):
        for pocket in ["other"] + pockets:
            grp = df[df["pocket"] == pocket]
            ax.scatter(grp[xc], grp[yc],
                       s=10, alpha=0.3 if pocket == "other" else 0.7,
                       color=colors[pocket],
                       label=f"{pocket} (n={len(grp)})" if pocket != "other" else None)
        ax.set_xlabel(f"{xl} (Å)")
        ax.set_ylabel(f"{yl} (Å)")
        ax.grid(alpha=0.3)
    axes[-1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle("Aligned ligand centroid clusters (per conformation)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")


def write_pymol_script(summary: pd.DataFrame, out_pml: Path) -> None:
    lines = ["bg_color white", "set ray_shadows, 0", ""]
    ref_pocket = None
    for _, row in summary.iterrows():
        pocket = row["pocket"]
        cif = row["rep_cif"]
        lines.append(f"load {cif}, {pocket}")
        if ref_pocket is None:
            ref_pocket = pocket
    lines += ["", "show cartoon, polymer", "show sticks, organic",
              "set cartoon_transparency, 0.7", ""]
    for _, row in summary.iterrows():
        pocket = row["pocket"]
        if pocket == ref_pocket:
            continue
        lines.append(f"align {pocket}, {ref_pocket}")
    lines += ["", "util.cbag organic", "zoom organic", "zoom"]
    out_pml.write_text("\n".join(lines))
    print(f"Saved: {out_pml}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-structures", type=int, default=None)
    args = parser.parse_args()

    config  = load_config(args.config)
    step0   = Path(config["results_dir"]) / "step0"
    out_dir = Path(config["results_dir"]) / "step1"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(step0 / "index.csv")
    if args.max_structures:
        df = df.sample(n=args.max_structures, random_state=42).reset_index(drop=True)
        print(f"  (test mode: using {args.max_structures} structures)")

    # Ensure conformation_label exists
    if "conformation_label" not in df.columns:
        print("  WARNING: conformation_label not found, defaulting to 0")
        df["conformation_label"] = 0

    ref_cif = Path((step0 / "reference.txt").read_text().strip())

    chain_protein       = config.get("chain_protein", "A")
    chain_ligand        = config.get("chain_ligand", "B")
    eps                 = config.get("pocket_eps", 8.0)
    top_n               = config.get("pocket_top_n", 10)
    split_by_conformation = config.get("split_by_conformation", False)

    print(f"\nExtracting ligand centroids ({len(df)} structures)...")
    df = extract_centroids(df, ref_cif, chain_protein, chain_ligand)

    if split_by_conformation and "conformation_label" in df.columns:
        print(f"\nClustering pockets per conformation (eps={eps}Å, top {top_n})...")
        df, summary = cluster_pockets(df, eps, top_n)
    else:
        if split_by_conformation:
            print("  WARNING: split_by_conformation=true but conformation_label missing, ignoring")
        print(f"\nClustering pockets (eps={eps}Å, top {top_n}, conformation ignored)...")
        # Use single clustering, rename pockets without conf prefix
        df["conformation_label"] = df.get("conformation_label", 0)
        df, summary = cluster_pockets_single(df, eps, top_n, conf_label="")
        # Strip "conf_pocket_" -> "pocket_"
        df["pocket"] = df["pocket"].str.replace("conf_pocket_", "pocket_", regex=False)
        summary["pocket"] = summary["pocket"].str.replace("conf_pocket_", "pocket_", regex=False)
    summary = add_representative_cif(df, summary)

    # Save
    df.to_csv(out_dir / "centroids.csv", index=False)
    summary.to_csv(out_dir / "pockets.csv", index=False)
    plot_centroids(df, out_dir / "centroids.png")

    print(f"\nPocket summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved: {out_dir}/centroids.csv ({len(df)} rows)")
    print(f"Saved: {out_dir}/pockets.csv ({len(summary)} rows)")
    write_pymol_script(summary, out_dir / "load_pockets.pml")


if __name__ == "__main__":
    main()