#!/usr/bin/env python3
"""
step3_cluster_poses.py
For each selected pocket, cluster ligand poses by full ligand RMSD.
Usage: python pipeline/steps/step3_cluster_poses.py --config targets/T2383/config.yaml
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import DBSCAN

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.cif_parser import get_ca_and_ligand
from utils.kabsch import kabsch, apply_transform


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_aligned_ligand(cif_path: Path,
                        ref_ca: np.ndarray,
                        chain_protein: str) -> np.ndarray | None:
    """Load ligand coords aligned to reference via Kabsch on Cα."""
    try:
        ca, lig = get_ca_and_ligand(cif_path,
                                     chain_protein=chain_protein,
                                     chain_ligand=None)
        if len(ca) != len(ref_ca):
            return None
        rot, mob_center, ref_center = kabsch(ca, ref_ca)
        return apply_transform(lig, rot, mob_center, ref_center)
    except Exception:
        return None


def ligand_rmsd(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """RMSD between two ligand coord sets (assumed same atom order)."""
    if coords_a.shape != coords_b.shape:
        return np.inf
    diff = coords_a - coords_b
    return float(np.sqrt((diff ** 2).sum(axis=1).mean()))


def compute_rmsd_matrix(coords_list: list[np.ndarray]) -> np.ndarray:
    """Compute pairwise RMSD matrix."""
    n = len(coords_list)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            r = ligand_rmsd(coords_list[i], coords_list[j])
            mat[i, j] = r
            mat[j, i] = r
    return mat


def cluster_poses(rmsd_matrix: np.ndarray,
                  eps: float,
                  min_samples: int = 1) -> np.ndarray:
    """DBSCAN clustering on precomputed RMSD matrix."""
    return DBSCAN(eps=eps,
                  min_samples=min_samples,
                  metric="precomputed").fit_predict(rmsd_matrix)


def get_representative(coords_list: list[np.ndarray],
                       indices: list[int]) -> int:
    """Return index of structure closest to cluster centroid."""
    cluster_coords = [coords_list[i] for i in indices]
    centroid = np.mean(cluster_coords, axis=0)
    dists = [ligand_rmsd(coords_list[i], centroid) for i in indices]
    return indices[int(np.argmin(dists))]


def write_pymol_script(pocket: str,
                       summary: pd.DataFrame,
                       ref_cif: Path,
                       out_pml: Path) -> None:
    lines = []
    lines.append(f"load {ref_cif}, reference")
    lines.append("hide everything, reference")
    lines.append("show cartoon, reference and polymer")
    lines.append("set cartoon_transparency, 0.7, reference")
    lines.append("color gray80, reference")

    for _, row in summary.iterrows():
        cluster = row["pose_cluster"]
        cif = row["rep_cif"]
        name = f"{pocket}_{cluster}"
        lines.append(f"load {cif}, {name}")
        lines.append(f"hide everything, {name}")
        lines.append(f"show sticks, {name} and organic")
        lines.append(f"align {name}, reference")

    lines.append("util.cbag organic")
    lines.append("zoom organic")
    out_pml.write_text("\n".join(lines))
    print(f"  Saved: {out_pml}")


def process_pocket(pocket: str,
                   members: pd.DataFrame,
                   ref_ca: np.ndarray,
                   chain_protein: str,
                   eps: float,
                   out_dir: Path,
                   ref_cif: Path) -> pd.DataFrame:
    print(f"\n  {pocket}: {len(members)} structures")

    # Load aligned ligand coords
    coords_list = []
    valid_idx = []
    for i, row in members.iterrows():
        coords = load_aligned_ligand(Path(row["cif_path"]),
                                     ref_ca, chain_protein)
        if coords is not None:
            coords_list.append(coords)
            valid_idx.append(i)

    print(f"    Loaded {len(coords_list)} ligand structures")
    if len(coords_list) < 2:
        print(f"    Too few structures, skipping")
        return pd.DataFrame()

    # Check atom count consistency
    n_atoms = [c.shape[0] for c in coords_list]
    most_common = max(set(n_atoms), key=n_atoms.count)
    coords_list_f = []
    valid_idx_f = []
    for c, i in zip(coords_list, valid_idx):
        if c.shape[0] == most_common:
            coords_list_f.append(c)
            valid_idx_f.append(i)
    print(f"    Using {len(coords_list_f)} structures with {most_common} atoms")

    # Pairwise RMSD
    print(f"    Computing pairwise RMSD matrix...")
    rmsd_mat = compute_rmsd_matrix(coords_list_f)

    # Cluster
    labels = cluster_poses(rmsd_mat, eps)
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    order = unique[np.argsort(-counts)]
    label_to_cluster = {l: f"cluster_{i+1}" for i, l in enumerate(order)}

    # Build results
    valid_members = members.loc[valid_idx_f].copy().reset_index(drop=True)
    valid_members["pose_cluster"] = [
        label_to_cluster.get(l, "noise") for l in labels
    ]
    valid_members["pocket"] = pocket

    # Summary per cluster
    rows = []
    for cluster_name in [f"cluster_{i+1}" for i in range(len(order))]:
        cluster_mask = valid_members["pose_cluster"] == cluster_name
        cluster_members = valid_members[cluster_mask]
        local_indices = list(cluster_members.index)
        rep_local = get_representative(coords_list_f, local_indices)
        rep_row = valid_members.loc[rep_local]
        rows.append({
            "pocket":        pocket,
            "pose_cluster":  cluster_name,
            "n":             len(cluster_members),
            "rep_cif":       rep_row["cif_path"],
            "rep_model":     rep_row["model"],
            "rep_seed":      rep_row["seed"],
            "rep_sample":    rep_row["sample"],
            "rep_ranking_score": rep_row["ranking_score"],
            "rep_iptm_ligand":   rep_row["iptm_ligand"],
        })

    summary = pd.DataFrame(rows)
    print(f"    Found {len(summary)} pose clusters")
    print(summary[["pose_cluster", "n", "rep_model",
                   "rep_seed", "rep_ranking_score"]].to_string(index=False))

    # Save per-pocket results
    valid_members.to_csv(out_dir / f"{pocket}_members.csv", index=False)
    summary.to_csv(out_dir / f"{pocket}_clusters.csv", index=False)
    write_pymol_script(pocket, summary, ref_cif,
                       out_dir / f"{pocket}_poses.pml")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--eps", type=float, default=2.0,
                        help="RMSD cutoff (Å) for pose clustering")
    parser.add_argument("--max-per-pocket", type=int, default=None,
                        help="Max structures per pocket (for testing)")
    args = parser.parse_args()

    config  = load_config(args.config)
    step0   = Path(config["results_dir"]) / "step0"
    step1   = Path(config["results_dir"]) / "step1"
    step2   = Path(config["results_dir"]) / "step2"
    out_dir = Path(config["results_dir"]) / "step3"
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_cif       = Path((step0 / "reference.txt").read_text().strip())
    chain_protein = config.get("chain_protein", "A")

    # Load reference Ca
    from utils.cif_parser import get_ca_coords, load_structure
    ref_st = load_structure(ref_cif)
    ref_ca = get_ca_coords(ref_st, chain=chain_protein)
    print(f"Reference: {len(ref_ca)} Cα atoms")

    # Load centroids + selected pockets
    centroids_df = pd.read_csv(step1 / "centroids.csv")
    pockets_df   = pd.read_csv(step2 / "pockets_validated.csv")
    selected     = pockets_df[pockets_df["selected"]]["pocket"].tolist()
    print(f"Selected pockets: {selected}")
    
    max_per_pocket = args.max_per_pocket or config.get("max_per_pocket", None)
    
    all_summaries = []
    for pocket in selected:
        members = centroids_df[centroids_df["pocket"] == pocket].copy()    
        
        if max_per_pocket:
            members = members.nlargest(min(max_per_pocket, len(members)),
                                    "ranking_score")
        summary = process_pocket(pocket, members, ref_ca,
                                 chain_protein, args.eps,
                                 out_dir, ref_cif)
        if not summary.empty:
            all_summaries.append(summary)

    # Save combined summary
    if all_summaries:
        combined = pd.concat(all_summaries, ignore_index=True)
        combined.to_csv(out_dir / "all_clusters.csv", index=False)
        print(f"\nSaved: {out_dir}/all_clusters.csv ({len(combined)} clusters)")


if __name__ == "__main__":
    main()
