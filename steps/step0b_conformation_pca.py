#!/usr/bin/env python3
"""
step0b_conformation_pca.py
Conformation analysis using BioPython Superimposer (proven correct).
- Top 1 per (model, seed) by iptm_ligand
- Aligns all to initial reference using BioPython Superimposer
- Skips structures with Ca count mismatch
- Joint PCA on all structures
- Plot colored by model (PC1 vs PC2, PC1 vs PC3)
- PC1 histogram per model
- Bimodality detection via PC1 gap
- Updates index.csv with conformation_label
- Updates reference.txt with consensus reference

Usage: python step0b_conformation_pca.py --config targets/T2383/config.yaml
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from Bio.PDB import MMCIFParser
from Bio.PDB.Superimposer import Superimposer
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA


CHAIN_ID = "A"
bparser = MMCIFParser(QUIET=True)


def get_ca_atoms(struct, chain_id=CHAIN_ID):
    for model in struct:
        atoms = []
        for chain in model:
            if chain.id != chain_id:
                continue
            for res in chain:
                if "CA" in res:
                    atoms.append(res["CA"])
        return atoms
    return []


def get_ca_coords(struct, chain_id=CHAIN_ID):
    atoms = get_ca_atoms(struct, chain_id)
    return np.array([a.get_vector().get_array() for a in atoms])


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gap-threshold", type=float, default=0.2,
                        help="PC1 gap fraction to detect bimodality (default: 0.2)")
    args = parser.parse_args()

    config = load_config(args.config)
    target_id = config["target_id"]
    out_dir = Path(config["results_dir"]) / "step0"
    out_dir.mkdir(parents=True, exist_ok=True)

    index_csv = out_dir / "index.csv"
    ref_txt   = out_dir / "reference.txt"

    print(f"[{target_id}] Loading index from {index_csv}")
    df = pd.read_csv(index_csv)
    print(f"  {len(df)} structures in index")

    # ------------------------------------------------------------------
    # Step 1: Top 1 per (model, seed) by iptm_ligand
    # ------------------------------------------------------------------
    print("\n[1] Selecting top 1 per (model, seed) by iptm_ligand...")
    sub = (df.dropna(subset=["iptm_ligand"])
             .sort_values("iptm_ligand", ascending=False)
             .groupby(["model", "seed"], sort=False)
             .first()
             .reset_index())
    models_present = sorted(sub["model"].unique())
    print(f"  Selected {len(sub)} structures from {len(models_present)} models:")
    for m in models_present:
        print(f"    {m}: {(sub['model']==m).sum()} structures")

    # ------------------------------------------------------------------
    # Step 2: Load reference (highest iptm_ligand)
    # ------------------------------------------------------------------
    print("\n[2] Loading alignment reference (highest iptm_ligand)...")
    init_ref_idx = sub["iptm_ligand"].idxmax()
    ref_cif = sub.loc[init_ref_idx, "cif_path"]
    ref_struct = bparser.get_structure("ref", ref_cif)
    ref_atoms  = get_ca_atoms(ref_struct)
    ref_coords = get_ca_coords(ref_struct)
    n_res = len(ref_atoms)
    ref_row = sub.loc[init_ref_idx]
    print(f"  model={ref_row['model']}, seed={ref_row['seed']}, "
          f"iptm_ligand={ref_row['iptm_ligand']:.3f}, n_residues={n_res}")

    # ------------------------------------------------------------------
    # Step 3: Align all structures using BioPython Superimposer
    # ------------------------------------------------------------------
    print(f"\n[3] Aligning {len(sub)} structures (BioPython Superimposer)...")
    sup = Superimposer()
    records = []
    n_skip = 0
    for i, (_, row) in enumerate(sub.iterrows()):
        try:
            struct = bparser.get_structure(f"s{i}", row["cif_path"])
            atoms  = get_ca_atoms(struct)
            if len(atoms) != n_res:
                n_skip += 1
                continue
            sup.set_atoms(ref_atoms, atoms)
            sup.apply(struct.get_atoms())
            coords = get_ca_coords(struct)
            records.append({"row": row, "coords": coords.flatten()})
        except Exception as e:
            warnings.warn(f"Failed {row['cif_path']}: {e}")
            n_skip += 1
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(sub)}...")

    print(f"  {len(records)} loaded, {n_skip} skipped")
    X_flat    = np.array([r["coords"] for r in records])
    sub_valid = pd.DataFrame([r["row"] for r in records]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Step 4: Joint PCA
    # ------------------------------------------------------------------
    print("\n[4] Joint PCA (3 components)...")
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_flat)
    var = pca.explained_variance_ratio_ * 100
    print(f"  PC1={var[0]:.1f}%, PC2={var[1]:.1f}%, PC3={var[2]:.1f}%")

    # ------------------------------------------------------------------
    # Step 5: Plot colored by model
    # ------------------------------------------------------------------
    print("\n[5] Saving PCA plot...")
    model_colors = {"af3": "#4c8eda", "bt2": "#e07b4c",
                    "pt2": "#4cba6e", "of3": "#9b59b6"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"{target_id} — Protein Conformation PCA (top 1 per seed)", fontsize=13)
    for model in models_present:
        mask = (sub_valid["model"] == model).values
        color = model_colors.get(model, "#8C8C8C")
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                        alpha=0.4, s=10, color=color, label=f"{model} (n={mask.sum()})")
        axes[1].scatter(X_pca[mask, 0], X_pca[mask, 2],
                        alpha=0.4, s=10, color=color, label=model)
    axes[0].set_xlabel(f"PC1 ({var[0]:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({var[1]:.1f}%)")
    axes[0].set_title("PC1 vs PC2")
    axes[0].legend(fontsize=9)
    axes[1].set_xlabel(f"PC1 ({var[0]:.1f}%)")
    axes[1].set_ylabel(f"PC3 ({var[2]:.1f}%)")
    axes[1].set_title("PC1 vs PC3")
    axes[1].legend(fontsize=9)
    plt.tight_layout()
    pca_path = out_dir / "conformation_pca.png"
    plt.savefig(pca_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {pca_path}")

    # PC1 histogram
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    fig2.suptitle(f"{target_id} — PC1 Distribution by Model", fontsize=13)
    for model in models_present:
        mask = (sub_valid["model"] == model).values
        color = model_colors.get(model, "#8C8C8C")
        ax2.hist(X_pca[mask, 0], bins=20, alpha=0.6, color=color, label=model)
    ax2.set_xlabel(f"PC1 ({var[0]:.1f}%)")
    ax2.set_ylabel("Count")
    ax2.legend()
    plt.tight_layout()
    hist_path = out_dir / "conformation_pc1_hist.png"
    plt.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {hist_path}")

    # ------------------------------------------------------------------
    # Step 6: Bimodality detection via PC1 gap
    # ------------------------------------------------------------------
    print(f"\n[6] Bimodality detection (gap threshold={args.gap_threshold*100:.0f}%)...")
    pc1_all   = np.sort(X_pca[:, 0])
    pc1_range = pc1_all[-1] - pc1_all[0]
    gaps      = np.diff(pc1_all)
    max_gap   = gaps.max()
    gap_frac  = max_gap / pc1_range if pc1_range > 0 else 0
    bimodal   = gap_frac > args.gap_threshold
    threshold = None
    if bimodal:
        gap_idx   = int(np.argmax(gaps))
        threshold = (pc1_all[gap_idx] + pc1_all[gap_idx + 1]) / 2
    print(f"  PC1 range={pc1_range:.1f}, max_gap={max_gap:.1f} "
          f"({gap_frac*100:.1f}%) -> {'BIMODAL' if bimodal else 'unimodal'}")

    # ------------------------------------------------------------------
    # Step 7: Assign conformation_label to full index
    # ------------------------------------------------------------------
    print("\n[7] Assigning conformation labels...")
    if bimodal:
        pc1_vals  = X_pca[:, 0]
        sub_labels = (pc1_vals >= threshold).astype(int)
        seed_model_to_label = dict(zip(
            zip(sub_valid["model"], sub_valid["seed"]), sub_labels))
        df["conformation_label"] = df.apply(
            lambda r: seed_model_to_label.get((r["model"], r["seed"]), 0), axis=1)
        n0 = (sub_labels == 0).sum()
        n1 = (sub_labels == 1).sum()
        print(f"  threshold={threshold:.1f}, conf0={n0}, conf1={n1}")
    else:
        df["conformation_label"] = 0
        print("  Single conformation, all labeled 0")
    df.to_csv(index_csv, index=False)
    print(f"  Saved updated index.csv")

    # ------------------------------------------------------------------
    # Step 8: Consensus reference (most central in PC space)
    # ------------------------------------------------------------------
    print("\n[8] Selecting consensus reference...")
    dist_m   = cdist(X_pca, X_pca)
    best_pos = int(np.argmin(dist_m.mean(axis=1)))
    ref_row  = sub_valid.iloc[best_pos]
    new_ref_path = ref_row["cif_path"]
    ref_txt.write_text(new_ref_path)
    print(f"  model={ref_row['model']}, seed={ref_row['seed']}, "
          f"sample={ref_row['sample']}, iptm_ligand={ref_row['iptm_ligand']:.3f}")
    print(f"  Updated reference.txt -> {new_ref_path}")

    # ------------------------------------------------------------------
    # Step 9: PyMOL .pml
    # ------------------------------------------------------------------
    print("\n[9] PyMOL script...")
    ref_name = f"ref_{ref_row['model']}_s{int(ref_row['seed'])}"
    pymol_colors = {"af3": "blue", "bt2": "orange", "pt2": "green", "of3": "purple"}
    pml_lines = [
        f"# PyMOL: {target_id} conformation analysis",
        f"# Consensus reference: {ref_row['model']} seed={int(ref_row['seed'])}",
        "", "bg_color white", "set ray_shadows, 0", "",
        "# === Consensus Reference ===",
        f"load {new_ref_path}, {ref_name}",
        f"color red, {ref_name}", f"show cartoon, {ref_name}", "",
        "# === Top 1 per seed per model ===",
    ]
    for model in models_present:
        sub_m = sub_valid[sub_valid["model"] == model].reset_index(drop=True)
        mask = (sub_valid["model"] == model).values
        pc1_m = X_pca[mask, 0]
        color = pymol_colors.get(model, "cyan")
        if bimodal and threshold is not None:
            # One rep per conformation: closest to PC1 median of each side
            left  = np.where(pc1_m < threshold)[0]
            right = np.where(pc1_m >= threshold)[0]
            rep_indices = []
            if len(left):
                rep_indices.append(left[np.argmax(pc1_m[left])])   # closest to threshold from left
            if len(right):
                rep_indices.append(right[np.argmin(pc1_m[right])]) # closest to threshold from right
        else:
            # Unimodal: just the most central (closest to PC1 mean)
            rep_indices = [int(np.argmin(np.abs(pc1_m - pc1_m.mean())))]
        pml_lines.append(f"# {model} ({len(rep_indices)} conformation representatives)")
        for i, idx in enumerate(rep_indices):
            row = sub_m.iloc[idx]
            obj = f"{model}_conf{i}_s{int(row['seed'])}"
            pml_lines += [
                f"load {row['cif_path']}, {obj}",
                f"align {obj}, {ref_name}",
                f"color {color}, {obj}",
                f"show cartoon, {obj}",
            ]
        pml_lines.append("")
    pml_lines += [f"zoom {ref_name}", "orient"]
    pml_path = out_dir / "conformation_view.pml"
    pml_path.write_text("\n".join(pml_lines))
    print(f"  Saved: {pml_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"SUMMARY: {target_id}")
    print(f"  Structures for PCA:    {len(records)} ({n_skip} skipped)")
    print(f"  PC1 variance:          {var[0]:.1f}%")
    print(f"  Bimodal:               {'YES (threshold='+str(round(threshold,1))+')' if bimodal else 'NO'}")
    print(f"  Consensus reference:   {new_ref_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()