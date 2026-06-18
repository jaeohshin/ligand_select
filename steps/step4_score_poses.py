#!/usr/bin/env python3
"""
step4_score_poses.py
Score representative poses per pocket cluster using GNINA.
Usage: python pipeline/steps/step4_score_poses.py --config targets/T2383/config.yaml
"""

from __future__ import annotations
import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
import yaml
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, str(Path(__file__).parent.parent))

PROTEIN_RESNAMES = {
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY",
    "HIS","ILE","LEU","LYS","MET","PHE","PRO","SER",
    "THR","TRP","TYR","VAL"
}


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def cif_to_receptor_pdb(cif_path: Path, out_pdb: Path) -> None:
    """Extract protein only from CIF, write PDB."""
    st = gemmi.read_structure(str(cif_path))
    st.remove_ligands_and_waters()
    st.write_pdb(str(out_pdb))


def cif_to_ligand_sdf(cif_path: Path, smiles: str, out_sdf: Path) -> bool:
    """Extract ligand coords from CIF, assign to SMILES mol, write SDF."""
    st = gemmi.read_structure(str(cif_path))
    coords = []
    for model in st:
        for chain in model:
            for res in chain:
                if res.name in PROTEIN_RESNAMES:
                    continue
                for atom in res:
                    if atom.element.name == "H":
                        continue
                    p = atom.pos
                    coords.append((p.x, p.y, p.z))
        break

    if not coords:
        return False

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    mol = Chem.RemoveHs(mol)

    if mol.GetNumAtoms() != len(coords):
        return False

    conf = mol.GetConformer()
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, (x, y, z))

    writer = Chem.SDWriter(str(out_sdf))
    writer.write(mol)
    writer.close()
    return True


def run_gnina(gnina_sif: str,
              receptor_pdb: Path,
              ligand_sdf: Path,
              out_sdf: Path) -> dict | None:
    """Run GNINA local_only, return score dict."""
    cmd = [
        "singularity", "exec", "--nv", "--bind", "/gpfs",
        gnina_sif, "gnina",
        "--receptor", str(receptor_pdb),
        "--ligand",   str(ligand_sdf),
        "--local_only",
        "--out",      str(out_sdf),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr

    scores = {}
    for line in output.splitlines():
        line = line.strip()
        m = re.match(r"Affinity:\s*([-\d.]+)", line)
        if m:
            scores["gnina_affinity"] = float(m.group(1))
        m = re.match(r"CNNscore:\s*([-\d.]+)", line)
        if m:
            scores["gnina_cnn_score"] = float(m.group(1))
        m = re.match(r"CNNaffinity:\s*([-\d.]+)", line)
        if m:
            scores["gnina_cnn_affinity"] = float(m.group(1))
        m = re.match(r"CNNvariance:\s*([-\d.]+)", line)
        if m:
            scores["gnina_cnn_variance"] = float(m.group(1))
        m = re.match(r"RMSD:\s*([-\d.]+)", line)
        if m:
            scores["gnina_rmsd"] = float(m.group(1))

    return scores if scores else None


def write_pymol_script(df: pd.DataFrame, ref_cif: Path, out_pml: Path) -> None:
    lines = []
    lines.append(f"load {ref_cif}, reference")
    lines.append("hide everything, reference")
    lines.append("show cartoon, reference and polymer")
    lines.append("set cartoon_transparency, 0.7, reference")
    lines.append("color gray80, reference")

    for _, row in df.iterrows():
        name = f"{row['pocket']}_{row['pose_cluster']}"
        lines.append(f"load {row['rep_cif']}, {name}")
        lines.append(f"hide everything, {name}")
        lines.append(f"show sticks, {name} and organic")
        lines.append(f"align {name}, reference")

    lines.append("util.cbag organic")
    lines.append("zoom organic")
    out_pml.write_text("\n".join(lines))
    print(f"Saved: {out_pml}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--top-per-pocket", type=int, default=5,
                        help="Top N clusters per pocket to score")
    args = parser.parse_args()

    config  = load_config(args.config)
    step0   = Path(config["results_dir"]) / "step0"
    step3   = Path(config["results_dir"]) / "step3"
    out_dir = Path(config["results_dir"]) / "step4"
    out_dir.mkdir(parents=True, exist_ok=True)

    minimized_dir = out_dir / "minimized"
    minimized_dir.mkdir(exist_ok=True)

    ref_cif   = Path((step0 / "reference.txt").read_text().strip())
    gnina_sif = config.get("gnina_sif",
                "/gpfs/deepfold/casp/casp17-ligand/models/gnina/gnina.sif")

    ligands = config.get("ligands", [])
    smiles  = ligands[0]["smiles"] if ligands else None
    if not smiles:
        raise ValueError("No SMILES found in config")

    all_clusters = pd.read_csv(step3 / "all_clusters.csv")

    results = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        for pocket, pocket_df in all_clusters.groupby("pocket"):
            top = pocket_df.nlargest(args.top_per_pocket, "n").copy()
            print(f"\n{pocket}: scoring {len(top)} clusters...")

            for _, row in top.iterrows():
                cif_path = Path(row["rep_cif"])
                cluster  = row["pose_cluster"]
                tag      = f"{pocket}_{cluster}"

                print(f"  {tag} (n={row['n']}, "
                      f"model={row['rep_model']}, "
                      f"seed={row['rep_seed']})... ", end="", flush=True)

                # Convert — temp files for receptor/ligand input
                rec_pdb = tmpdir / f"{tag}_receptor.pdb"
                lig_sdf = tmpdir / f"{tag}_ligand.sdf"

                # Minimized output saved permanently
                out_sdf = minimized_dir / f"{tag}_minimized.sdf"

                try:
                    cif_to_receptor_pdb(cif_path, rec_pdb)
                    ok = cif_to_ligand_sdf(cif_path, smiles, lig_sdf)
                    if not ok:
                        print("FAILED (ligand conversion)")
                        continue
                except Exception as e:
                    print(f"FAILED ({e})")
                    continue

                # Score with local_only
                scores = run_gnina(gnina_sif, rec_pdb, lig_sdf, out_sdf)
                if scores is None:
                    print("FAILED (gnina)")
                    continue

                print(f"affinity={scores.get('gnina_affinity'):.2f} "
                      f"cnn={scores.get('gnina_cnn_score'):.3f} "
                      f"rmsd={scores.get('gnina_rmsd', 0):.3f}")

                result = row.to_dict()
                result.update(scores)
                result["minimized_sdf"] = str(out_sdf)
                results.append(result)

    if not results:
        print("No results!")
        return

    df = pd.DataFrame(results)

    # Rank within each pocket by CNNscore
    df["pocket_rank"] = df.groupby("pocket")["gnina_cnn_score"].rank(
        ascending=False, method="first").astype(int)

    # Global top 25
    top25 = df.nlargest(25, "gnina_cnn_score").copy()
    top25["global_rank"] = range(1, len(top25) + 1)

    # Save
    df.to_csv(out_dir / "poses_scored.csv", index=False)
    top25.to_csv(out_dir / "top25.csv", index=False)
    write_pymol_script(top25, ref_cif, out_dir / "load_final.pml")

    print(f"\n=== TOP 25 CANDIDATES ===")
    cols = ["pocket", "pose_cluster", "n", "rep_model", "rep_seed",
            "gnina_affinity", "gnina_cnn_score", "gnina_cnn_affinity",
            "gnina_rmsd"]
    print(top25[cols].to_string(index=False))
    print(f"\nSaved: {out_dir}/poses_scored.csv")
    print(f"Saved: {out_dir}/top25.csv")
    print(f"Saved: {out_dir}/load_final.pml")
    print(f"Minimized poses: {minimized_dir}/")

if __name__ == "__main__":
    main()
