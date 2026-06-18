#!/usr/bin/env python3
"""
utils/cif_parser.py
Parse mmCIF files using gemmi. Handles of3 LIG0 quirk automatically.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import gemmi

PROTEIN_RESNAMES = {
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY",
    "HIS","ILE","LEU","LYS","MET","PHE","PRO","SER",
    "THR","TRP","TYR","VAL"
}


def load_structure(cif_path: Path) -> gemmi.Structure:
    return gemmi.read_structure(str(cif_path))


def get_ca_coords(st: gemmi.Structure, chain: str = "A") -> np.ndarray:
    coords = []
    for model in st:
        for ch in model:
            if ch.name != chain:
                continue
            for res in ch:
                if res.name not in PROTEIN_RESNAMES:
                    continue
                for atom in res:
                    if atom.name == "CA" and not atom.is_hydrogen():
                        p = atom.pos
                        coords.append([p.x, p.y, p.z])
        break
    if not coords:
        raise ValueError(f"No Cα atoms found for chain {chain} in {cif_path}")
    return np.array(coords, dtype=float)


def get_ligand_coords(st: gemmi.Structure,
                      chain: str | None = None) -> np.ndarray:
    coords = []
    for model in st:
        for ch in model:
            if chain is not None and ch.name != chain:
                continue
            for res in ch:
                if res.name in PROTEIN_RESNAMES:
                    continue
                for atom in res:
                    if atom.is_hydrogen():
                        continue
                    p = atom.pos
                    coords.append([p.x, p.y, p.z])
        break
    if not coords:
        raise ValueError(f"No ligand heavy atoms found in {cif_path}")
    return np.array(coords, dtype=float)


def get_ligand_centroid(st: gemmi.Structure,
                        chain: str | None = None) -> np.ndarray:
    return get_ligand_coords(st, chain=chain).mean(axis=0)


def get_ca_and_ligand(cif_path: Path,
                      chain_protein: str = "A",
                      chain_ligand: str | None = None
                      ) -> tuple[np.ndarray, np.ndarray]:
    st = load_structure(cif_path)
    ca  = get_ca_coords(st, chain=chain_protein)
    lig = get_ligand_coords(st, chain=chain_ligand)
    return ca, lig


def detect_n_ca(cif_path: Path, chain: str = "A") -> int:
    st = load_structure(cif_path)
    return len(get_ca_coords(st, chain=chain))