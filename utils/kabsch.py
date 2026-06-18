#!/usr/bin/env python3
"""
utils/kabsch.py
Kabsch algorithm for optimal superposition of two point sets.
"""

from __future__ import annotations
import numpy as np


def kabsch(mobile: np.ndarray, reference: np.ndarray
           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute optimal rotation and translation to align mobile onto reference.
    Both arrays shape (N, 3).
    Returns: (rotation 3x3, mobile_center, reference_center)
    """
    if mobile.shape != reference.shape:
        raise ValueError(f"Shape mismatch: {mobile.shape} vs {reference.shape}")

    mob_center = mobile.mean(axis=0)
    ref_center = reference.mean(axis=0)

    p = mobile    - mob_center
    q = reference - ref_center

    H = p.T @ q
    U, _, Vt = np.linalg.svd(H)
    rot = U @ Vt
    if np.linalg.det(rot) < 0:
        U[:, -1] *= -1
        rot = U @ Vt

    return rot, mob_center, ref_center


def apply_transform(coords: np.ndarray,
                    rot: np.ndarray,
                    mob_center: np.ndarray,
                    ref_center: np.ndarray) -> np.ndarray:
    """Apply rotation + translation to coords."""
    return (coords - mob_center) @ rot + ref_center


def align_and_transform(mobile_coords: np.ndarray,
                        mobile_ca: np.ndarray,
                        reference_ca: np.ndarray) -> np.ndarray:
    """
    Align mobile_ca onto reference_ca, apply same transform to mobile_coords.
    Useful for aligning ligand coords using protein backbone fit.
    """
    rot, mob_center, ref_center = kabsch(mobile_ca, reference_ca)
    return apply_transform(mobile_coords, rot, mob_center, ref_center)


def rmsd(mobile: np.ndarray, reference: np.ndarray) -> float:
    """RMSD between two aligned coordinate sets."""
    diff = mobile - reference
    return float(np.sqrt((diff ** 2).sum(axis=1).mean()))


def aligned_rmsd(mobile: np.ndarray, reference: np.ndarray) -> float:
    """Kabsch-align then compute RMSD."""
    rot, mob_center, ref_center = kabsch(mobile, reference)
    aligned = apply_transform(mobile, rot, mob_center, ref_center)
    return rmsd(aligned, reference)
