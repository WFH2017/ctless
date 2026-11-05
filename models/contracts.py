"""Shared data, geometry and experiment contracts. Distances are in cm."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

CACHE_VERSION = 2
CHECKPOINT_VERSION = 3
UNROLLED_LOSS_VERSION = 2  # v1 (including unmarked checkpoints) counted stage gradients twice.


def assert_resume_loss_compatible(checkpoint):
    """Reject silent objective changes; inference of historical weights stays valid.

    Direct-only and stage0 checkpoints never optimized the unrolled loss, so
    they can initialize/resume with the corrected objective after normal checks.
    """
    if checkpoint['args']['num_stages'] > 0 and checkpoint['phase'] == 'unrolled':
        version = checkpoint.get('unrolled_loss_version', 1)
        if version != UNROLLED_LOSS_VERSION:
            raise ValueError(
                f'Unrolled loss version {version} differs from {UNROLLED_LOSS_VERSION}. '
                'Historical checkpoints have an unverified/different gradient-loss objective. '
                'Use the archived source to reproduce that run, or start a separately named '
                'corrected experiment from a verified stage0 checkpoint; do not mix objectives '
                'with --resume.')


def scanner_config(vendor, n_angles):
    name = str(vendor).strip().upper()
    if name in ("GE", "GE MEDICAL SYSTEMS"):
        collimator = "GI-LEHR"
        # Support standard 60-angle and 120-angle dual-head acquisition protocols for GE
        if n_angles in (60, 120):
            subsets = 6
        elif n_angles % 6 == 0:
            subsets = 6
        else:
            raise ValueError(f"GE protocol expects 60 or 120 angles (or multiple of 6), received {n_angles}")
        return ("GE", collimator, subsets)
    elif name in ("SIEMENS", "SIEMENS NM"):
        collimator = "SY-LEHR"
        # Support standard 64-angle and 128-angle acquisition protocols for Siemens
        if n_angles in (64, 128):
            subsets = 8
        elif n_angles % 8 == 0:
            subsets = 8
        else:
            raise ValueError(f"Siemens protocol expects 64 or 128 angles (or multiple of 8), received {n_angles}")
        return ("Siemens", collimator, subsets)
    else:
        raise ValueError(f"Unknown vendor {vendor!r}; supply an explicit supported acquisition protocol")


def geometry(object_shape, dr, projection_shape, projection_dr):
    shape = tuple(int(v) for v in object_shape)
    spacing = tuple(float(v) for v in dr)
    pshape = tuple(int(v) for v in projection_shape)
    pspacing = tuple(float(v) for v in projection_dr)
    if len(shape) != 3 or len(spacing) != 3 or len(pshape) != 2 or len(pspacing) != 2:
        raise ValueError("Expected object XYZ shape/spacing and detector RZ shape/spacing")
    if not np.isfinite(spacing + pspacing).all() or min(spacing + pspacing) <= 0:
        raise ValueError("Spacing must be finite and positive")
    # PyTomography rotate/sum requires a square transverse grid and matched detector bins.
    if shape[0] != shape[1] or pshape != (shape[1], shape[2]):
        raise ValueError(f"Unsupported rotate/sum grids: object {shape}, detector {pshape}")
    if not np.isclose(spacing[0], spacing[1]) or not np.allclose(pspacing, (spacing[1], spacing[2]), rtol=1e-5):
        raise ValueError("Object XY spacing must match; detector RZ must match object YZ spacing")
    return shape, spacing, pshape, pspacing


def source_signature(case):
    sources = {}
    for key in ("projections", "mu_map", "ctless", "label", "support_mask"):
        path = Path(case[key]).resolve() if case.get(key) else None
        stat = path.stat() if path and path.exists() else None
        sources[key] = (str(path), stat.st_size if stat else None, stat.st_mtime_ns if stat else None)
    sources["vendor"] = case.get("vendor")
    sources["patient_id"] = case.get("patient_id")
    sources["config_fingerprint"] = case.get("config_fingerprint")
    sources["preprocessing"] = "raw_dual_window_2iter_scpsf_no_filter_support_v2"
    return hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()


def cache_path(cache_dir, case):
    return Path(cache_dir) / f"{case['case_id']}.v{CACHE_VERSION}.{source_signature(case)[:16]}.pt"


def load_split_cases(specs, splits_dir):
    """Accept one split name/path, a comma-separated list, or a sequence of specs."""
    if isinstance(specs, str):
        specs = specs.split(",")
    cases = []
    for spec in specs:
        path = Path(spec)
        if not path.is_file():
            path = Path(splits_dir) / (spec if str(spec).endswith(".json") else f"{spec}.json")
        with path.open(encoding="utf-8") as stream:
            cases.extend(json.load(stream)["cases"])
    if not cases:
        raise ValueError("Split is empty")
    ids = [c["case_id"] for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate cases in combined split")
    patient_ids(cases)  # Require explicit patient identity for leakage checks.
    return cases


def patient_ids(cases):
    if any(not c.get("patient_id") for c in cases):
        raise ValueError("Every case must provide patient_id for split isolation checks")
    return {str(c["patient_id"]) for c in cases}


def assert_disjoint(first, second, context):
    overlap = set(first) & set(second)
    if overlap:
        raise ValueError(f"Patient leakage in {context}: {len(overlap)} overlapping patients")


def checkpoint_model_config(checkpoint):
    if checkpoint.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError("Legacy checkpoint uses a different training/operator protocol. Retrain with v3; do not silently resume old weights.")
    args = checkpoint["args"]
    return {
        "num_stages": args["num_stages"],
        "base_ch": args["base_ch"],
        "share_weights": args.get("share_weights", False),
        "residual_mode": args.get("residual_mode", "attenuation_grad"),
        "net_depth": args.get("net_depth", 2),
        "norm_mode": args.get("norm_mode", "amax"),
        "input_mode": args.get("input_mode", "dual"),
        "use_film": not args.get("disable_film", False),
    }


def assert_test_isolation(checkpoint, test_cases):
    if "train_patient_ids" not in checkpoint or "val_patient_ids" not in checkpoint:
        raise ValueError("Checkpoint lacks train/validation patient identities; test isolation cannot be verified")
    used = set(checkpoint["train_patient_ids"]) | set(checkpoint["val_patient_ids"])
    assert_disjoint(used, patient_ids(test_cases), "checkpoint train/validation versus test")
