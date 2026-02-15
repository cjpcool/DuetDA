#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resplit CGCNN IID/OOD/Train using fit_difficulty ranking.

You asked:
- fit_difficulty.csv contains IDs from the *training pool*
- New OOD logic: select the *hardest* samples (largest fit_difficulty) as OOD
- Output files and paths must match prepare_cgcnn_count_based_ood.py exactly

This script:
1) Loads MatBench task (same TASK/OUT_ROOT config as prepare_cgcnn_count_based_ood.py)
2) For each MatBench fold k:
   - train_pool/test split comes from MatBench folds (same as prepare script)
   - within train_pool, merge in fit_difficulty
   - pick top-N hardest as OOD candidates (N determined by OOD_TARGET_* ratios)
   - sample OOD_TRAINING_RATIO of OOD into training, rest into ood_val
   - sample IID_VAL_RATIO of remaining into iid_val, rest into iid_train
   - train_candidates = iid_train + ood_train
3) Writes the SAME split artifacts:
   splits/fold{k}/train_candidates.txt, iid_val.txt, ood_val.txt, test.txt
   splits/fold{k}/id_prop_*.csv, id_prop.csv, stats.json

Usage:
  python resplit_fit_difficulty_by_rank.py --fit_csv fit_difficulty.csv

Optional:
  --fold K          only process one fold
  --export_fit_split  write fit_difficulty_with_split.csv into each fold dir

Notes:
- Hardness direction: assumes larger fit_difficulty means harder.
  If you want the opposite, pass --harder_sign=-1
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# import the reference config + helpers
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

import prepare_cgcnn_count_based_ood as base  # noqa: E402


def _read_fit_csv(path: Path, harder_sign: int = 1) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "id" not in df.columns:
        raise ValueError(f"'id' column not found in {path}")
    if "fit_difficulty" not in df.columns:
        raise ValueError(f"'fit_difficulty' column not found in {path}")
    df = df[["id", "fit_difficulty"]].copy()
    df["id"] = df["id"].astype(str)
    df["fit_difficulty"] = pd.to_numeric(df["fit_difficulty"], errors="coerce")
    df["fit_difficulty"] = df["fit_difficulty"] * float(harder_sign)
    return df


def _load_matbench_df(task_name: str):
    """Load MatBench task.df and add standardized columns id, y."""
    from matbench.bench import MatbenchBenchmark

    mb = MatbenchBenchmark(autoload=False, subset=[task_name])
    task = base._get_matbench_task(mb, task_name)
    task.load()

    df = task.df.copy()
    df = df.reset_index(drop=False).rename(columns={"index": "_row_index"})

    id_col = base._infer_id_col(df)
    y_col = base._infer_target_col(df)

    if id_col is None:
        if "mbid" in df.columns:
            df["id"] = df["mbid"].astype(str)
        else:
            df["id"] = df["_row_index"].astype(str)
    else:
        df["id"] = df[id_col].astype(str)

    df["y"] = df[y_col].astype(float)
    return task, df


def _pick_n_ood(pool_size: int) -> int:
    """Pick N OOD samples using OOD_TARGET_LO/HI/MID ratios (same config)."""
    if pool_size <= 0:
        return 0
    n_mid = int(round(pool_size * base.OOD_TARGET_MID))
    n_lo = int(np.ceil(pool_size * base.OOD_TARGET_LO))
    n_hi = int(np.floor(pool_size * base.OOD_TARGET_HI))

    # If bounds are inconsistent (small pool), fall back to mid
    if n_hi < n_lo:
        n = max(1, n_mid)
    else:
        n = min(max(n_mid, n_lo), max(n_hi, 1))

    return min(max(n, 1), pool_size)


def _with_cif_prefix(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    sid = out["id"].astype(str)
    out["id"] = np.where(sid.str.startswith("cifs/"), sid, "cifs/" + sid)
    return out


def _write_split_files(fold_dir: Path, train_candidates, iid_val, ood_val, test_df, stat: dict):
    fold_dir.mkdir(parents=True, exist_ok=True)

    def dump_ids(name, ids):
        (fold_dir / f"{name}.txt").write_text("\n".join(ids) + "\n")

    def dump_id_prop(name, frame):
        _with_cif_prefix(frame[["id", "y"]]).to_csv(fold_dir / f"id_prop_{name}.csv", index=False, header=False)

    dump_ids("train_candidates", train_candidates["id"].tolist())
    dump_ids("iid_val", iid_val["id"].tolist())
    dump_ids("ood_val", ood_val["id"].tolist())
    dump_ids("test", test_df["id"].tolist())

    dump_id_prop("train_candidates", train_candidates)
    dump_id_prop("iid_val", iid_val)
    dump_id_prop("ood_val", ood_val)
    dump_id_prop("test", test_df)

    all_ids = pd.concat([
        train_candidates[["id", "y"]],
        iid_val[["id", "y"]],
        ood_val[["id", "y"]],
        test_df[["id", "y"]],
    ], ignore_index=True)
    all_ids.drop_duplicates(subset=["id"], inplace=True)
    _with_cif_prefix(all_ids[["id", "y"]]).to_csv(fold_dir / "id_prop.csv", index=False, header=False)

    (fold_dir / "stats.json").write_text(json.dumps(stat, indent=2))


def _write_test_id_split_csv(fold_dir: Path, test_df: pd.DataFrame, fit_map: dict, cutoff):
    """Write test ID/OOD assignment under difficulty policy."""
    out = test_df[["id"]].copy()
    out["id"] = out["id"].astype(str)
    if cutoff is None:
        out["data_split"] = "ID"
    else:
        test_fit = out["id"].map(fit_map).fillna(-np.inf)
        out["data_split"] = np.where(test_fit >= float(cutoff), "OOD", "ID")
    out.to_csv(fold_dir / "test_id_split.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit_csv", type=str, default="fit_difficulty.csv")
    ap.add_argument("--task", type=str, default=base.TASK)
    ap.add_argument("--out_root", type=str, default=Path(f"cgcnn_data/{base.TASK}_difficultyOOD1"),)
    ap.add_argument("--fold", type=int, default=None, help="Only process one fold (default: all folds)")
    ap.add_argument("--harder_sign", type=int, default=1, choices=[-1, 1],
                    help="1: larger fit_difficulty is harder (default). -1: smaller is harder")
    ap.add_argument("--export_fit_split", action="store_true",
                    help="Export fit_difficulty_with_split.csv into each fold directory")
    args = ap.parse_args()

    # Reproducibility
    random.seed(base.SEED)
    np.random.seed(base.SEED)
    rng = np.random.default_rng(base.SEED)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    fit_path = Path(args.fit_csv)
    if not fit_path.exists():
        raise FileNotFoundError(f"fit_csv not found: {fit_path.resolve()}")

    df_fit = _read_fit_csv(fit_path, harder_sign=args.harder_sign)
    fit_map = df_fit.set_index("id")["fit_difficulty"].to_dict()

    # Load MatBench (same as prepare script)
    task, df = _load_matbench_df(args.task)

    # Build/upgrade meta + CIFs to match prepare script (safe to rerun)
    cif_dir = out_root / "cifs"
    cif_dir.mkdir(exist_ok=True)
    meta_path = out_root / "metadata_with_chemistry.parquet"

    # Need structure column for meta and CIF writing
    if "structure" not in df.columns:
        raise RuntimeError("MatBench df must include 'structure' column.")

    meta = base._upgrade_or_build_meta(df, meta_path)
    meta = meta.drop_duplicates(subset=["id"], keep="first")

    # Write CIFs (same logic, but do not spam already-written)
    cif_index_path = out_root / "cif_written_ids.txt"
    already = set()
    if cif_index_path.exists():
        already = set(x.strip() for x in cif_index_path.read_text().splitlines() if x.strip())

    from pymatgen.io.cif import CifWriter
    to_write = df[~df["id"].isin(already)]
    new_written = []
    for _, row in to_write.iterrows():
        sid = row["id"]
        struct = row["structure"]
        out = cif_dir / f"{sid}.cif"
        if not out.exists():
            try:
                writer = CifWriter(struct)
                writer.write_file(str(out))
            except Exception:
                pass
        if out.exists():
            new_written.append(sid)

    if new_written:
        with open(cif_index_path, "a") as f:
            f.write("\n".join(new_written) + "\n")

    # Splits
    splits_root = out_root / "splits"
    splits_root.mkdir(exist_ok=True)

    folds = range(len(task.folds)) if args.fold is None else [args.fold]

    for k in folds:
        if k < 0 or k >= len(task.folds):
            raise ValueError(f"fold out of range: {k}, valid 0..{len(task.folds)-1}")

        train_idx, test_idx = base._get_fold_indices(task, k)
        train_pool = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        # Attach meta to train_pool (same merge key)
        meta_idxed = meta.set_index("id")
        train_pool_meta = meta_idxed.reindex(train_pool["id"].values).reset_index()
        train_pool = train_pool.merge(train_pool_meta, on=["id", "y"], how="left")

        # Attach fit_difficulty
        train_pool["fit_difficulty"] = train_pool["id"].map(fit_map)
        missing = train_pool["fit_difficulty"].isna().sum()
        if missing > 0:
            # Not fatal: treat missing as very easy so they won't be picked as OOD
            train_pool["fit_difficulty"] = train_pool["fit_difficulty"].fillna(-np.inf)

        pool_size = len(train_pool)
        n_ood = _pick_n_ood(pool_size)

        # Pick hardest top-N as OOD candidates
        train_sorted = train_pool.sort_values("fit_difficulty", ascending=False)
        ood_candidates = train_sorted.iloc[:n_ood].copy()
        rest = train_sorted.iloc[n_ood:].copy()

        # OOD -> small part into training
        ood_train_n = int(round(len(ood_candidates) * base.OOD_TRAINING_RATIO))
        ood_train_ids = (
            rng.choice(ood_candidates["id"].tolist(), size=min(ood_train_n, len(ood_candidates)), replace=False).tolist()
            if len(ood_candidates) > 0 and ood_train_n > 0 else []
        )
        ood_train = ood_candidates[ood_candidates["id"].isin(ood_train_ids)].copy()
        ood_val = ood_candidates[~ood_candidates["id"].isin(ood_train_ids)].copy()

        # IID val sampled from rest
        iid_n = int(round(len(rest) * base.IID_VAL_RATIO))
        rest_ids = rest["id"].tolist()
        rest_y = rest["y"].tolist()

        if args.task == "matbench_mp_gap" and base.STRATIFY_ZERO_GAP:
            iid_ids = base._stratified_sample(rest_ids, rest_y, iid_n, rng)
        else:
            iid_ids = rng.choice(rest_ids, size=min(iid_n, len(rest_ids)), replace=False).tolist() if iid_n > 0 else []

        iid_val = rest[rest["id"].isin(iid_ids)].copy()
        iid_train = rest[~rest["id"].isin(iid_ids)].copy()

        train_candidates = pd.concat([iid_train, ood_train], ignore_index=True)

        # Stats
        cutoff = float(ood_candidates["fit_difficulty"].min()) if len(ood_candidates) else None
        stat = {
            "fold": int(k),
            "train_pool": int(len(train_pool)),
            "train_candidates": int(len(train_candidates)),
            "iid_val": int(len(iid_val)),
            "ood_val": int(len(ood_val)),
            "test": int(len(test_df)),
            "ood_method": "fit_difficulty_topk",
            "ood_ratio_target_mid": float(base.OOD_TARGET_MID),
            "ood_ratio_effective": float(len(ood_candidates) / max(1, len(train_pool))),
            "ood_n": int(len(ood_candidates)),
            "ood_cutoff_fit_difficulty": cutoff,
            "ood_training_ratio_within_ood": float(base.OOD_TRAINING_RATIO),
            "iid_val_ratio_within_rest": float(base.IID_VAL_RATIO),
            "missing_fit_difficulty_in_train_pool": int(missing),
            "harder_sign": int(args.harder_sign),
        }

        fold_dir = splits_root / f"fold{k}"
        _write_split_files(fold_dir, train_candidates, iid_val, ood_val, test_df, stat)
        _write_test_id_split_csv(fold_dir, test_df, fit_map, cutoff)
        base.ensure_split_loader_links(out_root, fold_dir)

        if args.export_fit_split:
            split_map = {i: "train" for i in train_candidates["id"].tolist()}
            split_map.update({i: "iid" for i in iid_val["id"].tolist()})
            split_map.update({i: "ood" for i in ood_val["id"].tolist()})
            out_fit = pd.read_csv(fit_path)
            out_fit["id"] = out_fit["id"].astype(str)
            out_fit["split"] = out_fit["id"].map(split_map).fillna(out_fit.get("split", "train"))
            out_fit.to_csv(fold_dir / "fit_difficulty_with_split.csv", index=False)

        print(f"[fold{k}] train={len(train_candidates)} iid_val={len(iid_val)} ood_val={len(ood_val)} test={len(test_df)} | cutoff={cutoff}")

    print("\nDone.")
    print("CIFs:", cif_dir)
    print("Splits:", splits_root)


if __name__ == "__main__":
    main()
