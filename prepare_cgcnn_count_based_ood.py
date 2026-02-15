
import os, json, random
from pathlib import Path
import math
import numpy as np
import pandas as pd

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifWriter
import shutil

# Set random seeds for reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)

# --------- config ----------
TASK = "matbench_log_kvrh"  # matbench_log_kvrh  matbench_mp_gap
OUT_ROOT = Path(f"cgcnn_data/{TASK}_CountBasedOOD")  # Output root directory
SYMPREC = 0.1
OOD_TRAINING_RATIO = 0.05  # Sample 5% of the OOD pool into the training set

# Target OOD ratio range
OOD_TARGET_LO = 0.08   # Lower bound: 8%
OOD_TARGET_HI = 0.12   # Upper bound: 12%
OOD_TARGET_MID = 0.10  # Target center: 10%

# If specified, prioritize this method; otherwise fall back to the method closest to target_mid
IDENTIFY_BEST_METHOD = 'spacegroup'  # Options: 'chemical_system', 'spacegroup', 'crystal_system', 'crystal_system+chemical_system'

# Group count filtering parameters
MIN_GROUP_COUNT = 1    # Minimum samples per group (no filtering by default)
MAX_GROUP_COUNT = None  # Maximum pool fraction a group may occupy (None=no cap; e.g., 0.15 prevents domination)

# Maximum number of groups to keep (prevents OOD from spanning too many tiny groups)
MAX_GROUPS = None  # None=no limit

# Fraction drawn into iid_val from (train_pool \ ood_val)
IID_VAL_RATIO = 0.10

# Whether to stratify y==0 vs y>0 (recommended for matbench_mp_gap)
STRATIFY_ZERO_GAP = False


def _ensure_root_atom_init(data_root: Path):
    root_atom_init = data_root / "atom_init.json"
    if root_atom_init.exists():
        return

    candidates = [
        "./atom_init.json",
        Path(__file__).resolve().parent / "data" / "sample-regression" / "atom_init.json",
        Path(__file__).resolve().parent / "data" / "sample-classification" / "atom_init.json",
    ]
    for src in candidates:
        if src.exists():
            shutil.copy2(src, root_atom_init)
            print(f"Copied atom_init.json to dataset root: {root_atom_init}")
            return

    print(f"Warning: atom_init.json not found for dataset root {data_root}")


def ensure_split_loader_links(data_root: Path, split_dir: Path):
    """Prepare split_dir for CIFData loading: atom_init, cifs symlink, id_prop prefix."""
    data_root = Path(data_root)
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    _ensure_root_atom_init(data_root)

    atom_init_path = split_dir / "atom_init.json"
    root_atom_init = data_root / "atom_init.json"
    if not atom_init_path.exists():
        if root_atom_init.exists():
            shutil.copy2(root_atom_init, atom_init_path)
            print(f"Copied atom_init.json to split dir: {atom_init_path}")
        else:
            print(f"Warning: atom_init.json not found in dataset root: {data_root}")

    cifs_symlink = split_dir / "cifs"
    desired_target = str((data_root / "cifs").resolve())
    if cifs_symlink.is_symlink():
        current_target = str(cifs_symlink.resolve())
        if current_target != desired_target:
            cifs_symlink.unlink()
            os.symlink(desired_target, str(cifs_symlink), target_is_directory=True)
            print(f"Recreated directory symlink: cifs -> {desired_target}")
        else:
            print("cifs link already exists; skipping")
    elif not cifs_symlink.exists():
        os.symlink(desired_target, str(cifs_symlink), target_is_directory=True)
        print(f"Created directory symlink: cifs -> {desired_target}")
    else:
        print("cifs path exists and is not a symlink; leaving as is")

    id_prop_files = sorted(split_dir.glob("id_prop*.csv"))
    if not id_prop_files:
        print(f"Warning: no id_prop*.csv found in split dir: {split_dir}")
        return

    for id_prop_path in id_prop_files:
        id_prop_original = Path(str(id_prop_path) + ".original")
        if not id_prop_original.exists():
            shutil.copy2(id_prop_path, id_prop_original)
        with open(id_prop_original, "r") as f_in, open(id_prop_path, "w") as f_out:
            for line in f_in:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    if not parts[0].startswith("cifs/"):
                        parts[0] = "cifs/" + parts[0]
                    f_out.write(",".join(parts) + "\n")
        print(f"Modified {id_prop_path.name} with cifs/ prefix")


def _write_test_id_split_csv(fold_dir: Path, test_df: pd.DataFrame, ood_mask):
    """Write test ID/OOD assignment for evaluation."""
    out = pd.DataFrame({
        "id": test_df["id"].astype(str).values,
        "data_split": np.where(np.asarray(ood_mask), "OOD", "ID"),
    })
    out.to_csv(fold_dir / "test_id_split.csv", index=False)


def _with_cif_prefix(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    sid = out["id"].astype(str)
    out["id"] = np.where(sid.str.startswith("cifs/"), sid, "cifs/" + sid)
    return out


# ---------- helpers ----------
def _safe_spacegroup(struct):
    """Return (spacegroup_number, crystal_system)"""
    try:
        sga = SpacegroupAnalyzer(struct, symprec=SYMPREC)
        sg_num = int(sga.get_space_group_number())
        cs = str(sga.get_crystal_system())
        return sg_num, cs
    except Exception:
        return None, None


def _safe_chemical_system(struct):
    """Return the element set (e.g., 'Fe-O-Si') rather than a reduced formula (e.g., 'Fe2O3')."""
    try:
        comp = struct.composition
        # Some pymatgen versions expose a chemical_system attribute
        cs = getattr(comp, "chemical_system", None)
        if cs:
            return str(cs)
        # Fallback: build from the element set
        elems = sorted([el.symbol for el in comp.elements])
        return "-".join(elems) if elems else None
    except Exception:
        return None


def _infer_id_col(df: pd.DataFrame):
    # More robust: support material_id/mbid so only df['id'] construction changes
    for c in ("material_id", "mbid"):
        if c in df.columns:
            return c
    return None


def _infer_target_col(df: pd.DataFrame):
    # FIX: exclude _row_index to avoid misidentifying it as the target
    ban = {"structure", "material_id", "index", "_row_index", "mbid", "id", "y"}
    candidates = []
    for c in df.columns:
        if c in ban:
            continue
        if np.issubdtype(df[c].dtype, np.number):
            candidates.append(c)
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(f"Multiple numeric columns found: {candidates}. Please set TARGET_COL manually.")


def _get_matbench_task(mb, task_name: str):
    """FIX: Support both dict and list variations of mb.tasks to avoid pulling key strings."""
    tasks = mb.tasks
    if isinstance(tasks, dict):
        if task_name in tasks:
            return tasks[task_name]
        # Fallback: take the first task object
        return next(iter(tasks.values()))
    # tasks might already be a list/iterable
    tasks_list = list(tasks)
    # Try to match by dataset/task name
    for t in tasks_list:
        if getattr(t, "dataset_name", None) == task_name or getattr(t, "task_name", None) == task_name:
            return t
    return tasks_list[0]


def _get_fold_indices(task, fold_idx: int):
    """Use matbench API to get train/test data and extract integer positions."""
    train_inputs, train_outputs = task.get_train_and_val_data(fold_idx)
    test_inputs = task.get_test_data(fold_idx, include_target=False)

    train_idx = task.df.index.get_indexer_for(train_inputs.index)
    test_idx = task.df.index.get_indexer_for(test_inputs.index)

    # FIX: prevent silent mistakes (-1 would make iloc pick the last row)
    bad_train = np.where(train_idx < 0)[0]
    bad_test = np.where(test_idx < 0)[0]
    if len(bad_train) > 0 or len(bad_test) > 0:
        raise RuntimeError(
            f"Index mapping failed (found -1). "
            f"bad_train={len(bad_train)}, bad_test={len(bad_test)}. "
            f"Please check task.df index alignment."
        )

    return train_idx.tolist(), test_idx.tolist()


def _stratified_sample(ids, y, n, rng):
    """Sample n ids with stratification for y==0 vs y>0"""
    ids = np.array(ids)
    y = np.array(y)
    zero_mask = (y == 0.0)
    ids0, ids1 = ids[zero_mask], ids[~zero_mask]
    if len(ids0) == 0 or len(ids1) == 0:
        return rng.choice(ids, size=n, replace=False).tolist()

    ratio0 = len(ids0) / len(ids)
    n0 = int(round(n * ratio0))
    n0 = min(n0, len(ids0))
    n1 = n - n0
    n1 = min(n1, len(ids1))

    if n0 + n1 < n:
        remain = n - (n0 + n1)
        if len(ids0) - n0 >= remain:
            n0 += remain
        else:
            n1 += remain

    pick0 = rng.choice(ids0, size=n0, replace=False) if n0 > 0 else np.array([])
    pick1 = rng.choice(ids1, size=n1, replace=False) if n1 > 0 else np.array([])
    out = np.concatenate([pick0, pick1])
    rng.shuffle(out)
    return out.tolist()


def _normalize_group_cols(frame: pd.DataFrame, group_cols):
    """
    FIX: Normalize NA handling to ensure
    1) groupby treats unknown values as real groups
    2) later membership checks are consistent (avoid NaN != NaN mismatches)
    """
    if isinstance(group_cols, str):
        group_cols = (group_cols,)

    tmp = frame[list(group_cols)].copy()
    for c in group_cols:
        if pd.api.types.is_numeric_dtype(tmp[c]):
            tmp[c] = tmp[c].fillna(-1)
        else:
            tmp[c] = tmp[c].astype("object").where(~tmp[c].isna(), "unknown")
    return tmp, tuple(group_cols)


def _build_ood_mask(train_pool: pd.DataFrame, group_cols, ood_groups: set):
    """FIX: Build the mask with the same NA normalization as select_ood_by_group_count."""
    norm, cols = _normalize_group_cols(train_pool, group_cols)
    if len(cols) == 1:
        return norm[cols[0]].isin(ood_groups)

    key_index = pd.MultiIndex.from_frame(norm, names=list(cols))
    ood_index = pd.MultiIndex.from_tuples(list(ood_groups), names=list(cols))
    return key_index.isin(ood_index)


def _get_group_counts(frame: pd.DataFrame, group_cols):
    """Count each group (using the same NA normalization) and return {group_key: count}."""
    if isinstance(group_cols, str):
        group_cols = (group_cols,)
    
    norm, cols = _normalize_group_cols(frame, group_cols)
    
    if len(cols) == 1:
        group_counts = norm[cols[0]].value_counts().to_dict()
    else:
        group_counts = norm.groupby(list(cols), dropna=False).size().to_dict()
    
    return group_counts



def select_ood_by_group_count(
    df: pd.DataFrame,
    group_cols,
    pool_size: int,
    target_lo: float = 0.08,
    target_hi: float = 0.12,
    target_mid: float = 0.10,
    min_count: int = 1,
    max_count_ratio: float = None,
    max_groups: int = None,
):
    """Select OOD samples by group count while keeping the API unchanged."""
    if isinstance(group_cols, str):
        group_cols = (group_cols,)

    # FIX: normalize NA handling to avoid dropping or mismatching NaN groups
    key_df, group_cols = _normalize_group_cols(df, group_cols)

    # Count occurrences per group
    group_counts = key_df.groupby(list(group_cols), dropna=False).size()

    # Filter by min_count
    valid_groups = group_counts[group_counts >= min_count]

    # Filter by max_count_ratio (prevents any single group from taking over)
    if max_count_ratio is not None:
        # FIX: ceil and enforce at least 1 so int(...) never becomes 0 and wipes everything
        max_count = max(1, int(math.ceil(pool_size * float(max_count_ratio))))
        valid_groups = valid_groups[valid_groups <= max_count]

    if len(valid_groups) == 0:
        print(f"  [WARNING] No valid groups after filtering")
        return {
            'ood_groups': set(),
            'ood_ratio': 0.0,
            'n_groups': 0,
            'group_counts': valid_groups.to_dict(),
            'cumsum_ratio': [],
        }

    # Sort ascending by count so rare groups are chosen first
    valid_groups = valid_groups.sort_values(ascending=True)

    # Apply max_groups limit if specified
    if max_groups is not None and len(valid_groups) > max_groups:
        valid_groups = valid_groups.iloc[:max_groups]

    # Sweep forward and find the prefix closest to the target ratio
    cumsum = 0
    cumsum_ratios = []
    best_idx = 0
    best_diff = float('inf')

    for idx, (grp, cnt) in enumerate(valid_groups.items()):
        cumsum += int(cnt)
        ratio = cumsum / float(pool_size)
        cumsum_ratios.append(ratio)

        # Stop as soon as the ratio lands in the target interval
        if target_lo <= ratio <= target_hi:
            best_idx = idx
            best_diff = 0.0
            break
        
        # While below the lower bound, keep the prefix closest to target_mid
        if ratio < target_lo:
            diff = abs(ratio - target_mid)
            if diff < best_diff:
                best_diff = diff
                best_idx = idx

    # If nothing falls in the interval, fall back to the prefix nearest target_mid
    if best_diff == float('inf'):
        best_idx = int(np.argmin(np.abs(np.array(cumsum_ratios) - target_mid)))

    ood_groups = set(valid_groups.index[:best_idx + 1])
    ood_ratio = cumsum_ratios[best_idx] if best_idx < len(cumsum_ratios) else 0.0

    return {
        'ood_groups': ood_groups,
        'ood_ratio': ood_ratio,
        'n_groups': len(ood_groups),
        'group_counts': valid_groups.to_dict(),
        'cumsum_ratio': cumsum_ratios[:best_idx + 1],
    }


def _upgrade_or_build_meta(df: pd.DataFrame, meta_path: Path):
    """
    FIX:
    - Meta may be incomplete/outdated: incrementally patch it using df
    - If chemical_system looks like 'Fe2O3' (contains digits) treat it as a reduced formula and replace it with an element set such as 'Fe-O'
    """
    need_cols = {"id", "y", "spacegroup", "crystal_system", "chemical_system"}

    if meta_path.exists():
        meta = pd.read_parquet(meta_path)
        # Remove duplicates
        if "id" in meta.columns:
            meta = meta.drop_duplicates(subset=["id"], keep="first")

        # Rebuild entirely if required columns are missing
        if not need_cols.issubset(set(meta.columns)):
            meta = None
    else:
        meta = None

    if meta is None:
        # Full rebuild
        sgs, css, chems = [], [], []
        for s in df["structure"]:
            sg, cs = _safe_spacegroup(s)
            sgs.append(sg)
            css.append(cs)
            chems.append(_safe_chemical_system(s))
        meta = df[["id", "y"]].copy()
        meta["spacegroup"] = sgs
        meta["crystal_system"] = css
        meta["chemical_system"] = chems
        meta.to_parquet(meta_path, index=False)
        return meta

    # Incremental upgrade
    meta = meta.copy()
    meta = meta.set_index("id")

    # Reindex to the df ID space to keep alignment
    meta = meta.reindex(df["id"].values)

    # Determine rows needing backfill: missing spacegroup/crystal_system or missing/outdated chemical_system
    def _looks_like_formula(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return True
        s = str(x)
        return any(ch.isdigit() for ch in s)  # Legacy reduced formulas often contain digits

    need_sg = meta["spacegroup"].isna() if "spacegroup" in meta.columns else pd.Series(True, index=meta.index)
    need_cs = meta["crystal_system"].isna() if "crystal_system" in meta.columns else pd.Series(True, index=meta.index)
    need_chem = meta["chemical_system"].apply(_looks_like_formula) if "chemical_system" in meta.columns else pd.Series(True, index=meta.index)

    # Build an id->structure map for missing rows only
    id2struct = df.set_index("id")["structure"].to_dict()

    # Fill spacegroup/crystal_system
    if need_sg.any() or need_cs.any():
        for sid in meta.index[need_sg | need_cs]:
            struct = id2struct.get(sid, None)
            if struct is None:
                continue
            sg, cs = _safe_spacegroup(struct)
            if "spacegroup" in meta.columns and pd.isna(meta.at[sid, "spacegroup"]):
                meta.at[sid, "spacegroup"] = sg
            if "crystal_system" in meta.columns and pd.isna(meta.at[sid, "crystal_system"]):
                meta.at[sid, "crystal_system"] = cs

    # Upgrade or fill chemical_system
    if need_chem.any():
        for sid in meta.index[need_chem]:
            struct = id2struct.get(sid, None)
            if struct is None:
                continue
            meta.at[sid, "chemical_system"] = _safe_chemical_system(struct)

    # Override y with df values to avoid merge issues from floating-point drift
    meta["y"] = df.set_index("id")["y"].reindex(meta.index).values

    meta = meta.reset_index()
    meta.to_parquet(meta_path, index=False)
    return meta


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    cif_dir = OUT_ROOT / "cifs"
    cif_dir.mkdir(exist_ok=True)

    # ---- load via MatBench ----
    from matbench.bench import MatbenchBenchmark
    mb = MatbenchBenchmark(autoload=False, subset=[TASK])
    task = _get_matbench_task(mb, TASK)
    task.load()

    if not hasattr(task, "df") or task.df is None:
        raise RuntimeError("task.df not found. Please upgrade matbench.")

    df = task.df.copy()
    df = df.reset_index(drop=False).rename(columns={"index": "_row_index"})

    id_col = _infer_id_col(df)
    y_col = _infer_target_col(df)

    print(df.head())

    if id_col is None:
        # Legacy logic preferred mbid; fall back to _row_index to avoid KeyErrors
        if "mbid" in df.columns:
            df["id"] = df["mbid"].astype(str)
        else:
            df["id"] = df["_row_index"].astype(str)
    else:
        df["id"] = df[id_col].astype(str)

    df["y"] = df[y_col].astype(float)

    # ---- compute/upgrade spacegroup, crystal_system, chemical_system ----
    meta_path = OUT_ROOT / "metadata_with_chemistry.parquet"
    meta = _upgrade_or_build_meta(df, meta_path)

    # ---- write CIF ----
    cif_index_path = OUT_ROOT / "cif_written_ids.txt"
    already = set()
    if cif_index_path.exists():
        already = set(x.strip() for x in cif_index_path.read_text().splitlines() if x.strip())

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
            except Exception as e:
                print(f"  [WARN] Failed to write CIF for {sid}: {e}")

        # FIX: only append IDs once the CIF file actually exists
        if out.exists():
            new_written.append(sid)

    if new_written:
        with open(cif_index_path, "a") as f:
            f.write("\n".join(new_written) + "\n")

    # ---- generate per-fold splits ----
    splits_root = OUT_ROOT / "splits"
    splits_root.mkdir(exist_ok=True)

    # Drop duplicate meta IDs to avoid duplicate merges
    meta = meta.drop_duplicates(subset=["id"], keep="first")

    for k in range(len(task.folds)):
        print(f"\n{'='*60}")
        print(f"Processing fold {k}")
        print(f"{'='*60}")

        train_idx, test_idx = _get_fold_indices(task, k)

        train_pool = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        # Retrieve train_pool metadata (FIX: reindex to avoid KeyErrors)
        meta_idxed = meta.set_index("id")
        train_pool_meta = meta_idxed.reindex(train_pool["id"].values).reset_index()
        train_pool = train_pool.merge(train_pool_meta, on=["id", "y"], how="left")
        test_meta = meta_idxed.reindex(test_df["id"].values).reset_index()
        test_df = test_df.merge(test_meta, on=["id", "y"], how="left")

        pool_size = len(train_pool)
        print(f"train_pool size: {pool_size}")

        # Explore multiple grouping dimensions and keep the best outcome
        method_results = {}

        method_specs = [
            ('chemical_system',),
            ('spacegroup',),
            ('crystal_system',),
            ('crystal_system', 'chemical_system'),
        ]

        for cols in method_specs:
            method_name = "+".join(cols)

            # Ensure the columns exist and contain non-NaN values
            if any((c not in train_pool.columns) or train_pool[c].isna().all() for c in cols):
                print(f"  [{method_name}] Skipped (column missing or all NaN)")
                continue

            result = select_ood_by_group_count(
                train_pool,
                cols,
                pool_size=pool_size,
                target_lo=OOD_TARGET_LO,
                target_hi=OOD_TARGET_HI,
                target_mid=OOD_TARGET_MID,
                min_count=MIN_GROUP_COUNT,
                max_count_ratio=MAX_GROUP_COUNT,
                max_groups=MAX_GROUPS,
            )

            method_results[method_name] = result
            print(f"  [{method_name}] ratio={result['ood_ratio']:.3f}, n_groups={result['n_groups']}, "
                  f"in_target={OOD_TARGET_LO <= result['ood_ratio'] <= OOD_TARGET_HI}")

        # Select the best grouping method
        candidate_methods = list(method_results.keys())
        if not candidate_methods:
            raise RuntimeError("No valid grouping columns found for OOD selection")

        def _is_usable_result(r):
            return (r is not None) and (r.get("n_groups", 0) > 0) and (r.get("ood_ratio", 0.0) > 0.0)

        # FIX: fall back if IDENTIFY_BEST_METHOD is unusable (empty result)
        if IDENTIFY_BEST_METHOD and IDENTIFY_BEST_METHOD in method_results and _is_usable_result(method_results[IDENTIFY_BEST_METHOD]):
            best_method = IDENTIFY_BEST_METHOD
        else:
            best_method = min(
                candidate_methods,
                key=lambda m: abs(method_results[m]['ood_ratio'] - OOD_TARGET_MID)
            )

        best_result = method_results[best_method]
        print(f"\n>>> Best method: {best_method} with ratio {best_result['ood_ratio']:.3f}")

        # Apply the chosen grouping
        ood_groups = best_result['ood_groups']
        best_cols = best_method.split('+')

        # FIX: construct ood_mask using the same NA normalization as the selection stage
        ood_mask = _build_ood_mask(train_pool, best_cols, ood_groups)
        test_ood_mask = _build_ood_mask(test_df, best_cols, ood_groups)

        ood_candidates = train_pool[ood_mask].copy()
        rest = train_pool[~ood_mask].copy()

        # Sample a small portion of the OOD pool into the training set
        ood_train_ratio = OOD_TRAINING_RATIO  # Pull 5% of OOD into training
        ood_train_n = int(round(len(ood_candidates) * ood_train_ratio))
        ood_train_ids = rng.choice(ood_candidates["id"].tolist(), size=min(ood_train_n, len(ood_candidates)), replace=False).tolist()
        
        ood_train = ood_candidates[ood_candidates["id"].isin(ood_train_ids)].copy()
        ood_val = ood_candidates[~ood_candidates["id"].isin(ood_train_ids)].copy()

        # Draw the iid_val subset
        iid_n = int(round(len(rest) * IID_VAL_RATIO))
        rest_ids = rest["id"].tolist()
        rest_y = rest["y"].tolist()

        if TASK == 'matbench_mp_gap' and STRATIFY_ZERO_GAP:
            iid_ids = _stratified_sample(rest_ids, rest_y, iid_n, rng)
        else:
            iid_ids = rng.choice(rest_ids, size=min(iid_n, len(rest_ids)), replace=False).tolist()

        iid_val = rest[rest["id"].isin(iid_ids)].copy()
        iid_train = rest[~rest["id"].isin(iid_ids)].copy()
        
        # Merge IID training data with the sampled OOD portion
        train_candidates = pd.concat([iid_train, ood_train], ignore_index=True)

        # Count how many samples each group contributes to OOD and IID
        ood_group_counts = _get_group_counts(ood_val, best_cols)
        iid_group_counts = _get_group_counts(iid_val, best_cols)

        # Persist the list of split IDs
        fold_dir = splits_root / f"fold{k}"
        fold_dir.mkdir(exist_ok=True)

        def dump_ids(name, ids):
            # Match the naming convention of prepare_cgcnn_matbench_mp_gap
            (fold_dir / f"{name}.txt").write_text("\n".join(ids) + "\n")

        def dump_id_prop(name, frame):
            _with_cif_prefix(frame[["id", "y"]]).to_csv(
                fold_dir / f"id_prop_{name}.csv", index=False, header=False
            )

        dump_ids("train_candidates", train_candidates["id"].tolist())
        dump_ids("iid_val", iid_val["id"].tolist())
        dump_ids("ood_val", ood_val["id"].tolist())
        dump_ids("test", test_df["id"].tolist())

        dump_id_prop("train_candidates", train_candidates)
        dump_id_prop("iid_val", iid_val)
        dump_id_prop("ood_val", ood_val)
        dump_id_prop("test", test_df)

        # all
        all_ids = pd.concat([
            train_candidates[["id", "y"]],
            iid_val[["id", "y"]],
            ood_val[["id", "y"]],
            test_df[["id", "y"]]
        ])
        all_ids.drop_duplicates(subset=["id"], inplace=True)
        _with_cif_prefix(all_ids[["id", "y"]]).to_csv(fold_dir / "id_prop.csv", index=False, header=False)

        _write_test_id_split_csv(fold_dir, test_df, test_ood_mask)

        ensure_split_loader_links(OUT_ROOT, fold_dir)

        # Collect split statistics
        stat = {
            "fold": k,
            "train_pool": len(train_pool),
            "train_candidates": len(train_candidates),
            "iid_val": len(iid_val),
            "ood_val": len(ood_val),
            "test": len(test_df),
            "ood_method": best_method,
            "ood_ratio": float(best_result['ood_ratio']),
            "ood_ratio_in_train_pool": float(len(ood_val) / max(1, len(train_pool))),
            "ood_n_groups": best_result['n_groups'],
            "ood_groups": sorted([str(x) for x in ood_groups]),
            "cumsum_ratios": [float(r) for r in best_result['cumsum_ratio']],
            # Include per-group counts for both OOD and IID
            "ood_group_counts": {str(k): int(v) for k, v in ood_group_counts.items()},
            "iid_group_counts": {str(k): int(v) for k, v in iid_group_counts.items()},
        }

        # Store metrics from every method for later reference
        stat['all_methods'] = {}
        for mname, mresult in method_results.items():
            stat['all_methods'][mname] = {
                'ratio': float(mresult['ood_ratio']),
                'n_groups': mresult['n_groups'],
            }

        (fold_dir / "stats.json").write_text(json.dumps(stat, indent=2))

        print(f"[fold{k}] train={len(train_candidates)} iid_val={len(iid_val)} ood_val={len(ood_val)} test={len(test_df)}")
        print(f"\n[fold{k}] OOD group counts:")
        for group, count in sorted(ood_group_counts.items()):
            print(f"  {group}: {count}")
        print(f"\n[fold{k}] IID group counts:")
        for group, count in sorted(iid_group_counts.items()):
            print(f"  {group}: {count}")

    print("\nDone. CIFs:", cif_dir)
    print("Splits:", splits_root)



if __name__ == "__main__":
    main()
