"""
Train BP prediction models (SBP + DBP) tu PPG-BP Dataset (Liang et al. 2018).

Dataset thuc te nam o:
    DATA/PPG_BP database/
    |-- Data File/
    |   |-- 0_subject/
    |   |   |-- 100_1.txt  (2100 mau / file @ 1000 Hz, tab-separated, 1 dong)
    |   |   |-- ...         (657 files = 219 subjects x 3 recordings)
    |   `-- PPG-BP dataset.xlsx
    `-- Table 1.xlsx         (ban duplicate voi PPG-BP dataset.xlsx)

Quy trinh:
    1. Load labels tu Excel (subject_ID -> SBP, DBP, HR, age, ...)
    2. Load 657 file .txt, resample 1000 Hz -> 100 Hz (khop ESP32 + MAX30102)
    3. Extract 38 features qua ml_models.extract_features()
    4. Aggregate 3 recordings/subject -> mean vector (co the doi thanh per-recording neu can)
    5. Train RF / GB / SVR cho moi target (SBP, DBP); cross-validation 5-fold
    6. Chon model tot nhat theo MAE, luu ra models/<model_type>_models.pkl
       (format khop ClassicalMLModel._try_load_models trong ml_models.py)

CHAY:
    cd "ML code"
    pip install -r requirements.txt
    pip install pandas openpyxl
    python train_ppg_bp.py

Script KHONG tich hop vao main.py — chi tao file .pkl cho ClassicalMLModel tu load.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Import feature extraction tu project chinh
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ml_models import (  # noqa: E402
    MODEL_DIR,
    NUM_TOTAL_FEATURES,
    PPGFeatures,
    extract_features,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_ppg_bp")

# ============================================================
# Cau hinh
# ============================================================
PROJECT_ROOT = HERE.parent
DATA_ROOT = PROJECT_ROOT / "DATA" / "PPG_BP database"
SIGNAL_DIR = DATA_ROOT / "Data File" / "0_subject"
LABEL_XLSX_CANDIDATES = [
    DATA_ROOT / "Data File" / "PPG-BP dataset.xlsx",
    DATA_ROOT / "Table 1.xlsx",
]

FS_RAW = 1000      # Hz — sample rate goc PPG-BP
FS_TARGET = 100    # Hz — khop ESP32 + MAX30102
BP_TARGETS = ("sbp", "dbp")
BP_RANGES = {"sbp": (70.0, 200.0), "dbp": (40.0, 130.0)}

# ============================================================
# Load labels
# ============================================================
def _find_label_file() -> Path:
    for p in LABEL_XLSX_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Khong tim thay file label Excel trong cac duong dan:\n  - "
        + "\n  - ".join(str(p) for p in LABEL_XLSX_CANDIDATES)
    )


def _normalize_colname(c: str) -> str:
    """Chuan hoa ten cot: lowercase, loai khoang trang va dau cau."""
    return "".join(ch for ch in c.lower() if ch.isalnum())


def load_labels() -> pd.DataFrame:
    """
    Load bang label tu Excel. Cot trong Liang 2018 co dang:
      subject_ID | Sex | Age(year) | Height(cm) | Weight(kg) | BMI(kg/m^2)
      | Systolic Blood Pressure(mmHg) | Diastolic Blood Pressure(mmHg)
      | Heart Rate(b/m) | Hypertension | Diabetes | ...

    Ham nay tra ve DataFrame co cac cot chuan: subject_id, sbp, dbp, hr, age, sex, bmi.
    """
    path = _find_label_file()
    logger.info("Load labels tu %s", path)

    # Excel co the co header o row 0 hoac row 1; thu ca hai
    df = pd.read_excel(path, header=0, engine="openpyxl")
    # Neu cot dau tien khong phai subject ID (toan NaN), thu header=1
    if df.iloc[:, 0].isna().all() or "subject" not in _normalize_colname(str(df.columns[0])):
        df = pd.read_excel(path, header=1, engine="openpyxl")

    col_map: Dict[str, str] = {}
    for col in df.columns:
        norm = _normalize_colname(str(col))
        if "subject" in norm and "id" in norm:
            col_map[col] = "subject_id"
        elif "systolic" in norm:
            col_map[col] = "sbp"
        elif "diastolic" in norm:
            col_map[col] = "dbp"
        elif norm.startswith("heartrate") or (norm.startswith("hr") and "bm" in norm):
            col_map[col] = "hr"
        elif norm.startswith("age"):
            col_map[col] = "age"
        elif norm == "sex":
            col_map[col] = "sex"
        elif norm.startswith("bmi"):
            col_map[col] = "bmi"

    df = df.rename(columns=col_map)
    needed = {"subject_id", "sbp", "dbp"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(
            f"Excel thieu cot bat buoc: {missing}. Cot hien co: {list(df.columns)}"
        )

    # Loai hang thieu label BP
    df = df.dropna(subset=["sbp", "dbp"]).copy()
    df["subject_id"] = df["subject_id"].astype(int)
    df["sbp"] = df["sbp"].astype(float)
    df["dbp"] = df["dbp"].astype(float)
    logger.info("Loaded %d subjects co label BP", len(df))
    return df[["subject_id", "sbp", "dbp"] + [c for c in ("hr", "age", "sex", "bmi") if c in df.columns]]


# ============================================================
# Load signals
# ============================================================
def load_signal_file(path: Path) -> Optional[np.ndarray]:
    """
    Doc 1 file .txt chua PPG 1000 Hz, tab-separated, thuong 1 dong nhung ho tro nhieu dong.
    Tra ve np.ndarray hoac None neu loi / do dai bat hop le.
    """
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        logger.warning("Khong doc duoc %s: %s", path.name, exc)
        return None

    tokens = raw.replace("\n", "\t").replace(",", "\t").split("\t")
    values: List[float] = []
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        try:
            values.append(float(tok))
        except ValueError:
            continue
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < 1000:
        logger.debug("File %s qua ngan (%d mau)", path.name, len(arr))
        return None
    return arr


def resample_to_target(signal: np.ndarray, fs_in: int = FS_RAW,
                       fs_out: int = FS_TARGET) -> np.ndarray:
    """Resample PPG bang polyphase filter (chat luong tot hon scipy.signal.resample)."""
    from math import gcd
    g = gcd(fs_in, fs_out)
    up = fs_out // g
    down = fs_in // g
    return resample_poly(signal, up, down).astype(np.float64)


# ============================================================
# Feature extraction pipeline
# ============================================================
def load_features_and_labels(
    labels_df: pd.DataFrame,
    aggregate: str = "mean",
) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[int]]:
    """
    Duyet 657 file .txt, trich xuat 38 features, tra ve:
      X : (N, 38) feature matrix
      y : {"sbp": (N,), "dbp": (N,)}
      subject_ids : list cac subject_id tuong ung moi hang X

    aggregate:
      "mean"    — trung binh 3 recordings/subject (N = so subject co du file)
      "per_rec" — giu nguyen moi recording nhu 1 sample (N ≈ 657)
    """
    sbp_lookup = dict(zip(labels_df["subject_id"], labels_df["sbp"]))
    dbp_lookup = dict(zip(labels_df["subject_id"], labels_df["dbp"]))

    per_subject_feats: Dict[int, List[np.ndarray]] = {}
    kept = 0
    skipped = 0

    signal_files = sorted(SIGNAL_DIR.glob("*.txt"))
    logger.info("Scan %d file tin hieu trong %s", len(signal_files), SIGNAL_DIR)

    signal_dir_resolved = SIGNAL_DIR.resolve()
    for i, fpath in enumerate(signal_files, 1):
        # P1 fix: chan path traversal — glob() trong PathLib da an toan voi
        # thu muc truc tiep, nhung neu SIGNAL_DIR la symlink hoac co bug
        # trong Python, verify lai de chac chan file nam trong SIGNAL_DIR.
        try:
            fpath.resolve().relative_to(signal_dir_resolved)
        except ValueError:
            logger.warning("Path traversal attempt bi chan: %s", fpath)
            skipped += 1
            continue

        stem = fpath.stem  # vi du "100_1"
        try:
            subject_id = int(stem.split("_")[0])
        except ValueError:
            skipped += 1
            continue
        if subject_id not in sbp_lookup:
            skipped += 1
            continue

        signal_raw = load_signal_file(fpath)
        if signal_raw is None:
            skipped += 1
            continue

        signal_ds = resample_to_target(signal_raw, FS_RAW, FS_TARGET)
        # PPG-BP chi co 1 kenh; dung cung tin hieu cho IR va Red
        # (chap nhan duoc cho BP task — khong anh huong den feature IR-based)
        try:
            feats: PPGFeatures = extract_features(signal_ds, signal_ds, FS_TARGET)
        except Exception as exc:
            logger.debug("Loi extract %s: %s", stem, exc)
            skipped += 1
            continue

        flat = feats.to_flat_array()
        # P2 fix: guard feature-count — neu ai do doi NUM_*_FEATURES trong
        # ml_models.py, np.stack() duoi se crash voi shape mismatch kho debug.
        if flat.shape[0] != NUM_TOTAL_FEATURES:
            logger.warning(
                "Feature count mismatch cho %s: %d (ky vong %d) - skip.",
                stem, flat.shape[0], NUM_TOTAL_FEATURES,
            )
            skipped += 1
            continue
        if not np.all(np.isfinite(flat)):
            skipped += 1
            continue

        per_subject_feats.setdefault(subject_id, []).append(flat)
        kept += 1

        if i % 100 == 0:
            logger.info("  ...da xu ly %d/%d (kept=%d, skipped=%d)",
                        i, len(signal_files), kept, skipped)

    logger.info("Xu ly xong: %d recording ok, %d skipped", kept, skipped)

    X_rows: List[np.ndarray] = []
    y_sbp: List[float] = []
    y_dbp: List[float] = []
    subject_ids_out: List[int] = []

    for subject_id, feat_list in per_subject_feats.items():
        if aggregate == "mean":
            vec = np.mean(np.stack(feat_list, axis=0), axis=0)
            X_rows.append(vec)
            y_sbp.append(sbp_lookup[subject_id])
            y_dbp.append(dbp_lookup[subject_id])
            subject_ids_out.append(subject_id)
        else:  # per_rec
            for vec in feat_list:
                X_rows.append(vec)
                y_sbp.append(sbp_lookup[subject_id])
                y_dbp.append(dbp_lookup[subject_id])
                subject_ids_out.append(subject_id)

    X = np.stack(X_rows, axis=0) if X_rows else np.zeros((0, NUM_TOTAL_FEATURES))
    y = {"sbp": np.asarray(y_sbp, dtype=np.float64),
         "dbp": np.asarray(y_dbp, dtype=np.float64)}
    logger.info("Feature matrix: X.shape=%s (aggregate=%s)", X.shape, aggregate)
    return X, y, subject_ids_out


# ============================================================
# Training
# ============================================================
def build_model(name: str):
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=300, max_depth=12, min_samples_leaf=2,
            random_state=42, n_jobs=-1,
        )
    if name == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.9, random_state=42,
        )
    if name == "svr":
        return SVR(kernel="rbf", C=10.0, epsilon=1.5, gamma="scale")
    raise ValueError(f"Model khong ho tro: {name}")


def eval_cv(model_name: str, X: np.ndarray, y: np.ndarray,
            groups: Optional[List[int]] = None,
            n_splits: int = 5) -> Tuple[float, float, float]:
    """
    Subject-level 5-fold CV. Tra ve (MAE, RMSE, R^2).

    Scaler duoc nhung vao Pipeline -> fit rieng trong moi fold, tranh data leakage.
    Neu groups duoc truyen vao (moi phan tu la subject_id cua sample tuong ung),
    dung GroupKFold de dam bao cac recording cua cung 1 subject khong lan giua
    train/test fold. Neu groups la None hoac so subject < n_splits, fallback
    sang KFold thuong.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", build_model(model_name)),
    ])
    cv_kwargs = {}
    if groups is not None and len(set(groups)) >= n_splits:
        cv = GroupKFold(n_splits=n_splits)
        cv_kwargs["groups"] = np.asarray(groups)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_pred = cross_val_predict(pipe, X, y, cv=cv, n_jobs=-1, **cv_kwargs)
    mae = mean_absolute_error(y, y_pred)
    rmse = float(np.sqrt(np.mean((y_pred - y) ** 2)))
    r2 = r2_score(y, y_pred)
    return mae, rmse, r2


def fit_final(model_name: str, X: np.ndarray, y: np.ndarray):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = build_model(model_name)
    model.fit(X_scaled, y)
    return model, scaler


def save_bundle(model_type: str,
                models: Dict[str, object],
                scalers: Dict[str, object],
                metadata: Dict[str, object]) -> Path:
    """
    Lưu theo format ClassicalMLModel._try_load_models chấp nhận:
        {"models": {target: regressor}, "scalers": {target: scaler}, "metadata": {...}}

    Ghi atomic: .sha256 trước → rename temp→.pkl sau.
    Nếu bị kill giữa chừng: hoặc không có gì thay đổi (temp bị xóa),
    hoặc .sha256 mới + .pkl cũ (checksum mismatch → server từ chối load, an toàn).
    """
    out_dir = Path(MODEL_DIR)
    if not out_dir.is_absolute():
        out_dir = HERE / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_type}_models.pkl"
    sha_path = out_path.parent / f"{out_path.name}.sha256"

    payload = pickle.dumps({"models": models, "scalers": scalers, "metadata": metadata})
    digest = hashlib.sha256(payload).hexdigest()

    # Atomic write: NamedTemporaryFile tạo file với O_CREAT|O_EXCL (không TOCTOU).
    # Thứ tự: rename .pkl trước (atomic) → ghi .sha256 sau.
    # Fail-safe: nếu chết sau rename nhưng trước ghi .sha256 → .pkl mới không có
    # .sha256 → server log WARNING (non-strict) hoặc từ chối load (strict). An toàn.
    with tempfile.NamedTemporaryFile(dir=out_dir, suffix=".tmp", delete=False) as tf:
        tmp_pkl = Path(tf.name)
        tf.write(payload)
    try:
        tmp_pkl.replace(out_path)
        sha_path.write_text(digest + "\n", encoding="utf-8")
    except Exception:
        tmp_pkl.unlink(missing_ok=True)
        raise

    logger.info("Lưu model -> %s (sha256=%s...)", out_path, digest[:12])
    return out_path


# ============================================================
# Main
# ============================================================
def main() -> int:
    if not SIGNAL_DIR.exists():
        logger.error("Khong tim thay thu muc tin hieu: %s", SIGNAL_DIR)
        return 1

    logger.info("Step 1 — Load labels Excel")
    labels_df = load_labels()

    logger.info("Step 2 — Load signals + extract 38 features (1000 Hz -> 100 Hz)")
    X, y, subject_ids = load_features_and_labels(labels_df, aggregate="mean")
    if X.shape[0] < 20:
        logger.error("Qua it sample sau khi loc (%d). Dung.", X.shape[0])
        return 1

    logger.info("Step 3 — 5-fold CV cho 3 thuat toan x 2 target")
    baselines = {
        "sbp": {"Liang2018": 11.64, "Chowdhury2020": 3.02, "ElHajj2021": 5.77},
        "dbp": {"Liang2018": 7.62,  "Chowdhury2020": 1.74, "ElHajj2021": 3.33},
    }
    candidates = ["random_forest", "gradient_boosting", "svr"]
    results: Dict[str, Dict[str, Tuple[float, float, float]]] = {t: {} for t in BP_TARGETS}
    best_per_target: Dict[str, str] = {}

    for target in BP_TARGETS:
        print(f"\n=== Target: {target.upper()} ===")
        best_mae = float("inf")
        best_name = ""
        for name in candidates:
            mae, rmse, r2 = eval_cv(name, X, y[target], groups=subject_ids)
            results[target][name] = (mae, rmse, r2)
            print(f"  {name:<20s} MAE={mae:6.2f}  RMSE={rmse:6.2f}  R2={r2:+.3f}")
            if mae < best_mae:
                best_mae = mae
                best_name = name
        best_per_target[target] = best_name
        print(f"  -> best: {best_name} (MAE={best_mae:.2f} mmHg)")
        print(f"  so voi baseline {target.upper()} (mmHg):")
        for ref, ref_mae in baselines[target].items():
            mark = "v" if best_mae <= ref_mae else "x"
            print(f"    [{mark}] {ref:<16s} MAE={ref_mae:.2f} (ours: {best_mae:.2f})")

    logger.info("Step 4 — Refit tren toan bo data voi model tot nhat + save .pkl")
    # Refit tung target voi model tot nhat.
    fitted_models: Dict[str, object] = {}
    fitted_scalers: Dict[str, object] = {}
    for target in BP_TARGETS:
        name = best_per_target[target]
        model, scaler = fit_final(name, X, y[target])
        fitted_models[target] = model
        fitted_scalers[target] = scaler

    base_metadata = {
        "dataset": "PPG-BP (Liang et al. 2018)",
        "num_subjects": int(X.shape[0]),
        "feature_count": int(X.shape[1]),
        "fs_target_hz": FS_TARGET,
        "fs_raw_hz": FS_RAW,
        "aggregation": "mean_of_3_recordings_per_subject",
        "targets": list(BP_TARGETS),
        "best_per_target": best_per_target,
        "cv_results": {
            t: {name: {"mae": r[0], "rmse": r[1], "r2": r[2]}
                for name, r in results[t].items()}
            for t in BP_TARGETS
        },
        "baselines": baselines,
    }

    saved_paths: List[Path] = []
    # Neu SBP va DBP chung 1 algorithm -> luu 1 bundle voi tag algorithm do,
    # chua ca hai target. Neu khac -> luu 2 bundle rieng, moi bundle chi chua
    # 1 target de ClassicalMLModel load dung model theo MODEL_TYPE env.
    if best_per_target["sbp"] == best_per_target["dbp"]:
        chosen = best_per_target["sbp"]
        metadata = {**base_metadata, "bundle_targets": list(BP_TARGETS)}
        saved_paths.append(
            save_bundle(chosen, fitted_models, fitted_scalers, metadata)
        )
        logger.info("SBP va DBP cung chon '%s' -> luu 1 bundle.", chosen)
    else:
        logger.warning(
            "SBP chon '%s' nhung DBP chon '%s' -> luu 2 bundle rieng.",
            best_per_target["sbp"], best_per_target["dbp"],
        )
        for target in BP_TARGETS:
            name = best_per_target[target]
            metadata = {**base_metadata, "bundle_targets": [target]}
            saved_paths.append(
                save_bundle(
                    name,
                    {target: fitted_models[target]},
                    {target: fitted_scalers[target]},
                    metadata,
                )
            )

    print("\nTOM TAT")
    print(f"  - N subjects: {X.shape[0]}")
    print(f"  - Features:   {X.shape[1]}")
    for target in BP_TARGETS:
        name = best_per_target[target]
        mae, rmse, r2 = results[target][name]
        print(f"  - {target.upper()}: {name}  MAE={mae:.2f} mmHg  R2={r2:+.3f}")
    for p in saved_paths:
        print(f"  - Saved -> {p}")
    print("  - ClassicalMLModel se tu load file nay khi server khoi dong.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
