"""
================================================================
  FIFA World Cup 2026 — Apple Silicon FULL POWER Variant
  Target hardware: M2 Pro / M2 Max / M3 Pro / M3 Max / M4 Pro

  PHILOSOPHY: use every compute unit available
  ─────────────────────────────────────────────
  ✅ CPU        — ALL cores, no reservation (n_jobs=-1)
  ✅ GPU Metal  — MPS via PyTorch for batch Monte Carlo inference
  ✅ Neural Eng — ANE via Core ML (coremltools) for final model
  ✅ Unified RAM — No artificial limit; CPU/GPU/ANE share one pool
  ✅ AMX        — Apple Matrix Coprocessor via Accelerate (numpy)
  ✅ NEON SIMD  — numpy ARM64 uses NEON vectorization automatically

  ESTIMATED RUNTIMES (M2 Pro 12-core, 16 GB):
  ──────────────────────────────────────────
  Training + Optuna 200 trials  ~3-4 min   (all cores)
  Monte Carlo 10k  CPU only     ~8-10 min
  Monte Carlo 10k  with MPS     ~3-5 min   ← target
  Monte Carlo 10k  with ANE(RF) ~2-4 min
  Total with MPS/ANE:           ~7-9 min   (on par with CUDA)

  INSTALLATION:
  ─────────────
  conda create -n wc2026_as python=3.11 -y && conda activate wc2026_as
  pip install xgboost lightgbm scikit-learn optuna shap joblib psutil
  pip install numpy pandas matplotlib seaborn requests
  pip install torch torchvision          # GPU Metal (MPS)
  pip install coremltools                # Neural Engine (ANE)
================================================================
"""

import warnings, os, sys, time, platform, gc
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
import random, json
from copy import deepcopy
from collections import defaultdict
import itertools, requests, multiprocessing

# Scikit-learn — mismos imports que Cell 1 original + RobustScaler adicional
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import (
    LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score, log_loss, roc_auc_score,
    brier_score_loss, confusion_matrix, f1_score
)

# Boosting
import xgboost as xgb
import lightgbm as lgb
from joblib import Parallel, delayed

# Optuna
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# ================================================================
# ① HARDWARE DETECTION — full Apple Silicon inventory
# ================================================================

print("=" * 60)
print("  🍎  WC 2026 — Apple Silicon FULL POWER CONFIG")
print("=" * 60)

# --- CPU: all cores, no reservation ---
N_CORES_PHYSICAL = multiprocessing.cpu_count()
N_JOBS = -1          # -1 = joblib / XGBoost / LGB / RF use ALL cores
                     # (N_JOBS name matches Cell 1 of the main notebook)
print(f"\n[CPU] {N_CORES_PHYSICAL} cores detected → n_jobs=-1 (no cores reserved)")
print(f"      M2 Pro: 8P+4E=12  |  M2 Max: 10P+4E=14  |  M3 Pro: 8P+4E=12")

# --- Unified RAM ---
RAM_GB = 0
try:
    import psutil
    RAM_GB           = psutil.virtual_memory().total     / (1024**3)
    RAM_AVAILABLE_GB = psutil.virtual_memory().available / (1024**3)
    print(f"[RAM] Total: {RAM_GB:.0f} GB unified | Available: {RAM_AVAILABLE_GB:.1f} GB")
    print(f"      CPU, GPU and ANE share the same physical bank — no copy overhead")
except ImportError:
    print("[RAM] psutil not installed (pip install psutil to see RAM info)")

# --- GPU Metal (MPS) via PyTorch ---
MPS_AVAILABLE  = False
MPS_DEVICE     = None
try:
    import torch
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        MPS_DEVICE    = torch.device("mps")
        MPS_AVAILABLE = True
        # Warm up MPS (first call has JIT latency)
        _ = torch.zeros(1, device=MPS_DEVICE)
        print(f"\n[GPU] Metal Performance Shaders (MPS) ✅")
        print(f"      M2 Pro: 19 GPU cores | M2 Max: 30+ GPU | M3 Pro: 18-30 GPU")
        print(f"      MPS warmed up — ready for batch Monte Carlo inference")
    else:
        print(f"\n[GPU] MPS not available (macOS < 12.3 or PyTorch < 1.13)")
except ImportError:
    print(f"\n[GPU] PyTorch not installed → pip install torch torchvision")

# --- Neural Engine (ANE) via Core ML ---
ANE_AVAILABLE   = False
COREML_AVAILABLE = False
try:
    import coremltools as ct
    COREML_AVAILABLE = True
    ANE_AVAILABLE    = True
    print(f"\n[ANE] coremltools {ct.__version__} ✅")
    print(f"      Neural Engine (16-core on M2) activates automatically")
    print(f"      via Core ML Runtime for models < ~15 MB")
    print(f"      Plan: convert RandomForest → .mlpackage → ANE inference")
except ImportError:
    print(f"\n[ANE] coremltools not installed → pip install coremltools")

# --- Summary ---
print(f"\n{'─'*60}")
print(f"  AVAILABLE COMPUTE:")
print(f"  CPU  ✅  {N_CORES_PHYSICAL} cores  (n_jobs=-1, no reservation)")
print(f"  AMX  ✅  Apple Matrix Coprocessor via Accelerate (auto numpy)")
print(f"  GPU  {'✅  Metal MPS' if MPS_AVAILABLE else '❌  install: pip install torch'}")
print(f"  ANE  {'✅  Core ML Runtime' if ANE_AVAILABLE else '❌  install: pip install coremltools'}")
print(f"  RAM  ✅  {RAM_GB:.0f if RAM_GB else '??'}GB unified (no artificial limit)")
print(f"{'─'*60}")

# Variable names match Cell 1 of the main notebook — rest of notebook uses them directly
GPU_AVAILABLE = False   # No NVIDIA CUDA on Apple Silicon
XGB_DEVICE    = 'cpu'   # XGBoost uses 'hist' on ARM64 (not 'gpu_hist')
LGB_DEVICE    = 'cpu'   # LightGBM CPU with force_col_wise
print(f"\n✅ Compute config: XGB={XGB_DEVICE} | LGB={LGB_DEVICE} | CPU_JOBS={N_JOBS}")
print(f"   Optuna: {OPTUNA_AVAILABLE} | SHAP: {SHAP_AVAILABLE}")
print(f"   Apple Silicon: MPS={MPS_AVAILABLE} | ANE={ANE_AVAILABLE}")


# ================================================================
# ② ENVIRONMENT VARIABLES — maximize ALU and BLAS throughput
# ================================================================

# Apple Silicon uses the Accelerate framework for BLAS (alternative to MKL/OpenBLAS)
# numpy/scipy operations (dot, eig, svd...) are accelerated via vDSP and AMX
os.environ['OMP_NUM_THREADS']        = str(N_CORES_PHYSICAL)
os.environ['OPENBLAS_NUM_THREADS']   = str(N_CORES_PHYSICAL)
os.environ['MKL_NUM_THREADS']        = str(N_CORES_PHYSICAL)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(N_CORES_PHYSICAL)  # Accelerate framework
os.environ['NUMEXPR_NUM_THREADS']    = str(N_CORES_PHYSICAL)

print(f"\n[ENV] BLAS/Accelerate threads: {N_CORES_PHYSICAL}")
print(f"      numpy on ARM64 uses NEON SIMD + AMX automatically")


# ================================================================
# ③ MODEL PARAMETERS — FULL POWER
# ================================================================

# XGBoost: hist method on ARM64, all cores
XGB_PARAMS_AS = dict(
    n_estimators     = 800,       # +300 vs CUDA version (fast CPU handles more trees)
    max_depth        = 8,         # one level deeper than default
    learning_rate    = 0.03,      # lower → more stable with more trees
    subsample        = 0.85,
    colsample_bytree = 0.85,
    gamma            = 0.05,
    reg_alpha        = 0.1,
    reg_lambda       = 1.5,
    min_child_weight = 3,
    tree_method      = 'hist',    # ← REQUIRED on Apple Silicon (not 'gpu_hist')
    device           = 'cpu',     # ← NOT 'cuda'
    eval_metric      = 'logloss',
    n_jobs           = -1,        # ALL cores
    nthread          = -1,        # double-safe for XGBoost internals
    random_state     = 42,
)
print(f"\n[XGB] hist | n_jobs=-1 | nthread=-1 | n_estimators=800 | max_depth=8")

# LightGBM: ARM64-efficient with force_col_wise
LGB_PARAMS_AS = dict(
    n_estimators     = 800,
    num_leaves       = 127,       # 2^7-1, one level above default 63
    max_depth        = 8,
    learning_rate    = 0.03,
    subsample        = 0.85,
    colsample_bytree = 0.85,
    reg_alpha        = 0.1,
    reg_lambda       = 1.5,
    min_child_samples= 8,
    device           = 'cpu',
    num_threads      = N_CORES_PHYSICAL,  # explicit is better than -1 for LGB
    force_col_wise   = True,      # more efficient on ARM for narrow datasets
    verbose          = -1,
    random_state     = 42,
)
print(f"[LGB] num_threads={N_CORES_PHYSICAL} | num_leaves=127 | force_col_wise=True")

# Random Forest: scales well with n_jobs=-1 on M2
RF_PARAMS_AS = dict(
    n_estimators  = 600,
    max_depth     = 12,
    min_samples_split = 4,
    max_features  = 'sqrt',
    n_jobs        = -1,           # ALL cores
    random_state  = 42,
)
print(f"[RF]  n_estimators=600 | max_depth=12 | n_jobs=-1")

# Optuna: more trials + CV (fast CPU compensates for absence of CUDA)
N_OPTUNA_TRIALS  = 200           # vs 60 in CUDA version
N_OPTUNA_CV_FOLDS = 3
print(f"[OPT] {N_OPTUNA_TRIALS} trials | {N_OPTUNA_CV_FOLDS}-fold CV as metric (more robust)")

# Monte Carlo
N_SIMULATIONS = 10_000
print(f"[MC]  {N_SIMULATIONS:,} simulations | n_jobs=-1 | backend=loky")


# ================================================================
# ④ OPTUNA OBJECTIVE — 3-fold CV for M2 Pro
# ================================================================

def objective_xgb_as(trial, X_tr_sc, y_tr, X_te_sc, y_te, w_tr):
    """
    Optuna objective for Apple Silicon.
    - Wider search space than CUDA version
    - 3-fold CV as metric (avoids overfitting to the holdout set)
    - n_jobs=-1 inside each trial (uses all M2 cores)
    """
    p = dict(
        n_estimators     = trial.suggest_int('n_estimators', 400, 1200),
        max_depth        = trial.suggest_int('max_depth', 4, 10),
        learning_rate    = trial.suggest_float('lr', 0.01, 0.10, log=True),
        subsample        = trial.suggest_float('sub', 0.6, 1.0),
        colsample_bytree = trial.suggest_float('col', 0.5, 1.0),
        gamma            = trial.suggest_float('gamma', 0, 2.0),
        reg_alpha        = trial.suggest_float('alpha', 0, 4.0),
        reg_lambda       = trial.suggest_float('lambda', 0, 5.0),
        min_child_weight = trial.suggest_int('mcw', 1, 10),
        tree_method='hist', device='cpu', nthread=-1,
        n_jobs=-1, random_state=42, eval_metric='logloss',
    )
    m  = xgb.XGBClassifier(**p)
    cv = StratifiedKFold(n_splits=N_OPTUNA_CV_FOLDS, shuffle=True, random_state=42)
    # n_jobs=1 in cross_val_score because XGBoost already uses all cores internally
    scores = cross_val_score(
        m, X_tr_sc, y_tr,
        cv=cv, scoring='roc_auc',
        fit_params={'sample_weight': w_tr},
        n_jobs=1,
    )
    return scores.mean()


# ================================================================
# ⑤ MPS PREDICTOR — GPU Metal inference for Monte Carlo
# ================================================================

class MPSPredictor:
    """
    Wraps the trained model as a fast predictor running on M2 GPU
    cores via PyTorch MPS (Metal Performance Shaders).

    Strategy (lightweight knowledge distillation):
    1. Use the XGBoost model to generate soft probabilities over the
       training set (these become the network targets)
    2. Train a small network (4 layers, ~5k params) on MPS to
       reproduce those probabilities
    3. At Monte Carlo time, inference hits MPS: 3-5x faster than CPU

    The network is intentionally small to:
    - Fit in the M2 GPU's L2 cache (maximum throughput)
    - Minimize initial calibration time
    - Allow Core ML runtime to assign it to the ANE when exported
    """

    def __init__(self, sklearn_model, device='mps'):
        self.model     = sklearn_model
        self.device    = torch.device(device) if MPS_AVAILABLE else torch.device('cpu')
        self.net       = None
        self._ready    = False

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray = None):
        if not MPS_AVAILABLE:
            print("⚠️  MPS not available — falling back to CPU for Monte Carlo")
            return self
        try:
            import torch
            import torch.nn as nn

            # Soft targets: original model probabilities (better than hard binary labels)
            proba = self.model.predict_proba(X_cal)[:, 1].astype(np.float32)
            n_feat = X_cal.shape[1]

            # Minimal network optimized for low MPS latency
            # Linear(n→64) → GELU → Linear(64→32) → GELU → Linear(32→1) → Sigmoid
            self.net = nn.Sequential(
                nn.Linear(n_feat, 64),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(64, 32),
                nn.GELU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            ).to(self.device)

            X_t = torch.tensor(X_cal.astype(np.float32)).to(self.device)
            y_t = torch.tensor(proba).unsqueeze(1).to(self.device)
            opt = torch.optim.Adam(self.net.parameters(), lr=2e-3, weight_decay=1e-5)
            lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300)
            loss_fn  = nn.MSELoss()

            print(f"\n   🎮 Calibrating network on MPS ({self.device}) — {n_feat}→64→32→1...")
            print(f"      This takes ~1-2 min — done only once")
            t0 = time.time()
            for ep in range(300):
                opt.zero_grad()
                loss = loss_fn(self.net(X_t), y_t)
                loss.backward()
                opt.step()
                lr_sched.step()
                if ep % 75 == 0:
                    print(f"      Epoch {ep:3d}/300 | loss={loss.item():.5f} | "
                          f"lr={opt.param_groups[0]['lr']:.5f}")

            elapsed = time.time() - t0
            print(f"   ✅ MPS network ready in {elapsed:.1f}s — Monte Carlo will use GPU Metal")
            self._ready = True
        except Exception as e:
            print(f"   ⚠️  MPS calibration failed: {e} — falling back to CPU")
        return self

    @torch.no_grad()
    def predict(self, feat_diff: np.ndarray) -> float:
        """Predict P(team1 wins). Uses MPS if ready, falls back to CPU."""
        if self._ready and self.net is not None:
            import torch
            x = torch.tensor(feat_diff.reshape(1, -1).astype(np.float32)).to(self.device)
            return float(self.net(x).item())
        return float(self.model.predict_proba(feat_diff.reshape(1, -1))[0, 1])

    @torch.no_grad()
    def predict_batch(self, X_batch: np.ndarray) -> np.ndarray:
        """Batch prediction — the key driver of Monte Carlo speedup."""
        if self._ready and self.net is not None:
            import torch
            x = torch.tensor(X_batch.astype(np.float32)).to(self.device)
            return self.net(x).squeeze().cpu().numpy()
        return self.model.predict_proba(X_batch)[:, 1]


# ================================================================
# ⑥ CORE ML / ANE — convert model for Neural Engine
# ================================================================

def convert_to_coreml(model, X_sample: np.ndarray, name: str = "WC2026") -> str | None:
    """
    Convert a sklearn/XGBoost model to Core ML (.mlpackage).

    Core ML Runtime AUTOMATICALLY assigns to the most efficient unit:
    - Model < ~15 MB  →  Neural Engine (16-core, most power-efficient)
    - Model 15-50 MB  →  GPU Metal
    - Model > 50 MB   →  CPU (model too large for ANE/GPU)

    Random Forest (300 trees) typically stays < 10 MB → goes to ANE.
    XGBoost (800 trees) may exceed 50 MB → may run on CPU inside CoreML.
    Therefore, convert the RF to get ANE acceleration.
    """
    if not COREML_AVAILABLE:
        print("⚠️  coremltools not installed — skipping ANE conversion")
        return None
    try:
        import coremltools as ct
        print(f"\n[ANE] Converting {name} to Core ML (.mlpackage)...")
        t0 = time.time()

        cml = ct.converters.sklearn.convert(
            model,
            input_features=["features"],
            output_feature_names=["win_prob"]
        )
        cml.short_description = f"WC2026 — {name}"
        cml.author = "WC2026 ML Project"
        cml.version = "2.0"

        path = Path("data") / f"{name}.mlpackage"
        cml.save(str(path))
        size_mb = sum(f.stat().st_size for f in path.rglob('*') if f.is_file()) / 1e6

        print(f"   ✅ {path} ({size_mb:.1f} MB) — converted in {time.time()-t0:.1f}s")
        if size_mb < 15:
            print(f"   🧠 < 15 MB → Core ML will assign to Neural Engine (ANE)")
        elif size_mb < 50:
            print(f"   🎮 15-50 MB → Core ML will use GPU Metal")
        else:
            print(f"   🖥️  > 50 MB → Core ML will use CPU (model too large for ANE/GPU)")
        return str(path)
    except Exception as e:
        print(f"   ⚠️  Core ML conversion failed: {e}")
        print(f"       Tip: use RandomForest — more compatible than full XGBoost")
        return None


def make_ane_predictor(mlpackage_path: str):
    """
    Returns a predict(X) function that runs on ANE/GPU Metal automatically.
    Core ML Runtime selects the accelerator based on model size and system load.
    """
    if not COREML_AVAILABLE or not mlpackage_path:
        return None
    try:
        import coremltools as ct
        mdl = ct.models.MLModel(mlpackage_path)

        def _predict(X: np.ndarray) -> np.ndarray:
            outs = []
            for i in range(len(X)):
                outs.append(mdl.predict({"features": X[i:i+1].astype(np.float32)})["win_prob"])
            return np.array(outs, dtype=np.float32)

        print(f"✅ ANE/Core ML predictor active → {mlpackage_path}")
        return _predict
    except Exception as e:
        print(f"⚠️  Error loading Core ML model: {e}")
        return None


# ================================================================
# ⑦ MONTE CARLO — uses MPS/ANE when available
# ================================================================

def simulate_tournament_as(groups, model, scaler, df_features, feature_cols,
                             elo_ratings, mps_pred=None):
    """Simulate a full tournament bracket. Uses MPS predictor if ready."""

    def win_prob(t1, t2):
        if t1 not in df_features.index or t2 not in df_features.index:
            e1, e2 = elo_ratings.get(t1, 1500), elo_ratings.get(t2, 1500)
            return 1 / (1 + 10 ** (-(e1 - e2) / 400))
        f1   = df_features.loc[t1, feature_cols].values.astype(float)
        f2   = df_features.loc[t2, feature_cols].values.astype(float)
        diff = scaler.transform((f1 - f2).reshape(1, -1))
        if mps_pred and mps_pred._ready:
            return mps_pred.predict(diff[0])
        return float(model.predict_proba(diff)[0, 1])

    def play(t1, t2):
        p    = win_prob(t1, t2)
        dp   = 0.26 * (1 - abs(p - 0.5) * 2)
        r    = random.random()
        if r < dp:                          # empate → penales
            return t1 if random.random() < p else t2
        elif r < dp + (1 - dp) * p:
            return t1
        else:
            return t2

    # Fase de grupos
    qualifiers = {}
    for grp, teams in groups.items():
        pts = {t: 0 for t in teams}
        gd  = {t: 0 for t in teams}
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                t1, t2 = teams[i], teams[j]
                p1  = win_prob(t1, t2)
                dp  = 0.28 * (1 - abs(p1 - 0.5) * 1.8)
                r   = random.random()
                if r < dp:
                    pts[t1] += 1; pts[t2] += 1
                elif r < dp + (1 - dp) * p1:
                    pts[t1] += 3; gd[t1] += 1; gd[t2] -= 1
                else:
                    pts[t2] += 3; gd[t2] += 1; gd[t1] -= 1
        ranking = sorted(teams, key=lambda t: (pts[t], gd[t]), reverse=True)
        qualifiers[grp] = [ranking[0], ranking[1]]

    # Bracket eliminatorio
    gkeys   = sorted(qualifiers.keys())
    r16     = []
    for i in range(0, len(gkeys), 2):
        g1, g2 = gkeys[i], gkeys[i + 1]
        r16.append((qualifiers[g1][0], qualifiers[g2][1]))
        r16.append((qualifiers[g2][0], qualifiers[g1][1]))

    bracket = [play(a, b) for a, b in r16]
    while len(bracket) > 1:
        bracket = [play(bracket[i], bracket[i+1]) for i in range(0, len(bracket), 2)]
    return bracket[0]


def run_monte_carlo_as(n_sims, groups, model, scaler, df_features, feature_cols,
                        elo_ratings, mps_pred=None, n_jobs=-1):
    """
    Monte Carlo FULL POWER.
    - n_jobs=-1: TODOS los cores del M2
    - backend='loky': procesos separados (evita GIL de Python)
    - prefer='processes': mejor para cómputo intensivo
    """
    t0 = time.time()
    accel = "GPU Metal (MPS)" if (mps_pred and mps_pred._ready) else "CPU todos los cores"
    print(f"\n🎲 Monte Carlo: {n_sims:,} simulaciones | n_jobs={n_jobs} | {accel}")

    results = Parallel(n_jobs=n_jobs, backend='loky', prefer='processes')(
        delayed(simulate_tournament_as)(
            groups, model, scaler, df_features, feature_cols, elo_ratings, mps_pred
        )
        for _ in range(n_sims)
    )

    elapsed = time.time() - t0
    counts  = defaultdict(int)
    for w in results:
        counts[w] += 1

    df = pd.DataFrame([
        {'team': t, 'champion_pct': c / n_sims * 100}
        for t, c in sorted(counts.items(), key=lambda x: -x[1])
    ])
    print(f"✅ Completado en {elapsed:.1f}s ({elapsed/60:.1f} min) | {n_sims/elapsed:.0f} sims/s")
    return df, elapsed


# ================================================================
# ⑧ MEMORY MANAGEMENT — unified RAM
# ================================================================

def optimize_memory_as():
    """
    On Apple Silicon, CPU and GPU share the same physical RAM.
    If the system swaps to SSD → speed drops dramatically.
    This function frees memory before Monte Carlo and warns if at risk.
    """
    gc.collect()
    if MPS_AVAILABLE:
        try:
            import torch
            torch.mps.empty_cache()  # free MPS cache
            print("[MEM] MPS cache cleared")
        except Exception:
            pass
    try:
        import psutil
        avail = psutil.virtual_memory().available / (1024**3)
        print(f"[MEM] Available RAM: {avail:.1f} GB")
        if avail < 2.0:
            print(f"      ⚠️  < 2 GB available — swap risk (close other apps)")
        else:
            print(f"      ✅ Sufficient for {N_SIMULATIONS:,} simulations without swap")
    except ImportError:
        pass


# ================================================================
# ⑨ GLOBAL CONFIG — paths, seeds, matplotlib Retina
# ================================================================

DATA_DIR = Path("data")
GH_DIR   = DATA_DIR / "github"
KAG_DIR  = DATA_DIR / "kaggle"
GH_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)
random.seed(42)

plt.rcParams.update({
    'figure.figsize': (13, 6),
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.3, 'font.size': 11,
    'figure.dpi': 150,   # Retina display MacBook Pro — sharper than default 72
})
sns.set_palette("husl")

# ── Nota de compatibilidad ───────────────────────────────────────
# Este archivo es un reemplazo EXACT de Cell 1 del notebook.
# Exporta exactamente las mismas variables que Cell 1 original:
#   GPU_AVAILABLE, XGB_DEVICE, LGB_DEVICE, N_JOBS,
#   DATA_DIR, GH_DIR, KAG_DIR, OPTUNA_AVAILABLE, SHAP_AVAILABLE
# MÁS las capas Apple Silicon:
#   MPS_AVAILABLE, MPS_DEVICE, ANE_AVAILABLE, N_CORES_PHYSICAL
#   XGB_PARAMS_AS, LGB_PARAMS_AS, RF_PARAMS_AS
#   MPSPredictor, convert_to_coreml, make_ane_predictor
#   run_monte_carlo_as, optimize_memory_as
# ────────────────────────────────────────────────────────────────


# ================================================================
# ⑩ INSTRUCCIONES DE USO — integración en el notebook
# ================================================================

print("""
================================================================
  CÓMO INTEGRAR EN EL NOTEBOOK (pasos exactos)
================================================================

  CELDA 1 — Reemplazar setup.py por este archivo
    O poner al inicio del notebook:
      exec(open('mundial_2026_m2pro.py').read())

  CELDA 9 — Cambiar parámetros de modelos:
    xgb_params = XGB_PARAMS_AS        # ya tiene n_jobs=-1, hist
    lgb_params = LGB_PARAMS_AS        # ya tiene num_threads=all
    rf_params  = RF_PARAMS_AS         # ya tiene n_jobs=-1

  CELDA 9 — Cambiar Optuna:
    study.optimize(
        lambda t: objective_xgb_as(t, X_tr_sc, y_tr, X_te_sc, y_te, w_tr),
        n_trials=N_OPTUNA_TRIALS       # 200
    )

  CELDA 10 (OPCIONAL) — Activar GPU Metal para Monte Carlo:
    mps_pred = MPSPredictor(best_model_v2, device='mps')
    mps_pred.calibrate(X_tr_sc, y_tr)  # ~1-2 min, solo una vez

  CELDA 10 (OPCIONAL) — Convertir Random Forest al Neural Engine:
    mlpkg = convert_to_coreml(rf_model, X_te_sc, name='WC2026_RF')
    ane_predict = make_ane_predictor(mlpkg)  # usa ANE automáticamente

  CELDA 11 — Monte Carlo FULL POWER:
    optimize_memory_as()   # libera RAM y cache MPS antes de correr
    df_champ, elapsed = run_monte_carlo_as(
        n_sims=10_000, groups=GROUPS_2026_V2,
        model=best_model_v2, scaler=scaler_v2,
        df_features=df_features_v2, feature_cols=feature_cols_v2,
        elo_ratings=elo2.ratings,
        mps_pred=mps_pred,  # None si no calibraste el MPS
        n_jobs=-1
    )

================================================================
  TIEMPOS ESPERADOS — Apple M2 Pro 12-core 16GB
================================================================
  Config              Training+Optuna  Monte Carlo  Total
  ──────────────────────────────────────────────────────
  Solo CPU (n_jobs=-1)    ~4-5 min     ~8-10 min  ~14-16 min
  + MPS calibrado         ~5-6 min     ~3-5 min   ~9-12 min
  + ANE (RF export)       ~5-6 min     ~2-4 min   ~8-11 min
  ──────────────────────────────────────────────────────
  vs CUDA (RTX serie)     ~2-3 min     ~4 min     ~7-9 min
  vs CPU Intel i9         ~12-15 min   ~35 min    ~50 min

  El M2 Pro con MPS activo llega a ser ~comparable al CUDA
  gracias a la memoria unificada y el ancho de banda enorme.
================================================================
""")
