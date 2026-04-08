# FIFA World Cup 2026 Predictor

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-orange?logo=xgboost)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6.0-green)
![Optuna](https://img.shields.io/badge/Optuna-4.7.0-blueviolet)
![AUC](https://img.shields.io/badge/Validation_AUC-0.7613-brightgreen)
![Simulations](https://img.shields.io/badge/Monte_Carlo-10%2C000_sims-blue)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **Machine learning ensemble model that predicts the FIFA World Cup 2026 champion.  
> Trained on 5,882 real historical matches. Validated at AUC = 0.7613 — professional grade for football prediction.**

---

## Final Predictions — Champion Probabilities

| # | Team | Probability | Confederation |
|---|------|------------|---------------|
| 🥇 | **Spain** | **21.98%** | UEFA |
| 🥈 | **Argentina** | **21.47%** | CONMEBOL |
| 🥉 | **England** | **15.10%** | UEFA |
| 4 | France | 7.76% | UEFA |
| 5 | Brazil | 6.43% | CONMEBOL |
| 6 | Portugal | 4.81% | UEFA |
| 7 | Netherlands | 3.92% | UEFA |
| 8 | Germany | 3.47% | UEFA |
| 9 | Belgium | 2.89% | UEFA |
| 10 | Colombia | 2.31% | CONMEBOL |
| 11 | Morocco | 1.98% | CAF |
| 12 | USA | 1.74% | CONCACAF |
| 13 | Japan | 1.52% | AFC |
| 14 | Uruguay | 1.21% | CONMEBOL |
| 15 | Croatia | 0.98% | UEFA |

> **Key insight:** Spain's draw (Group H: Cape Verde, Saudi Arabia, Uruguay) combined with Brazil's draw (Group C) gives Spain a manageable path to the final. England's 15% (+11.5pp vs provisional draw) is the biggest draw beneficiary — Group L is highly accessible.

---

## Model Architecture

```
 Raw Data (5,882 matches, 1930–2026)
         │
         ▼
 ┌─────────────────────┐
 │  Data Pipeline      │  ← Fjelstul GitHub (27 CSVs) + Kaggle + 64 injected recent matches
 └────────┬────────────┘
          │
          ▼
 ┌─────────────────────┐
 │  Dynamic ELO System │  ← Temporal decay (λ): recent matches carry more weight
 └────────┬────────────┘     4 curve variants, cross-confederation calibration
          │
          ▼
 ┌─────────────────────┐
 │  Feature Engineering│  ← 30+ features: elo_diff, rank_diff, squad_value_diff,
 └────────┬────────────┘     form_diff, discipline, is_host, goals_conceded_avg...
          │
          ▼
 ┌─────────────────────────────────────────────┐
 │         Voting Ensemble                      │
 │  ┌───────────┐ ┌──────────┐ ┌────────────┐  │
 │  │ XGBoost   │ │LightGBM  │ │RandomForest│  │  ← 200-trial Optuna + 3-fold CV
 │  │ (GPU/CPU) │ │          │ │            │  │
 │  └───────────┘ └──────────┘ └────────────┘  │
 └────────┬────────────────────────────────────┘
          │
          ▼
 ┌─────────────────────┐
 │  Stacking Ensemble  │  ← Logistic Regression meta-learner + Isotonic calibration
 └────────┬────────────┘
          │
          ▼
 ┌─────────────────────┐
 │  Monte Carlo (10k)  │  ← Simulates full 48-team bracket 10,000 times
 └────────┬────────────┘     joblib.Parallel(n_jobs=-1) — ~4 min on GPU, ~9 min on Apple Silicon
          │
          ▼
    Champion Probabilities
```

---

## Key Metrics

| Metric | Value |
|--------|-------|
| Validation AUC | **0.7613** |
| Training matches | 5,882 |
| Features | 30+ |
| Ensemble models | 3 (XGBoost + LGB + RF) |
| Optuna trials | 200 |
| Monte Carlo simulations | 10,000 |
| Runtime (NVIDIA GPU) | ~4 min |
| Runtime (Apple Silicon M2 Pro) | ~7–9 min |
| Football prediction ceiling | ~0.81 (irreducible noise) |

---

## Top Features (SHAP Analysis)

| Rank | Feature | Description |
|------|---------|-------------|
| 1 | `elo_diff` | ELO rating difference — strongest signal by a wide margin |
| 2 | `rank_diff` | FIFA ranking difference |
| 3 | `squad_value_diff` | Squad market value difference (€M) |
| 4 | `form_diff` | Recent win rate difference |
| 5 | `is_host` | Whether a team is a host nation |

> **Surprising finding:** `recent_goals_conceded_avg` outranked `recent_goals_scored_avg`.  
> *Defending well predicts champions more reliably than attacking well.*

---

## Repository Structure

```
WorldCupModel/
├── README.md
├── requirements.txt
├── .gitignore
│
├── mundial_2026_v2_advanced.ipynb   ← Main notebook: full pipeline (run top to bottom)
│
├── src/
│   ├── fix_groups_official.py       ← Inject official FIFA 2026 draw into notebook
│   └── mundial_2026_m2pro.py        ← Apple Silicon variant (MPS + Core ML / ANE)
│
├── data/
│   ├── teams_2026.csv               ← 48-team dataset (ELO, FIFA rank, squad value...)
│   ├── github/                      ← Fjelstul WC database (auto-downloaded by notebook)
│   └── kaggle/                      ← Kaggle WC historical dataset
│
├── results/
│   ├── resultados_mundial_2026_v2.csv  ← Champion probability table
│   ├── shap_summary_v2.png             ← SHAP feature importance plot
│   ├── feature_importance_v2.png       ← Tree-based feature importance
│   ├── mundial_2026_v2_predictions.png ← Top-15 champion probability chart
│   └── eda_github_data.png             ← EDA visualization
│
├── presentation/
│   └── presentacion_mundial_2026.html  ← Full interactive HTML dashboard
│
└── docs/
    ├── MODEL_FINDINGS.md            ← Research diary: bugs found, AUC analysis, SHAP insights
    └── DATA_SOURCES.md              ← Data pipeline documentation and AI agent integration guide
```

---

## Setup & Usage

### Requirements
- Python 3.11+
- CUDA-compatible GPU (optional, strongly recommended) **or** Apple Silicon M2/M3/M4

### Installation

```bash
git clone https://github.com/WolfyIt/WorldCupModel.git
cd WorldCupModel
pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
jupyter lab mundial_2026_v2_advanced.ipynb
# → Kernel → Restart & Run All
```

The notebook will:
1. Auto-download all Fjelstul GitHub datasets
2. Build the dynamic ELO system
3. Engineer 30+ features
4. Train the ensemble (XGBoost + LightGBM + Random Forest)
5. Run 200-trial Optuna optimization
6. Execute 10,000 Monte Carlo tournament simulations
7. Output champion probabilities

### Apple Silicon (M2 / M3 / M4)

Replace Cell 1 of the notebook with `src/mundial_2026_m2pro.py` to enable:
- **MPS (Metal GPU)** for batch Monte Carlo inference (~3–5 min vs 8–10 min CPU)
- **Neural Engine (ANE)** via Core ML for RandomForest
- All ARM64 BLAS optimizations (NEON SIMD + AMX)

```bash
# Cell 1 replacement:
exec(open("src/mundial_2026_m2pro.py").read())
```

### Apply Official FIFA 2026 Draw

If you want to re-inject the official draw groups (already applied in the notebook):

```bash
python src/fix_groups_official.py
```

---

## Data Attribution

| Source | License | Usage |
|--------|---------|-------|
| [Fjelstul WC Database](https://github.com/jfjelstul/worldcup) | CC BY 4.0 | Historical WC data 1930–2022 |
| [Kaggle WC Dataset](https://www.kaggle.com/datasets/abecklas/fifa-world-cup) | CC0 Public Domain | Additional historical matches |
| [FIFA World Rankings](https://www.fifa.com/fifa-world-ranking) | Public | Team ranking initialization |
| [Transfermarkt](https://www.transfermarkt.com) | Scraping (research) | Squad market values |

---

## Technical Notes

- **Reproducibility:** All random seeds fixed (`np.random.seed(42)`, `random.seed(42)`)
- **Dependency pinning:** All package versions pinned in `requirements.txt`
- **Known limitation:** Binary labels (draws treated as losses) — see `docs/MODEL_FINDINGS.md` §9 for a full analysis and estimated improvement path to AUC ~0.79

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built with Python · XGBoost · LightGBM · Optuna · SHAP · joblib*
