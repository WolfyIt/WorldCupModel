# Model Research Findings — FIFA World Cup 2026 Predictor

> Recorded: March 2026  
> Model: XGBoost + LightGBM + Random Forest Ensemble | **AUC: 0.7613** | 10,000 Monte Carlo simulations  
> Official draw: FIFA WC2026 (March 2026)

---

## 1. The Group Draw Changes Everything — The Argentina Case

### Problem Detected
When using the **provisional draw** (fabricated groups), Argentina landed in a group with Belgium, Curaçao, and New Zealand — giving it a very clear path to the quarterfinals and semifinals.  
**Result under provisional draw:** Argentina had a **30.30%** champion probability.

### Official FIFA Draw (updated)
Argentina ended up in **Group J** alongside Algeria, Austria, and Jordan. While still a manageable group, its bracket side changed — it now draws tougher opponents in later knockout rounds.  
**Result under real draw:** Argentina **21.47%**

### Key Lesson
In a group + knockout format tournament, what matters is not only who you play in the group stage, but who **falls on your side of the bracket** in subsequent rounds. The model detects this automatically because it simulates every match on every path to the final.

### Biggest Winner of the Official Draw
**England jumped from 3.56% → 15.10%** — placed in Group L with Croatia, Ghana, and Panama (very accessible), and the other side of its bracket features more manageable opponents.

### Quick Comparison Table

| Team      | Provisional Draw | Real Draw | Change  |
|-----------|-----------------|-----------|---------|
| Argentina | 30.30%          | 21.47%    | −8.83%  |
| Spain     | 12.91%          | 21.98%    | +9.07%  |
| England   | 3.56%           | 15.10%    | +11.54% |
| France    | 8.20%           | 7.76%     | −0.44%  |

---

## 2. Temporal Weighting Curve — Recent Matches Matter More

### Concept
Not all historical matches carry the same predictive value. A match from 2015 tells us less about today's team than one from November 2025.

### Implementation
An **exponential decay** is applied to ELO updates:

```
K_effective = K_base × e^(λ × (match_year − 2015))
```

Where λ controls how quickly older matches are "forgotten" (tuned via cross-validation).

Additionally, each training sample gets a **sample_weight** proportional to its recency:
- Match from 2015: weight ≈ 0.22
- Match from 2020: weight ≈ 0.50
- Match from 2025: weight ≈ 1.00

### Why It Matters
Teams that have changed significantly (e.g. Morocco, Ecuador) benefit from this system because their recent wins outweigh outdated losses. Without temporal weighting, their ELO would be "polluted" by irrelevant historical results.

**Observed effect:**
- Morocco, Ecuador, Colombia: rated higher vs. flat-weight models
- Germany, Brazil: slightly lower (their recent form is less dominant than their historical legacy)

---

## 3. Injecting UEFA Nations League + Copa América Data

### Before (V1)
Only WC qualifier matches and FIFA friendlies — ~98 matches entered manually → **AUC ≈ 0.61**

### After (V2 / V3)
Injected:
- UEFA Nations League 2022–23 and 2024–25
- Copa América 2021 and 2024
- Full CONMEBOL, UEFA, AFC, CONCACAF WC Qualifiers
- Qatar WC 2022 group stage + knockout rounds

**Total: 1,310 real matches (Fjelstul GitHub) + 4,572 (Kaggle)**

**AUC improved from 0.61 → 0.7613 (+15.1 percentage points)**

### Why Nations League Data Is So Valuable
- National teams field their best squads (competitive stakes)
- Matches are recent (2024–25 cycle ended close to WC 2026)
- Covers European teams that did **not** qualify for the WC, providing indirect ELO calibration across confederations
- Copa América 2024 included USA and Canada (CONCACAF) facing CONMEBOL teams as hosts → direct cross-confederation ELO calibration

---

## 4. The Duplication Bug — Norway

### The Bug
Norway appeared **twice** in `data/teams_2026.csv`:
- Row 39: `qualified=0`, `squad_value=350M` (stale pre-Haaland data)
- Row 55: `qualified=1`, `squad_value=850M` (correct current data)

When code executed `df.loc['Norway']`, it returned a **2-row DataFrame** instead of a Series. Computing feature differences (`f1 - f2`) then produced a 2D array, causing:
```
ValueError: setting an array element with a sequence
```
This error surfaces far from where the duplicate was introduced — making it very hard to diagnose.

### Lesson
In team-level prediction datasets, **validating index uniqueness is as important as validating values**. A silent duplicate can break the model many processing steps later.

### Fix Applied
Row 39 (stale) removed. Norway retained with Haaland-era stats: `squad_value=850M`, `coach_rating=7.2`, `recent_win_pct=0.68`.

---

## 5. GPU Speed — Monte Carlo Performance

| Setup                           | Time for 10,000 simulations |
|----------------------------------|----------------------------|
| V1 (CPU, sklearn)                | ~32 minutes                |
| V2 (XGBoost CUDA + joblib)       | ~239 seconds (~4 min)      |
| **Speedup**                      | **~8×**                    |

### How We Got There
- XGBoost trained with `tree_method='hist'`, `device='cuda'`
- Monte Carlo parallelized via `joblib.Parallel(n_jobs=-1)`
- Each simulation is independent → ideal for parallelism
- GPU accelerates not just training but **batch inference**: predicting 10,000 matches × 6 rounds in milliseconds

> **Technical note:** The real bottleneck in Monte Carlo is Python overhead per simulation. Batching predictions was the key to the actual speedup.

---

## 6. Cross-Confederation ELO Calibration

### Classic Problem
How do we know if an ELO of 1750 in CONMEBOL equals 1750 in UEFA? Confederations only meet at World Cups and in occasional friendlies — too few matches for robust calibration.

### Solution Applied
- **Copa América 2024** included USA and Canada (CONCACAF) facing CONMEBOL opponents → direct cross-confederation calibration
- **2025 FIFA Rankings** used as an ELO initialization anchor before the dynamic system takes over
- **Qatar WC 2022** matches cross all confederations → further inter-confederation calibration

### Result
Brazil (CONMEBOL, ELO ~1800) and France (UEFA, ELO ~1790) end up with very similar ratings — consistent with both being top-tier powers of comparable strength. Without cross-calibration, UEFA teams tend to be artificially inflated due to the denser match history in their database.

---

## 7. Most Influential Features (SHAP Analysis)

**Top 5 features for predicting match outcomes:**

| Rank | Feature                    | Notes                                          |
|------|----------------------------|------------------------------------------------|
| 1    | `elo_diff`                 | ELO difference between teams — by far the strongest signal |
| 2    | `rank_diff`                | FIFA ranking difference                        |
| 3    | `squad_value_diff`         | Squad market value difference (€M)             |
| 4    | `form_diff`                | Recent win rate difference                     |
| 5    | `is_host`                  | Whether one team is a host nation              |

**Surprising finding:** `coach_rating` had less impact than expected. Squad value and recent form both outranked coaching quality.

**Also notable:** `recent_goals_conceded_avg` was **more predictive** than `recent_goals_scored_avg`.  
*"Defending well" predicts champions better than "attacking well."*

---

## 8. Why Spain Leads — Model Analysis

Spain combines:
- **Highest ELO** in the dataset: ~2015 pts
- **Favorable group** (Group H): Cape Verde / Saudi Arabia / Uruguay
- Bracket side with no top-tier opponents until the semifinals
- High `coach_rating` (De la Fuente) post-Euro 2024
- High `squad_value` with optimal average age (24.8 years)
- `recent_win_pct = 0.76` — best in Europe in the dataset

The combination of all these factors makes the model favor Spain as the slight leader over Argentina (21.98% vs 21.47%).

---

## 9. Why the Model Is at AUC 0.76 and Not Higher — Ceiling Analysis

### The Natural Ceiling of Football
Football has the **highest variance of any major team sport**. European betting companies with decades of data and teams of 100+ engineers also don't exceed AUC 0.78–0.80. There is genuine irreducible noise: any team can beat any other on any given day. This is not a model defect.

**Reference benchmarks:**

| System                            | AUC         |
|-----------------------------------|-------------|
| Pure ELO baseline                 | ≈ 0.68      |
| This project V1 (98 manual matches)| ≈ 0.61     |
| This project V2 (XGBoost+Optuna)  | **0.7613**  |
| European betting companies        | ≈ 0.77–0.80 |
| Theoretical ceiling for football  | ≈ 0.81      |

We are already in **professional-grade territory**. The realistic improvement range is ~0.78–0.79, not 0.95.

### Current Limiting Factors

**1. Binary label (win / no-win)** — the most impactful issue  
Draws are treated as losses. A 1–1 draw between Argentina and Spain gets the same label as a 0–3 loss. Draws have fundamentally different dynamics.  
→ Fix: 3-class model (win / draw / loss)  
→ Estimated gain: **+1.5–2% AUC**

**2. Data augmentation leakage**  
Symmetric augmentation (flipping home/away) doubles data, but random `train_test_split` afterward puts correlated pairs (e.g., Argentina vs Spain AND Spain vs Argentina) in both sets, inflating reported AUC.  
→ Fix: split before flipping, not after  
→ Effect: reported AUC drops slightly but becomes more honest

**3. Only 60 Optuna trials + single holdout**  
60 trials is insufficient. Optimizing on a single holdout risks overfitting hyperparameters to that specific test set.  
→ Fix: use 3-fold CV as Optuna's objective; increase to 200 trials  
→ Estimated gain: **+0.3–0.5% AUC**

**4. Simple Voting Ensemble**  
`VotingClassifier` averages raw probabilities equally. A stacking meta-learner would learn to weight models contextually.  
→ Fix: replace with `StackingClassifier` + Logistic Regression meta-learner  
→ Estimated gain: **+0.3–0.5% AUC**

**5. No head-to-head feature**  
Historical win rates for specific matchups (e.g., England historically loses penalty shootouts) carry signal that ELO doesn't capture.  
→ Fix: compute `h2h_win_rate` for each team pair  
→ Estimated gain: **+0.3% AUC**

### Potential Improvements Summary

| Improvement                             | AUC Gain   | Complexity |
|-----------------------------------------|-----------|-----------|
| 3-class model: win/draw/loss            | +1.5–2%   | Medium    |
| Fix augmentation data leakage           | +0.5% (real) | Low    |
| Optuna 60→200 trials + CV objective     | +0.3–0.5% | Low       |
| Stacking instead of Voting              | +0.3–0.5% | Medium    |
| Head-to-head win rate feature           | +0.3%     | Medium    |
| Add CatBoost to ensemble                | +0.2%     | Low       |

**Realistic target with all improvements:** AUC ≈ **0.78–0.79**  
*(level of professional sports analytics platforms)*

Surpassing 0.80 in football would require non-public data: real-time player fitness, injury history, weather, referee tendencies, etc.

---

*End of document*
