# Data Sources & Integration Guide — FIFA World Cup 2026 Predictor

> Version 1.0 | March 2026

This document describes all data sources used for training the WC2026 prediction model, along with integration guidance for AI agents or future contributors.

The model architecture weights recent performance more heavily than historical results. **Data from 2022 onward is treated as the highest-priority signal.**

---

## 1. International Match Results

### 1.1 Fjelstul World Cup Database
| Property | Detail |
|----------|--------|
| **URL** | [github.com/jfjelstul/worldcup](https://github.com/jfjelstul/worldcup) |
| **Coverage** | All 22 men's World Cups (1930–2022), 27 datasets, 1.58M+ data points |
| **Format** | CSV / R package |
| **Key tables** | `team_appearances`, `match_results`, `group_standings`, `squad_compositions`, `award_winners` |
| **Model role** | Baseline historical priors — lower weight due to age of data |

### 1.2 worldfootballR (R library)
| Property | Detail |
|----------|--------|
| **URL** | [github.com/JaseZiv/worldfootballR](https://github.com/JaseZiv/worldfootballR) |
| **Coverage** | UEFA Euros, Copa América, CONMEBOL & UEFA WC Qualifiers, Nations League, Copa Libertadores — all updatable |
| **Format** | R DataFrames (exportable to CSV/Parquet) |
| **Key function** | `load_match_comp_results(comp_name = ...)` |
| **Model role** | **PRIMARY source** for national team performance 2021–2026. Covers Argentina Copa América 2021/2024, Finalissima 2022, Spain Euro 2024, Nations League wins |

### 1.3 StatsBomb Open Data
| Property | Detail |
|----------|--------|
| **URL** | [github.com/statsbomb/open-data](https://github.com/statsbomb/open-data) |
| **Coverage** | Event-level JSON for selected competitions: Euro 2020 & 2024, Copa América, Women's WC |
| **Format** | JSON (shots, passes, pressure events, 360° freeze frames) |
| **Model role** | Advanced feature engineering — xG per match, pressing intensity, progressive carry metrics |

---

## 2. Player Performance at Club Level (2020–2026)

### 2.1 FBref via soccerdata (Python)
| Property | Detail |
|----------|--------|
| **URL** | [soccerdata.readthedocs.io](https://soccerdata.readthedocs.io) / [fbref.com](https://fbref.com) |
| **Install** | `pip install soccerdata` |
| **Coverage** | Big 5 leagues (ENG, ESP, ITA, GER, FRA) + UCL, UEL — per player, per season |
| **Stat types** | Standard, shooting (xG/xA), passing (progressive), defense, possession, keeper |
| **Model role** | Squad quality features. Captures Yamal, Pedri, Rodri (Spain) and Messi, De Paul, Mac Allister (Argentina) current club form |

### 2.2 Understat
| Property | Detail |
|----------|--------|
| **URL** | [understat.com](https://understat.com) |
| **Coverage** | Big 5 leagues — match-level and player-level xG |
| **Format** | Scrapeable JSON API |
| **Model role** | Secondary xG source for cross-validation. Good for team-level offensive/defensive shape metrics |

### 2.3 SofaScore
| Property | Detail |
|----------|--------|
| **URL** | [sofascore.com](https://sofascore.com) |
| **Coverage** | Opta-sourced xG + player ratings across 500+ competitions |
| **Note** | Internal API required (no official public endpoint). Use Playwright/Selenium for scraping |
| **Model role** | Player ratings as squad quality proxy for non-Big-5 teams (Morocco, Japan, USA, etc.) |

---

## 3. Team Quality Proxies

### 3.1 Transfermarkt Market Values
| Property | Detail |
|----------|--------|
| **URL** | [transfermarkt.com](https://transfermarkt.com) / [github.com/dcaribou/transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets) |
| **Coverage** | Historical squad market values per player per season, all major nations |
| **Format** | Pre-scraped CSV dataset or live scraping via worldfootballR |
| **Model role** | Squad value = single-number proxy for squad quality. Highly correlated with WC performance. Key feature in group-stage predictions |

### 3.2 FIFA World Rankings (Time Series)
| Property | Detail |
|----------|--------|
| **URL** | [fifa.com/fifa-world-ranking](https://www.fifa.com/fifa-world-ranking) |
| **Coverage** | Monthly ranking points for all 211 FIFA member nations |
| **Format** | Scrapeable HTML or third-party APIs (api-football.com) |
| **Model role** | Ranking **trajectory** (delta over 12 months) used as a trend feature, not just absolute rank |

---

## 4. Real-Time & Current Form Feeds

### 4.1 API-Football
| Property | Detail |
|----------|--------|
| **URL** | [api-football.com](https://www.api-football.com) |
| **Coverage** | 1,200+ competitions, live scores, fixtures, lineups, player stats — real-time |
| **Format** | REST API (JSON). Free tier: 100 req/day |
| **Model role** | Injury updates, lineup confirmations, last-5-match form before each simulated match |

### 4.2 Sportmonks Football API
| Property | Detail |
|----------|--------|
| **URL** | [sportmonks.com](https://www.sportmonks.com) |
| **Coverage** | Win probabilities (ML-based), H2H, player availability, pre-match odds |
| **Format** | REST API (paid, trial plan available) |
| **Model role** | Cross-check model predictions against market-implied probabilities (calibration signal) |

---

## 5. Feature → Source Mapping

| Feature | Source | Priority | Format |
|---------|--------|----------|--------|
| WC historical results (1930–2022) | Fjelstul DB / Kaggle | Low (baseline) | CSV |
| National team results 2021–2026 | worldfootballR | 🔴 HIGH | R / CSV |
| Euro 2024, Copa América 2024 | worldfootballR + StatsBomb | 🔴 HIGH | R / JSON |
| CONMEBOL & UEFA WC Qualifiers | worldfootballR | 🔴 HIGH | R / CSV |
| Player club stats (Big 5) | FBref via soccerdata | 🟠 MED-HIGH | Python DataFrame |
| xG at player & team level | Understat / StatsBomb | 🟠 MED-HIGH | JSON / CSV |
| Squad market value | Transfermarkt datasets | 🟠 MED-HIGH | CSV |
| FIFA ranking time series | FIFA.com scraper | 🟠 MED-HIGH | CSV |
| Player ratings (non-Big-5 teams) | SofaScore | 🟡 MEDIUM | Scraped JSON |
| Injury & lineup (match day) | API-Football | 🔴 HIGH (live) | REST API |
| Pre-match odds / calibration | Sportmonks | 🟡 MEDIUM | REST API |

---

## 6. AI Agent Integration Notes

When an AI agent fetches and processes these sources, apply the following logic:

**Recency weighting:**
- Data from 2024–2026: highest weight
- Data from 2022–2023: medium weight
- Pre-2022: background context only

**Competition type hierarchy:**
> Official FIFA competitions > UEFA/CONMEBOL regional tournaments > Friendlies

**Spain context:**  
Models trained solely on World Cup data will underrate Spain. The agent must inject Spain's Euro 2024 title, Nations League performance, and club dominance (Barcelona, Real Madrid) as explicit features.

**Argentina context:**  
Argentina's 45-match unbeaten run, Copa América 2021 & 2024, and Finalissima 2022 are the strongest recency signals in the dataset. Treat as the #1 or #2 favorite.

**Data freshness:**
- API-Football and FIFA rankings: re-fetch 48h before each simulated or real match
- StatsBomb and FBref seasonal data: weekly cadence is sufficient

**Missing data handling:**  
For national teams outside the Big 5 pipeline (Morocco, Australia, USA), fall back to: FIFA ranking delta + Transfermarkt squad value + WC qualifier results as the feature set.

---

*End of document*
