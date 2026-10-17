# Color Trend Analysis — Fashion Retail

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-LinearRegression-orange)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

> Identify which color families are gaining momentum in fashion retail and forecast the next season's top performers using regression-based trend scoring.

---

## Overview

Color is one of the most powerful levers in fashion merchandising — the right palette at the right moment drives sell-through; the wrong one leaves inventory stranded on the rack. This project simulates six years of product sales data (2019–2024) across ten color families, performs seasonal exploratory analysis, calculates a **momentum score** (the slope of each color's linear trend line), and projects the top five colors for the upcoming season.

The methodology mirrors the kind of color-intelligence work done by trend forecasters at major fast-fashion and data-driven retail companies.

---

## Skills Demonstrated

| Area | Detail |
|---|---|
| **Languages** | Python 3 |
| **Data Wrangling** | `pandas` — pivot tables, groupby aggregations, seasonal filtering |
| **Statistical Analysis** | Linear regression for trend-slope (momentum) scoring |
| **Machine Learning** | `scikit-learn` LinearRegression for multi-color forecasting |
| **Visualization** | `matplotlib`, `seaborn` — line charts, heatmaps, bar charts |
| **Domain Knowledge** | Fashion seasonality, color merchandising cycles |

---

## Dataset

All data is **synthetically generated** by the script using `numpy` with a fixed random seed (reproducible). The dataset contains ~5,000 records with the following schema:

| Column | Type | Description |
|---|---|---|
| `product_id` | string | Unique product identifier (e.g., P00001) |
| `color_family` | string | Color group (e.g., Terracotta, Sage Green) |
| `season` | string | Spring/Summer or Fall/Winter |
| `year` | int | 2019–2024 |
| `sales_volume` | int | Units sold |

Color trends are baked into the generation logic to simulate realistic cycles (e.g., Sage Green and Lavender rising steadily, Slate Grey declining).

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/zhynyraheure/data-science-portfolio.git
cd data-science-portfolio/color-trend-analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the analysis
python3 color_trend_analysis.py
```

Charts are saved automatically to `sample_output/`.

---

## Key Findings

- **Sage Green** showed the strongest positive momentum score, reflecting its consistent rise across both Spring/Summer and Fall/Winter collections from 2021–2024 — consistent with real-world wellness and sustainability aesthetics.
- **Lavender** posted the second-highest momentum, echoing the industry's broader shift toward soft, gender-neutral palettes.
- **Hot Pink** spiked sharply in 2023 (likely correlating with cultural moments) and is projected to remain elevated in 2025.
- **Slate Grey** is the only color family with a negative momentum score, indicating saturation after peaking as a pandemic-era staple.
- The **2025 forecast** places Sage Green, Cream/Ivory, and Lavender in the top three — signaling that buyers should weight investment in soft, nature-adjacent neutrals for the upcoming season.

---

## Output Charts

| Chart | Description |
|---|---|
| `color_timeline.png` | Multi-line sales trajectory per color family, 2019–2024 |
| `seasonal_heatmap.png` | Average sales heatmap by season × color |
| `momentum_scores.png` | Regression-slope ranking (green = rising, red = declining) |
| `forecast_2025.png` | Top 5 projected colors for 2025 |

---

## Author

**Zhynyrah (Zoe) Eure**
M.S. Data Science · National University (exp. December 2026)
Fashion professional with 8+ years of industry experience across styling, design, and editorial production.
