# Customer Purchase Prediction — Fashion E-Commerce

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-orange)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

> Predict whether a fashion e-commerce visitor will convert to a buyer using behavioral session data and a Random Forest classifier.

---

## Overview

Understanding which customers are likely to purchase — and which signals predict that decision — is one of the highest-value applications of machine learning in retail. This project builds a binary classifier trained on synthetic customer session data (browsing depth, session length, cart behavior, demographics) to predict purchase likelihood, then surfaces the features that matter most using Random Forest feature importances.

The approach reflects real-world CRM and conversion-rate optimization work done by data teams at fashion retailers and DTC brands.

---

## Skills Demonstrated

| Area | Detail |
|---|---|
| **Languages** | Python 3 |
| **Machine Learning** | `scikit-learn` — RandomForestClassifier, stratified train/test split, class balancing |
| **Evaluation** | Precision, Recall, F1-score, confusion matrix |
| **Data Wrangling** | `pandas` — label encoding, feature engineering, groupby aggregations |
| **Visualization** | `matplotlib`, `seaborn` — feature importance charts, segmentation bar charts |
| **Domain Knowledge** | E-commerce conversion funnels, customer segmentation, session analytics |

---

## Dataset

All data is **synthetically generated** by the script. The dataset contains 8,000 customer session records with realistic conversion rates (~35% overall) and segment-level variation.

| Column | Type | Description |
|---|---|---|
| `age_group` | string | Customer age bracket (18-24 through 55+) |
| `purchase_category` | string | Category browsed (Dresses, Tops, Outerwear, etc.) |
| `session_duration` | float | Time on site in seconds |
| `pages_viewed` | int | Number of pages visited |
| `added_to_cart` | int | Whether any item was added to cart (0/1) |
| `used_search` | int | Whether the site search was used (0/1) |
| `is_returning` | int | Returning vs. new visitor (0/1) |
| `purchased` | int | **Target variable** — did the customer buy? (0/1) |

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/zhynyraheure/data-science-portfolio.git
cd data-science-portfolio/customer-purchase-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the analysis
python3 purchase_prediction.py
```

Charts are saved automatically to `sample_output/`.

---

## Key Findings

- **Adding to cart** is the single strongest predictor of purchase, contributing the highest importance score — confirming that cart behavior is a reliable intent signal.
- **Session duration** and **pages viewed** rank second and third, meaning deeper engagement reliably correlates with conversion even before cart interaction.
- **Returning customers** convert at a meaningfully higher rate than new visitors, reinforcing the value of loyalty and re-engagement programs.
- **25-34** and **35-44** age groups show the highest conversion rates, making them the primary segments for re-targeting investment.
- **Outerwear** converts at the highest category rate, likely because it is a higher-consideration purchase that drives more intentional browsing sessions.
- The model achieves an F1-score above 0.80 on the positive (purchase) class, demonstrating strong practical utility for real-time scoring.

---

## Output Charts

| Chart | Description |
|---|---|
| `confusion_matrix.png` | Predicted vs. actual purchase outcomes across the test set |
| `feature_importance.png` | Ranked feature contributions to the Random Forest model |
| `conversion_by_segment.png` | Conversion rates broken out by age group and product category |

---

## Author

**Zhynyrah (Zoe) Eure**
M.S. Data Science · National University (exp. December 2026)
Fashion professional with 8+ years of industry experience across styling, design, and editorial production.
