# A/B Test Analysis — Fashion Email Campaign

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![scipy](https://img.shields.io/badge/scipy-z--test-purple)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

> Determine which email subject line drives higher purchase conversion for a fashion brand using a two-proportion z-test, confidence intervals, and Cohen's h effect size.

---

## Overview

A/B testing is the backbone of data-driven decision-making in retail marketing — it transforms gut-feel copy choices into statistically defensible conclusions. This project simulates a real email campaign experiment for a fictional fashion brand (Velour & Co.), comparing two subject line strategies: an emotion-driven line vs. a social-proof-driven line. The analysis runs a two-proportion z-test, computes a 95% confidence interval on the difference, calculates Cohen's h effect size, and delivers a clear deployment recommendation.

---

## Skills Demonstrated

| Area | Detail |
|---|---|
| **Languages** | Python 3 |
| **Statistical Testing** | Two-proportion z-test (scipy), p-value interpretation, hypothesis framing |
| **Effect Size** | Cohen's h for proportions, interpretation (small / medium / large) |
| **Confidence Intervals** | Wilson score CI per variant, CI on the rate difference |
| **Visualization** | Error-bar comparison chart, z-distribution plot with rejection regions |
| **Domain Knowledge** | Email marketing KPIs, conversion rate optimization, campaign design |

---

## Dataset

All data is **synthetically generated** by the script using `numpy.random.binomial` — no real customer data is used. Each record represents one email recipient.

| Column | Type | Description |
|---|---|---|
| `recipient_id` | string | Unique recipient identifier |
| `variant` | string | Control (A) or Treatment (B) |
| `subject_line` | string | The email subject shown to this recipient |
| `converted` | int | Did the recipient click through and purchase? (0/1) |

**Experiment Setup:**

| Parameter | Value |
|---|---|
| Total recipients | 10,000 (5,000 per variant) |
| Variant A subject | "Your Spring Wardrobe Is Waiting 🌸" |
| Variant B subject | "New In: The Pieces Everyone's Wearing This Season" |
| Significance level (α) | 0.05 |
| Test type | Two-tailed, two-proportion z-test |

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/zhynyraheure/data-science-portfolio.git
cd data-science-portfolio/ab-testing-analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the analysis
python3 ab_test_analysis.py
```

Charts are saved automatically to `sample_output/`. A full results summary and recommendation are printed to the console.

---

## Key Findings

- **Variant B** (social-proof framing: "New In: The Pieces Everyone's Wearing This Season") achieved a **16.8% conversion rate** vs. **14.2%** for Variant A — a 2.6 percentage point absolute lift.
- The result is **statistically significant** (p < 0.05), meaning we can reject the null hypothesis that the two subject lines perform equally.
- The **95% confidence interval** on the difference is approximately [+1.4%, +3.8%], indicating the true lift is unlikely to be zero.
- **Cohen's h = 0.073**, classified as a small-to-medium effect — meaningful at scale.
- At 5,000 sends, Variant B generates approximately **130 additional conversions** per campaign. Across a year with monthly sends, that compounds to ~1,560 additional purchases attributable to subject line copy alone.
- **Recommendation:** Deploy Variant B as the new control. Run a follow-up test against a personalized subject line to explore further upside.

---

## Output Charts

| Chart | Description |
|---|---|
| `conversion_comparison.png` | Bar chart with 95% CI error bars; significance annotation |
| `z_distribution.png` | Standard normal curve showing where the observed z-stat falls relative to critical values |

---

## Author

**Zhynyrah (Zoe) Eure**
M.S. Data Science · National University (exp. December 2026)
Fashion professional with 8+ years of industry experience across styling, design, and editorial production.
