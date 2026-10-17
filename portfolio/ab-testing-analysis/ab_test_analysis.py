"""
A/B Test Analysis — Email Campaign for a Fashion Brand
=======================================================
Simulates an A/B test comparing two email subject lines for a fashion
retailer's promotional campaign. Runs a two-proportion z-test, calculates
p-value, confidence interval, and Cohen's h effect size, then prints a
structured results summary with a recommendation.

Author: Zhynyrah (Zoe) Eure
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import warnings
import os

warnings.filterwarnings("ignore")
np.random.seed(42)

# ---------------------------------------------------------------------------
# 1. Experiment Configuration
# ---------------------------------------------------------------------------

EXPERIMENT = {
    "name":         "Spring Launch Email Campaign — Subject Line Test",
    "brand":        "Velour & Co.",
    "variant_a": {
        "label":           "Control (A)",
        "subject":         "Your Spring Wardrobe Is Waiting 🌸",
        "send_size":       5000,
        "true_conv_rate":  0.142,   # 14.2% click-through → purchase
    },
    "variant_b": {
        "label":           "Treatment (B)",
        "subject":         "New In: The Pieces Everyone's Wearing This Season",
        "send_size":       5000,
        "true_conv_rate":  0.168,   # 16.8% — social-proof framing lifts conversion
    },
    "alpha":     0.05,    # significance threshold
    "metric":    "Conversion Rate (click → purchase)",
}


# ---------------------------------------------------------------------------
# 2. Simulate Campaign Data
# ---------------------------------------------------------------------------

def simulate_ab_data(config: dict) -> pd.DataFrame:
    """
    Generate individual-level records for each email recipient.
    Each row represents one recipient; 'converted' is the outcome.
    """
    records = []
    for variant_key in ("variant_a", "variant_b"):
        v = config[variant_key]
        n = v["send_size"]
        conversions = np.random.binomial(1, v["true_conv_rate"], size=n)
        for i, c in enumerate(conversions):
            records.append({
                "recipient_id": f"R{len(records)+1:06d}",
                "variant":      v["label"],
                "subject_line": v["subject"],
                "converted":    c,
            })

    df = pd.DataFrame(records)
    print(f"Simulated {len(df):,} email recipients across 2 variants.")
    return df


# ---------------------------------------------------------------------------
# 3. Two-Proportion Z-Test
# ---------------------------------------------------------------------------

def two_proportion_z_test(
    n_a: int, conv_a: int,
    n_b: int, conv_b: int,
    alpha: float = 0.05,
) -> dict:
    """
    Perform a two-proportion z-test.

    H0: p_A = p_B  (no difference in conversion rates)
    H1: p_A ≠ p_B  (two-tailed)

    Returns a dict with rates, z-statistic, p-value, CI, and effect size.
    """
    p_a = conv_a / n_a
    p_b = conv_b / n_b
    p_pool = (conv_a + conv_b) / (n_a + n_b)

    # Standard error under the null
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z_stat = (p_b - p_a) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))   # two-tailed

    # 95% Confidence Interval on the difference p_B - p_A
    se_diff = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    z_crit  = stats.norm.ppf(1 - alpha / 2)
    diff    = p_b - p_a
    ci_low  = diff - z_crit * se_diff
    ci_high = diff + z_crit * se_diff

    # Cohen's h (effect size for proportions)
    phi_a   = 2 * np.arcsin(np.sqrt(p_a))
    phi_b   = 2 * np.arcsin(np.sqrt(p_b))
    cohens_h = abs(phi_b - phi_a)

    # Effect size interpretation
    if cohens_h < 0.20:
        effect_label = "Small"
    elif cohens_h < 0.50:
        effect_label = "Medium"
    else:
        effect_label = "Large"

    significant = p_value < alpha

    return {
        "p_a": p_a, "p_b": p_b, "diff": diff,
        "z_stat": z_stat, "p_value": p_value,
        "ci_low": ci_low, "ci_high": ci_high,
        "cohens_h": cohens_h, "effect_label": effect_label,
        "significant": significant,
        "n_a": n_a, "n_b": n_b,
        "conv_a": conv_a, "conv_b": conv_b,
    }


# ---------------------------------------------------------------------------
# 4. Results Summary
# ---------------------------------------------------------------------------

def print_results(results: dict, config: dict) -> None:
    """Print a structured, readable experiment summary."""
    exp = config
    va  = exp["variant_a"]
    vb  = exp["variant_b"]
    r   = results

    print("\n" + "=" * 65)
    print(f"  EXPERIMENT: {exp['name']}")
    print(f"  Brand:      {exp['brand']}")
    print(f"  Metric:     {exp['metric']}")
    print("=" * 65)

    print(f"\n  Variant A — {va['label']}")
    print(f"    Subject   : \"{va['subject']}\"")
    print(f"    Recipients: {r['n_a']:,}")
    print(f"    Conversions: {r['conv_a']:,}")
    print(f"    Rate:       {r['p_a']:.2%}")

    print(f"\n  Variant B — {vb['label']}")
    print(f"    Subject   : \"{vb['subject']}\"")
    print(f"    Recipients: {r['n_b']:,}")
    print(f"    Conversions: {r['conv_b']:,}")
    print(f"    Rate:       {r['p_b']:.2%}")

    print("\n  --- Statistical Test Results ---")
    print(f"    Absolute Difference : {r['diff']:+.3%}")
    print(f"    Relative Lift       : {r['diff'] / r['p_a']:+.1%}")
    print(f"    Z-Statistic         : {r['z_stat']:.4f}")
    print(f"    P-Value             : {r['p_value']:.4f}")
    print(f"    95% Confidence Interval: [{r['ci_low']:+.3%}, {r['ci_high']:+.3%}]")
    print(f"    Cohen's h (effect)  : {r['cohens_h']:.4f} ({r['effect_label']})")
    print(f"    Significance Level  : α = {config['alpha']}")

    sig_str = "STATISTICALLY SIGNIFICANT" if r["significant"] else "NOT statistically significant"
    print(f"\n  Result: {sig_str} (p = {r['p_value']:.4f})")

    print("\n  --- Recommendation ---")
    if r["significant"] and r["p_b"] > r["p_a"]:
        print(
            f"  Deploy Variant B. The social-proof subject line\n"
            f"  outperforms the control by {r['diff']:+.2%} absolute "
            f"({r['diff'] / r['p_a']:+.1%} relative lift).\n"
            f"  At {r['n_b']:,} sends, this translates to approximately\n"
            f"  {int(r['diff'] * r['n_b']):,} additional conversions per campaign."
        )
    elif r["significant"] and r["p_b"] < r["p_a"]:
        print("  Keep Variant A. Treatment underperformed the control.")
    else:
        print(
            "  No significant difference detected. Continue with the control\n"
            "  or extend the test to increase statistical power."
        )

    print("=" * 65)


# ---------------------------------------------------------------------------
# 5. Visualizations
# ---------------------------------------------------------------------------

def plot_conversion_comparison(results: dict, config: dict, output_dir: str) -> None:
    """Bar chart with 95% CI error bars comparing variant conversion rates."""
    r   = results
    va  = config["variant_a"]
    vb  = config["variant_b"]

    labels = [va["label"], vb["label"]]
    rates  = [r["p_a"], r["p_b"]]

    # Wilson score CIs per variant
    def wilson_ci(p, n, z=1.96):
        denom = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denom
        margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
        return center - margin, center + margin

    ci_a = wilson_ci(r["p_a"], r["n_a"])
    ci_b = wilson_ci(r["p_b"], r["n_b"])

    err_low  = [rates[0] - ci_a[0], rates[1] - ci_b[0]]
    err_high = [ci_a[1] - rates[0], ci_b[1] - rates[1]]

    colors = ["#5b8db8", "#e07b54"]
    sig_color = "#27ae60" if r["significant"] else "#e74c3c"

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(
        labels, rates, color=colors, width=0.45, edgecolor="white",
        yerr=[err_low, err_high],
        error_kw={"elinewidth": 2, "capsize": 8, "capthick": 2, "ecolor": "#333333"},
        zorder=3,
    )

    # Annotate bars
    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(err_high) + 0.003,
            f"{rate:.2%}", ha="center", va="bottom",
            fontsize=13, fontweight="bold"
        )

    # Significance annotation
    y_bracket = max(rates) + max(err_high) + 0.015
    ax.annotate(
        "", xy=(1, y_bracket), xytext=(0, y_bracket),
        arrowprops=dict(arrowstyle="-", color=sig_color, lw=2)
    )
    sig_text = f"p = {r['p_value']:.4f} {'✓ Significant' if r['significant'] else '✗ Not Significant'}"
    ax.text(0.5, y_bracket + 0.004, sig_text, ha="center", va="bottom",
            fontsize=11, color=sig_color, fontweight="bold")

    ax.set_ylim(0, y_bracket + 0.030)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_ylabel("Conversion Rate", fontsize=12)
    ax.set_title(
        f"A/B Test: Email Subject Line Conversion Comparison\n{config['brand']}",
        fontsize=13, fontweight="bold", pad=14
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    sns.despine()
    plt.tight_layout()
    path = os.path.join(output_dir, "conversion_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nSaved: {path}")


def plot_p_value_distribution(results: dict, config: dict, output_dir: str) -> None:
    """
    Visualize where the observed z-statistic falls on the standard normal
    distribution, shading the rejection regions.
    """
    r      = results
    alpha  = config["alpha"]
    z_crit = stats.norm.ppf(1 - alpha / 2)

    x = np.linspace(-4.5, 4.5, 1000)
    y = stats.norm.pdf(x)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y, color="#333333", linewidth=2)

    # Rejection regions
    ax.fill_between(x, y, where=(x <= -z_crit), color="#e74c3c", alpha=0.35, label=f"Rejection region (α/2={alpha/2})")
    ax.fill_between(x, y, where=(x >=  z_crit), color="#e74c3c", alpha=0.35)

    # Observed z
    ax.axvline(r["z_stat"], color="#2980b9", linewidth=2.5, linestyle="--",
               label=f"Observed z = {r['z_stat']:.3f}")

    # Critical lines
    ax.axvline(-z_crit, color="#7f8c8d", linewidth=1.2, linestyle=":")
    ax.axvline( z_crit, color="#7f8c8d", linewidth=1.2, linestyle=":",
                label=f"Critical values (±{z_crit:.2f})")

    ax.set_title("Two-Proportion Z-Test: Standard Normal Distribution", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Z-Score", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.legend(fontsize=10)
    sns.despine()
    plt.tight_layout()
    path = os.path.join(output_dir, "z_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    output_dir = os.path.join(os.path.dirname(__file__), "sample_output")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 65)
    print("  A/B Test Analysis — Fashion Email Campaign")
    print("=" * 65)

    # Step 1: Simulate data
    df = simulate_ab_data(EXPERIMENT)

    # Step 2: Aggregate counts
    summary = df.groupby("variant").agg(
        recipients=("converted", "count"),
        conversions=("converted", "sum"),
    ).reset_index()

    a_row = summary[summary["variant"] == EXPERIMENT["variant_a"]["label"]].iloc[0]
    b_row = summary[summary["variant"] == EXPERIMENT["variant_b"]["label"]].iloc[0]

    # Step 3: Statistical test
    results = two_proportion_z_test(
        n_a=int(a_row["recipients"]),   conv_a=int(a_row["conversions"]),
        n_b=int(b_row["recipients"]),   conv_b=int(b_row["conversions"]),
        alpha=EXPERIMENT["alpha"],
    )

    # Step 4: Print results
    print_results(results, EXPERIMENT)

    # Step 5: Visualizations
    plot_conversion_comparison(results, EXPERIMENT, output_dir)
    plot_p_value_distribution(results, EXPERIMENT, output_dir)

    print("\n" + "=" * 65)
    print("  Analysis complete. Charts saved to sample_output/")
    print("=" * 65)


if __name__ == "__main__":
    main()
