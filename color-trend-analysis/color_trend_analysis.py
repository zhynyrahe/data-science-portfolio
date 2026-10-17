"""
Color Trend Analysis for Fashion Retail
========================================
Generates synthetic H&M-style fashion data, performs exploratory data analysis,
calculates color momentum scores using linear regression, and forecasts next
season's top colors.

Author: Zhynyrah (Zoe) Eure
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.linear_model import LinearRegression
import warnings
import os

warnings.filterwarnings("ignore")
np.random.seed(42)

# ---------------------------------------------------------------------------
# 1. Synthetic Data Generation
# ---------------------------------------------------------------------------

def generate_fashion_data(n_records: int = 5000) -> pd.DataFrame:
    """
    Generate synthetic product sales data resembling a fast-fashion retailer.

    Returns a DataFrame with columns:
        product_id, color_family, season, year, sales_volume
    """
    color_families = [
        "Cobalt Blue", "Terracotta", "Sage Green", "Cream/Ivory",
        "Hot Pink", "Chocolate Brown", "Lavender", "Butter Yellow",
        "Slate Grey", "Rust Orange",
    ]

    seasons = ["Spring/Summer", "Fall/Winter"]
    years = [2019, 2020, 2021, 2022, 2023, 2024]

    # Define base sales and trend multipliers per color per year
    # (simulating real-world color cycles)
    color_trends = {
        "Cobalt Blue":      [1.0, 0.9, 1.2, 1.5, 1.3, 1.1],
        "Terracotta":       [0.7, 0.8, 1.3, 1.6, 1.8, 1.5],
        "Sage Green":       [0.6, 0.7, 1.0, 1.4, 1.9, 2.0],
        "Cream/Ivory":      [1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
        "Hot Pink":         [0.9, 0.8, 0.7, 1.2, 1.8, 1.6],
        "Chocolate Brown":  [1.5, 1.2, 1.0, 0.9, 1.3, 1.7],
        "Lavender":         [0.5, 0.6, 0.8, 1.0, 1.4, 1.8],
        "Butter Yellow":    [1.1, 1.0, 0.9, 1.1, 1.3, 1.2],
        "Slate Grey":       [1.4, 1.5, 1.3, 1.2, 1.1, 1.0],
        "Rust Orange":      [0.8, 1.0, 1.2, 1.5, 1.4, 1.2],
    }

    base_sales = {color: np.random.randint(300, 700) for color in color_families}

    records = []
    pid = 1
    for year_idx, year in enumerate(years):
        for season in seasons:
            for color in color_families:
                multiplier = color_trends[color][year_idx]
                # Season modifier: some colors sell better in specific seasons
                season_boost = 1.15 if (
                    (color in ["Sage Green", "Butter Yellow", "Cobalt Blue"] and season == "Spring/Summer") or
                    (color in ["Chocolate Brown", "Slate Grey", "Terracotta"] and season == "Fall/Winter")
                ) else 1.0
                n_products = np.random.randint(30, 80)
                for _ in range(n_products):
                    sales = int(
                        base_sales[color] * multiplier * season_boost *
                        np.random.uniform(0.7, 1.3)
                    )
                    records.append({
                        "product_id": f"P{pid:05d}",
                        "color_family": color,
                        "season": season,
                        "year": year,
                        "sales_volume": max(sales, 10),
                    })
                    pid += 1

    df = pd.DataFrame(records)
    print(f"Dataset generated: {len(df):,} records | {df['product_id'].nunique():,} unique products")
    print(df.head())
    return df


# ---------------------------------------------------------------------------
# 2. Exploratory Data Analysis
# ---------------------------------------------------------------------------

def run_eda(df: pd.DataFrame) -> pd.DataFrame:
    """Print summary statistics and aggregate sales by color/year."""
    print("\n--- EDA Summary ---")
    print(df.describe())
    print("\nSales by Color Family (all years):")
    color_totals = (
        df.groupby("color_family")["sales_volume"]
        .sum()
        .sort_values(ascending=False)
    )
    print(color_totals.to_string())

    # Yearly aggregated pivot
    pivot = (
        df.groupby(["year", "color_family"])["sales_volume"]
        .sum()
        .reset_index()
        .pivot(index="year", columns="color_family", values="sales_volume")
        .fillna(0)
    )
    return pivot


# ---------------------------------------------------------------------------
# 3. Color Timeline Visualization
# ---------------------------------------------------------------------------

def plot_color_timeline(pivot: pd.DataFrame, output_dir: str) -> None:
    """Line chart showing each color family's sales trajectory over years."""
    palette = sns.color_palette("tab10", n_colors=len(pivot.columns))

    fig, ax = plt.subplots(figsize=(13, 7))
    for idx, color in enumerate(pivot.columns):
        ax.plot(
            pivot.index, pivot[color],
            marker="o", linewidth=2.2, label=color, color=palette[idx]
        )

    ax.set_title("Color Family Sales Trends (2019–2024)", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Year", fontsize=13)
    ax.set_ylabel("Total Sales Volume", fontsize=13)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    sns.despine()
    plt.tight_layout()
    path = os.path.join(output_dir, "color_timeline.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nSaved: {path}")


# ---------------------------------------------------------------------------
# 4. Seasonal Heatmap
# ---------------------------------------------------------------------------

def plot_seasonal_heatmap(df: pd.DataFrame, output_dir: str) -> None:
    """Heatmap of average sales by season and color family."""
    heat_data = (
        df.groupby(["season", "color_family"])["sales_volume"]
        .mean()
        .unstack(fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(13, 4))
    sns.heatmap(
        heat_data, annot=True, fmt=".0f", cmap="YlOrRd",
        linewidths=0.5, ax=ax, cbar_kws={"label": "Avg Sales Volume"}
    )
    ax.set_title("Average Sales Volume by Season & Color Family", fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=30, ha="right", fontsize=10)
    plt.tight_layout()
    path = os.path.join(output_dir, "seasonal_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# 5. Momentum Scoring via Linear Regression
# ---------------------------------------------------------------------------

def calculate_momentum_scores(pivot: pd.DataFrame) -> pd.DataFrame:
    """
    Fit a linear regression on each color's yearly sales to derive a
    momentum score (regression slope). Higher slope = stronger upward trend.
    """
    X = pivot.index.values.reshape(-1, 1)
    scores = {}
    for color in pivot.columns:
        y = pivot[color].values
        model = LinearRegression().fit(X, y)
        scores[color] = round(model.coef_[0], 2)

    momentum_df = (
        pd.DataFrame.from_dict(scores, orient="index", columns=["momentum_score"])
        .sort_values("momentum_score", ascending=False)
        .reset_index()
        .rename(columns={"index": "color_family"})
    )
    print("\n--- Momentum Scores (slope of linear regression) ---")
    print(momentum_df.to_string(index=False))
    return momentum_df


def plot_momentum_scores(momentum_df: pd.DataFrame, output_dir: str) -> None:
    """Horizontal bar chart of color momentum scores."""
    colors_bar = ["#2ecc71" if v >= 0 else "#e74c3c" for v in momentum_df["momentum_score"]]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        momentum_df["color_family"], momentum_df["momentum_score"],
        color=colors_bar, edgecolor="white", height=0.65
    )
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Color Momentum Scores (Linear Regression Slope)", fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Momentum Score (units / year)", fontsize=12)
    ax.invert_yaxis()
    sns.despine()
    plt.tight_layout()
    path = os.path.join(output_dir, "momentum_scores.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# 6. Forecast: Next Season's Top Colors
# ---------------------------------------------------------------------------

def forecast_next_season(pivot: pd.DataFrame, momentum_df: pd.DataFrame, output_dir: str) -> None:
    """
    Project each color's sales into 2025 using its linear regression model,
    then rank and visualize the top 5 forecasted colors.
    """
    X = pivot.index.values.reshape(-1, 1)
    forecast_year = np.array([[2025]])
    forecasts = {}
    for color in pivot.columns:
        y = pivot[color].values
        model = LinearRegression().fit(X, y)
        forecasts[color] = max(int(model.predict(forecast_year)[0]), 0)

    forecast_df = (
        pd.DataFrame.from_dict(forecasts, orient="index", columns=["projected_sales_2025"])
        .sort_values("projected_sales_2025", ascending=False)
        .reset_index()
        .rename(columns={"index": "color_family"})
    )

    print("\n--- 2025 Forecast: Projected Top Colors ---")
    print(forecast_df.to_string(index=False))

    top5 = forecast_df.head(5)
    palette_top = sns.color_palette("magma_r", n_colors=5)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        top5["color_family"], top5["projected_sales_2025"],
        color=palette_top, edgecolor="white", width=0.6
    )
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2000,
            f"{int(bar.get_height()):,}",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )
    ax.set_title("Top 5 Forecasted Colors for 2025", fontsize=15, fontweight="bold", pad=12)
    ax.set_ylabel("Projected Sales Volume", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlabel("")
    plt.xticks(rotation=20, ha="right", fontsize=10)
    sns.despine()
    plt.tight_layout()
    path = os.path.join(output_dir, "forecast_2025.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    output_dir = os.path.join(os.path.dirname(__file__), "sample_output")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  Color Trend Analysis — Fashion Retail")
    print("=" * 60)

    # Step 1: Generate data
    df = generate_fashion_data()

    # Step 2: EDA + pivot table
    pivot = run_eda(df)

    # Step 3: Visualizations
    plot_color_timeline(pivot, output_dir)
    plot_seasonal_heatmap(df, output_dir)

    # Step 4: Momentum scores
    momentum_df = calculate_momentum_scores(pivot)
    plot_momentum_scores(momentum_df, output_dir)

    # Step 5: Forecast
    forecast_next_season(pivot, momentum_df, output_dir)

    print("\n" + "=" * 60)
    print("  Analysis complete. Charts saved to sample_output/")
    print("=" * 60)


if __name__ == "__main__":
    main()
