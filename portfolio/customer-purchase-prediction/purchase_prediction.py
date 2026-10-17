"""
Customer Purchase Prediction — Fashion E-Commerce
==================================================
Generates synthetic customer session data for a fashion retailer, trains a
Random Forest classifier to predict purchase likelihood, and surfaces the
features that matter most to conversion.

Author: Zhynyrah (Zoe) Eure
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import warnings
import os

warnings.filterwarnings("ignore")
np.random.seed(42)

# ---------------------------------------------------------------------------
# 1. Synthetic Data Generation
# ---------------------------------------------------------------------------

def generate_customer_data(n_records: int = 8000) -> pd.DataFrame:
    """
    Generate synthetic customer session data for a fashion e-commerce site.

    Features designed to reflect realistic conversion drivers:
    - session_duration  : longer sessions correlate with higher purchase intent
    - pages_viewed      : proxy for browse depth and product exploration
    - age_group         : spending behavior differs across demographics
    - purchase_category : category browsed during session
    - purchased         : target label (1 = converted, 0 = bounced)
    """
    age_groups = ["18-24", "25-34", "35-44", "45-54", "55+"]
    categories = ["Dresses", "Tops", "Outerwear", "Accessories", "Shoes", "Activewear"]

    # Age-group base conversion rates (younger = lower conversion, mid = higher)
    age_conversion = {"18-24": 0.22, "25-34": 0.38, "35-44": 0.42, "45-54": 0.35, "55+": 0.28}
    # Category base conversion rates (higher-involvement categories convert better)
    cat_conversion  = {"Dresses": 0.40, "Tops": 0.32, "Outerwear": 0.45,
                       "Accessories": 0.30, "Shoes": 0.38, "Activewear": 0.35}

    records = []
    for _ in range(n_records):
        age   = np.random.choice(age_groups, p=[0.25, 0.30, 0.20, 0.15, 0.10])
        cat   = np.random.choice(categories)

        # Session behavior: non-purchasers browse less
        will_buy_base = age_conversion[age] * cat_conversion[cat] / 0.35  # normalize

        session_duration = np.random.gamma(shape=2.5, scale=120) if np.random.rand() < will_buy_base \
                           else np.random.gamma(shape=1.5, scale=80)
        pages_viewed     = np.random.poisson(lam=6 if np.random.rand() < will_buy_base else 3)
        pages_viewed     = max(pages_viewed, 1)

        # Derived signals
        added_to_cart    = int(np.random.rand() < (0.5 * will_buy_base + 0.05 * pages_viewed / 10))
        used_search      = int(np.random.rand() < 0.45)
        is_returning     = int(np.random.rand() < 0.35)

        # Purchase probability adjusted by behavioral signals
        purchase_prob = (
            will_buy_base
            + 0.30 * added_to_cart
            + 0.08 * is_returning
            + 0.04 * used_search
            + 0.002 * min(pages_viewed, 20)
            + 0.0003 * min(session_duration, 600)
        )
        purchased = int(np.random.rand() < min(purchase_prob, 0.95))

        records.append({
            "age_group":         age,
            "purchase_category": cat,
            "session_duration":  round(session_duration, 1),   # seconds
            "pages_viewed":      pages_viewed,
            "added_to_cart":     added_to_cart,
            "used_search":       used_search,
            "is_returning":      is_returning,
            "purchased":         purchased,
        })

    df = pd.DataFrame(records)
    conversion_rate = df["purchased"].mean()
    print(f"Dataset generated: {len(df):,} records | Overall conversion rate: {conversion_rate:.1%}")
    print(df.head())
    return df


# ---------------------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame):
    """Encode categoricals, split into train/test sets."""
    df_enc = df.copy()
    le_age = LabelEncoder()
    le_cat = LabelEncoder()
    df_enc["age_group_enc"]         = le_age.fit_transform(df_enc["age_group"])
    df_enc["purchase_category_enc"] = le_cat.fit_transform(df_enc["purchase_category"])

    feature_cols = [
        "age_group_enc", "purchase_category_enc",
        "session_duration", "pages_viewed",
        "added_to_cart", "used_search", "is_returning",
    ]
    X = df_enc[feature_cols]
    y = df_enc["purchased"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nTrain size: {len(X_train):,} | Test size: {len(X_test):,}")
    return X_train, X_test, y_train, y_test, feature_cols


# ---------------------------------------------------------------------------
# 3. Model Training
# ---------------------------------------------------------------------------

def train_model(X_train, y_train) -> RandomForestClassifier:
    """Train a Random Forest classifier with class balancing."""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("\nModel trained: RandomForestClassifier (200 trees, max_depth=12)")
    return model


# ---------------------------------------------------------------------------
# 4. Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test, y_test, output_dir: str) -> None:
    """Print classification report and save a confusion matrix."""
    y_pred = model.predict(X_test)

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=["No Purchase", "Purchase"]))

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=["No Purchase", "Purchase"],
        cmap="Blues", ax=ax
    )
    ax.set_title("Confusion Matrix — Purchase Prediction", fontsize=13, fontweight="bold", pad=10)
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# 5. Feature Importance
# ---------------------------------------------------------------------------

FEATURE_LABELS = {
    "age_group_enc":          "Age Group",
    "purchase_category_enc":  "Product Category",
    "session_duration":       "Session Duration (s)",
    "pages_viewed":           "Pages Viewed",
    "added_to_cart":          "Added to Cart",
    "used_search":            "Used Site Search",
    "is_returning":           "Returning Customer",
}

def plot_feature_importance(model, feature_cols: list, output_dir: str) -> None:
    """Horizontal bar chart of Random Forest feature importances."""
    importance_df = (
        pd.DataFrame({
            "feature":    feature_cols,
            "importance": model.feature_importances_,
        })
        .assign(label=lambda d: d["feature"].map(FEATURE_LABELS))
        .sort_values("importance", ascending=True)
    )

    palette = sns.color_palette("viridis", n_colors=len(importance_df))

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(
        importance_df["label"],
        importance_df["importance"],
        color=palette, edgecolor="white", height=0.65
    )
    # Value labels
    for bar in bars:
        ax.text(
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.3f}",
            va="center", ha="left", fontsize=9
        )
    ax.set_title("Feature Importance — Purchase Prediction Model", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Importance Score (Mean Decrease in Impurity)", fontsize=11)
    ax.set_xlim(0, importance_df["importance"].max() * 1.18)
    sns.despine()
    plt.tight_layout()
    path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# 6. Conversion Rate by Segment
# ---------------------------------------------------------------------------

def plot_conversion_by_segment(df: pd.DataFrame, output_dir: str) -> None:
    """Bar charts of conversion rate broken out by age group and category."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Age group
    age_conv = df.groupby("age_group")["purchased"].mean().reindex(
        ["18-24", "25-34", "35-44", "45-54", "55+"]
    )
    axes[0].bar(age_conv.index, age_conv.values,
                color=sns.color_palette("coolwarm", n_colors=5), edgecolor="white")
    axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    axes[0].set_title("Conversion Rate by Age Group", fontsize=13, fontweight="bold")
    axes[0].set_ylim(0, age_conv.max() * 1.25)

    # Category
    cat_conv = df.groupby("purchase_category")["purchased"].mean().sort_values(ascending=False)
    axes[1].bar(cat_conv.index, cat_conv.values,
                color=sns.color_palette("mako", n_colors=len(cat_conv)), edgecolor="white")
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    axes[1].set_title("Conversion Rate by Product Category", fontsize=13, fontweight="bold")
    axes[1].set_ylim(0, cat_conv.max() * 1.25)
    plt.setp(axes[1].get_xticklabels(), rotation=20, ha="right")

    for ax in axes:
        sns.despine(ax=ax)
    plt.tight_layout()
    path = os.path.join(output_dir, "conversion_by_segment.png")
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
    print("  Customer Purchase Prediction — Fashion E-Commerce")
    print("=" * 60)

    df = generate_customer_data()
    X_train, X_test, y_train, y_test, feature_cols = preprocess(df)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test, output_dir)
    plot_feature_importance(model, feature_cols, output_dir)
    plot_conversion_by_segment(df, output_dir)

    print("\n" + "=" * 60)
    print("  Analysis complete. Charts saved to sample_output/")
    print("=" * 60)


if __name__ == "__main__":
    main()
