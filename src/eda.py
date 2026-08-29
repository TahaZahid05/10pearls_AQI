import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import BASE_DIR, HISTORICAL_PARQUET

PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def plot_aqi_distribution(df: pd.DataFrame):
    def categorize_aqi(val):
        if val <= 50:
            return "Good (0-50)"
        elif val <= 100:
            return "Moderate (51-100)"
        elif val <= 150:
            return "Unhealthy for Sensitive (101-150)"
        elif val <= 200:
            return "Unhealthy (151-200)"
        elif val <= 300:
            return "Very Unhealthy (201-300)"
        else:
            return "Hazardous (301+)"

    categories = df["us_aqi"].apply(categorize_aqi)
    cat_counts = categories.value_counts()
    
    order = [
        "Good (0-50)",
        "Moderate (51-100)",
        "Unhealthy for Sensitive (101-150)",
        "Unhealthy (151-200)",
        "Very Unhealthy (201-300)",
        "Hazardous (301+)",
    ]
    cat_counts = cat_counts.reindex([c for c in order if c in cat_counts.index])
    colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#9b59b6", "#7e0023"][:len(cat_counts)]

    plt.figure(figsize=(9, 5))
    bars = plt.bar(cat_counts.index, cat_counts.values, color=colors, edgecolor="black", linewidth=0.8)
    
    for bar in bars:
        height = bar.get_height()
        pct = (height / len(df)) * 100
        plt.text(bar.get_x() + bar.get_width() / 2.0, height + 50, f"{height:,} ({pct:.1f}%)", ha="center", fontsize=9)

    plt.title("Distribution of Karachi Air Quality Categories (Historical Dataset)", fontsize=12, fontweight="bold")
    plt.xlabel("EPA AQI Category", fontsize=10)
    plt.ylabel("Number of Hours", fontsize=10)
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_path = PLOTS_DIR / "eda_aqi_distribution.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_correlation_heatmap(df: pd.DataFrame):
    cols = [
        "us_aqi",
        "pm2_5",
        "pm10",
        "ozone",
        "nitrogen_dioxide",
        "carbon_monoxide",
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "surface_pressure",
        "wind_v",
    ]
    valid_cols = [c for c in cols if c in df.columns]
    corr = df[valid_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax, label="Pearson Correlation")

    ax.set_xticks(range(len(valid_cols)))
    ax.set_yticks(range(len(valid_cols)))
    ax.set_xticklabels(valid_cols, rotation=45, ha="left", fontsize=9)
    ax.set_yticklabels(valid_cols, fontsize=9)

    for i in range(len(valid_cols)):
        for j in range(len(valid_cols)):
            val = corr.iloc[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

    plt.title("Correlation Matrix of Pollutants & Weather Features", fontsize=12, fontweight="bold", pad=20)
    plt.tight_layout()

    out_path = PLOTS_DIR / "eda_correlation_heatmap.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_diurnal_cycle(df: pd.DataFrame):
    hourly_stats = df.groupby("hour")["us_aqi"].agg(["mean", "std", "median"])

    plt.figure(figsize=(9, 5))
    plt.plot(hourly_stats.index, hourly_stats["mean"], marker="o", color="#2980b9", linewidth=2.5, label="Mean AQI")
    plt.plot(hourly_stats.index, hourly_stats["median"], linestyle="--", color="#e67e22", linewidth=2, label="Median AQI")
    plt.fill_between(
        hourly_stats.index,
        hourly_stats["mean"] - 0.5 * hourly_stats["std"],
        hourly_stats["mean"] + 0.5 * hourly_stats["std"],
        color="#2980b9",
        alpha=0.15,
        label="±0.5 Std Dev",
    )

    plt.title("Diurnal (24-Hour) Cycle of AQI in Karachi", fontsize=12, fontweight="bold")
    plt.xlabel("Hour of Day (Local Time)", fontsize=10)
    plt.ylabel("US AQI", fontsize=10)
    plt.xticks(range(0, 24))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    out_path = PLOTS_DIR / "eda_diurnal_cycle.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_seasonal_trends(df: pd.DataFrame):
    df_copy = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_copy["time"]):
        df_copy["time"] = pd.to_datetime(df_copy["time"])

    df_copy["year_month"] = df_copy["time"].dt.to_period("M").astype(str)
    monthly_stats = df_copy.groupby("year_month")["us_aqi"].mean()

    plt.figure(figsize=(10, 5))
    plt.plot(monthly_stats.index, monthly_stats.values, marker="s", color="#8e44ad", linewidth=2.5)

    plt.title("Monthly Average AQI Trend in Karachi", fontsize=12, fontweight="bold")
    plt.xlabel("Month", fontsize=10)
    plt.ylabel("Average US AQI", fontsize=10)
    plt.xticks(rotation=30, ha="right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_path = PLOTS_DIR / "eda_seasonal_trends.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    print(f"Loading dataset from {HISTORICAL_PARQUET}...")
    df = pd.read_parquet(HISTORICAL_PARQUET)
    print(f"Loaded {len(df)} records across {len(df.columns)} columns")

    print("Generating EDA visualizations...")
    plot_aqi_distribution(df)
    plot_correlation_heatmap(df)
    plot_diurnal_cycle(df)
    plot_seasonal_trends(df)
    print("All EDA plots generated successfully in plots/")


if __name__ == "__main__":
    main()
