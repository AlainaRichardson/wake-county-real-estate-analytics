import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set a professional visual style for portfolio presentation
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


def generate_portfolio_visuals(cleaned_data_path, output_dir="images"):
    """Generates and saves the core EDA charts required for the project README."""
    print("📊 Loading cleaned dataset for visual profiling...")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Created directory: {output_dir}")

    try:
        df = pd.read_csv(cleaned_data_path)
    except FileNotFoundError:
        print(f"❌ Error: Cleaned data file not found at {cleaned_data_path}")
        return

    # --- Chart 1: Feature Correlation Heatmap ---
    print("📈 Plotting Feature Correlation Heatmap...")
    plt.figure(figsize=(8, 6))
    numeric_df = df.select_dtypes(include=["number"])
    corr_matrix = numeric_df.corr()

    sns.heatmap(corr_matrix, annot=True, cmap="CoolWarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix of Wake County Housing Features", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_correlation_matrix.png", dpi=300)
    plt.close()

    # --- Chart 2: Square Footage vs Sale Price ---
    print("📉 Plotting Square Footage vs Sale Price Scatter...")
    plt.figure()
    sns.scatterplot(
        data=df,
        x="total_living_area",
        y="sale_price",
        alpha=0.4,
        color="#2b5c8f",
        edgecolor=None,
    )
    sns.regplot(
        data=df,
        x="total_living_area",
        y="sale_price",
        scatter=False,
        color="#d95f02",
        line_kws={"linewidth": 2},
    )

    plt.title("Impact of Total Living Area on Housing Sale Price", fontsize=14, pad=15)
    plt.xlabel("Total Living Area (Sq Ft)", fontsize=12)
    plt.ylabel("Sale Price ($)", fontsize=12)
    plt.gca().yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f"${x:,.0f}")
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/living_area_vs_price.png", dpi=300)
    plt.close()

    print(f"🎉 Success! Portfolio visuals saved inside the '{output_dir}/' folder.")


if __name__ == "__main__":
    generate_portfolio_visuals("data/wake_county_real_estate_cleaned.csv")
