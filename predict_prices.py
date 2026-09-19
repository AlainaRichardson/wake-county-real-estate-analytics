import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def train_housing_model(data_path):
    """Trains a Random Forest model to predict Wake County house values."""
    print("🤖 Initializing Machine Learning Pipeline...")

    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"❌ Error: Dataset not found at {data_path}")
        return

    # --- 1. Select Features and Target Variable ---
    # We choose core physical features that heavily impact pricing values
    feature_cols = ["total_living_area", "acreage", "property_age", "beds", "baths"]
    target_col = "sale_price"

    # Verify that all chosen features exist in the dataset
    missing_cols = [col for col in feature_cols + [target_col] if col not in df.columns]
    if missing_cols:
        print(f"❌ Error: Missing expected columns in data: {missing_cols}")
        print("Please run your clean_data.py pipeline first.")
        return

    X = df[feature_cols]
    y = df[target_col]

    # --- 2. Train / Test Split ---
    # Reserve 20% of data to validate how the model handles unseen properties
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"📉 Training sample size: {X_train.shape[0]} | Testing sample size: {X_test.shape[0]}")

    # --- 3. Model Training ---
    print("🌲 Training Random Forest Regressor Model (This may take a moment)...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # --- 4. Model Evaluation ---
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    print("\n📊 --- Model Evaluation Results ---")
    print(f"✅ R² Score (Variance Explained): {r2:.2f}")
    print(f"✅ Mean Absolute Error (MAE): ${mae:,.2f}")

    # --- 5. Feature Importance Breakdown ---
    print("\n🔍 --- Feature Importance Hierarchy ---")
    importances = model.feature_importances_
    for name, importance in sorted(
        zip(feature_cols, importances), key=lambda x: x[1], reverse=True
    ):
        print(f"⭐ {name.replace('_', ' ').title()}: {importance * 100:.1f}% impact")


if __name__ == "__main__":
    train_housing_model("data/wake_county_real_estate_cleaned.csv")
