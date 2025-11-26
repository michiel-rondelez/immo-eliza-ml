import pandas as pd
from immo_eliza_ml.clean import CleanData
from immo_eliza_ml.preprocessing import FeaturePreprocessor
from immo_eliza_ml.trainer import ModelTrainer


def main():

    print("\n=== IMMO ELIZA ML TRAINING PIPELINE ===\n")

    # ---------------------------
    # 1. Load raw data
    # ---------------------------
    print("➡ Loading raw dataset...")
    df = pd.read_csv("data/1_raw/raw_data.csv")

    # ---------------------------
    # 2. Clean data
    # ---------------------------
    print("➡ Cleaning dataset...")
    cleaner = CleanData()
    df = cleaner.clean(df)

    df.to_csv("data/2_cleaned/cleaned_data.csv", index=False)
    print("✔ Cleaned data saved to data/2_cleaned/cleaned_data.csv\n")

    # ---------------------------
    # 3. Preprocess data
    # ---------------------------
    print("➡ Preprocessing dataset...")
    pre = FeaturePreprocessor()
    X, y = pre.fit_transform(df)

    print("✔ Preprocessing completed.\n")

    # ---------------------------
    # 4. Train models (Baseline)
    # ---------------------------
    trainer = ModelTrainer()

    print("➡ Training baseline models...")
    baseline_results = trainer.train_baseline_models(X, y)

    print("\n=== BASELINE MODEL SCORES ===")
    for name, score in baseline_results.items():
        print(f"{name}: R² = {score:.4f}")

    # ---------------------------
    # 5. Grid Search for best models
    # ---------------------------
    print("\n➡ Running Grid Search (this may take time)...")
    grid_results = trainer.grid_search_all(X, y)

    print("\n=== GRID SEARCH RESULTS ===")
    print(grid_results)

    print(f"\n🏆 Best Model Selected: {trainer.best_model_name}")

    # ---------------------------
    # 6. Save artifacts
    # ---------------------------
    print("\n➡ Saving preprocessing pipeline & best models...")
    pre.save("models/preprocessor.pkl")
    trainer.save_models("models/")

    print("\n✅ Training complete. All artifacts saved in /models/\n")


if __name__ == "__main__":
    main()
