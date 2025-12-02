import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from immo_eliza_ml.clean import CleanData
from immo_eliza_ml.preprocessing import FeaturePreprocessor
from immo_eliza_ml.trainer import ModelTrainer
from immo_eliza_ml.visuals import Visualizations
from immo_eliza_ml.predict import Predict


def main():

    print("\n=== IMMO ELIZA ML TRAINING PIPELINE ===\n")

    # ---------------------------
    # 1. Load raw data
    # ---------------------------
    print("Loading raw dataset...")
    df = pd.read_csv("data/1_raw/raw_data.csv")
    print(f"Loaded {len(df)} rows\n")

    # ---------------------------
    # 2. Clean data
    # ---------------------------
    print("Cleaning dataset...")
    cleaner = CleanData()
    df = cleaner.clean(df)
    df.to_csv("data/2_cleaned/cleaned_data.csv", index=False)
    print(f"  {len(df)} rows after cleaning")
    print("  Saved to data/2_cleaned/cleaned_data.csv\n")


    # ---------------------------
    # 3. Split data (BEFORE preprocessing)
    # ---------------------------
    print("Splitting dataset...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Train: {len(train_df)}, Test: {len(test_df)}\n")

    # ---------------------------
    # 4. Preprocess features
    # ---------------------------
    print("Preprocessing features...")
    
    # NUMERIC = {
    #     "number_of_rooms", "living_area", "number_of_facades",
    #     "garden_surface", "terrace_surface", "postal_code",
    #     "total_outdoor", "outdoor_ratio", "luxury_score",
    #     "area_log", "area_per_room",
    # }

    # CATEGORICAL = {"subtype_of_property", "state_of_building", "region"}

    # BINARY = {"equipped_kitchen", "furnished", "open_fire", 
    #           "terrace", "garden", "swimming_pool"}

    prep = FeaturePreprocessor()
    # prep = FeaturePreprocessor(features={"number_of_rooms", "living_area", "number_of_facades"})
    prep.info()
    
    X_train, y_train = prep.fit_transform(train_df)
    X_test = prep.transform(test_df)
    y_test = prep.get_target(test_df)

    print(f"\nX_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}\n")

    # ---------------------------
    # 5. Train models
    # ---------------------------
    print("Training models...")
    trainer = ModelTrainer()
    trainer.train(X_train, X_test, y_train, y_test)

    # ---------------------------
    # 6. Results
    # ---------------------------
    trainer.summary()
    trainer.overfitting_summary()
    trainer.detailed_performance_report()
    
    # ---------------------------
    # 7. Save artifacts
    # ---------------------------
    print("\nSaving artifacts...")
    
    # Create folders
    os.makedirs("models", exist_ok=True)
    os.makedirs("predictions", exist_ok=True)
    os.makedirs("data/3_processed", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # Save preprocessor and models
    prep.save("models/preprocessor.json")
    trainer.save_training_models("models")  # Save fitted models as .pkl
    trainer.save_model_params_to_json("models")  # Save model parameters as .json
    trainer.save_detailed_report_json("models")  # Save detailed performance report with parameters
    trainer.save_predictions_models("predictions")
    
    # Save y values for later use
    y_values_data = {
        "y_train": y_train.tolist() if hasattr(y_train, 'tolist') else list(y_train),
        "y_test": y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test)
    }
    with open("data/3_processed/y_values.json", 'w') as f:
        json.dump(y_values_data, f, indent=2)
    print("Saved y values to data/3_processed/y_values.json")


    # ---------------------------
    # 8. Prediction
    # ---------------------------



    # Load models
    predictor = Predict().load()

    # Define property
    my_house = {
        "postal_code": 9000,
        "living_area": 120,
        "number_of_rooms": 3,
        "number_of_facades": 2,
        "equipped_kitchen": 1,
        "swimming_pool": 0,
        "garden": 1,
        "garden_surface": 150,
    }

    # Quick prediction
    price = predictor.predict_single(my_house)

    print("Quick price prediction:", price)
    # Compare all models
    predictor.display_all_predictions(my_house)


    # ---------------------------
    # 9. Visualization
    # ---------------------------
    print("\n=== PREDICTIONS & VISUALIZATION ===\n")
    
    pred = Visualizations()
    pred.load_all(
        models_folder="models",
        predictions_folder="predictions",
        data_folder="data/3_processed"
    )
    
    # Get best model
    best_name, best_r2 = pred.get_best_model()
    print(f"\nBest model: {best_name} (R² = {best_r2:.4f})")
    
    # Create all plots
    pred.plot_all(folder="plots")
    
    # View predictions as DataFrame
    df_preds = pred.predictions_to_dataframe("test")
    df_preds.to_csv("predictions/all_predictions.csv", index=False)
    print("\nSaved predictions to predictions/all_predictions.csv")
    
    print("\n=== PIPELINE COMPLETE ===\n")

if __name__ == "__main__":
    main()