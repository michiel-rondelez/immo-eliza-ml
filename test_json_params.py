"""
Test script to demonstrate JSON model parameter saving/loading.

This script shows how to:
1. Save model parameters (hyperparameters) to JSON
2. Load model parameters from JSON to recreate model architecture
"""

from src.immo_eliza_ml.trainer import ModelTrainer


def main():
    print("=" * 60)
    print("TESTING JSON MODEL PARAMETER SAVE/LOAD")
    print("=" * 60)

    # Create a ModelTrainer instance
    trainer = ModelTrainer()

    print("\nInitial models:")
    for name, model in trainer.models.items():
        print(f"  - {name}: {type(model).__name__}")

    # Save model parameters to JSON
    print("\n" + "-" * 60)
    print("STEP 1: Saving model parameters to JSON...")
    print("-" * 60)
    trainer.save_model_params_to_json(folder="test_models")

    # Create a new trainer instance
    print("\n" + "-" * 60)
    print("STEP 2: Creating new trainer and loading from JSON...")
    print("-" * 60)
    new_trainer = ModelTrainer()
    loaded_models = new_trainer.load_model_params_from_json(folder="test_models")

    # Verify the models were loaded correctly
    print("\n" + "-" * 60)
    print("STEP 3: Verifying loaded models...")
    print("-" * 60)
    for name, model in loaded_models.items():
        print(f"\n{name}:")
        print(f"  Type: {type(model).__name__}")
        print(f"  Parameters: {model.get_params()}")

    print("\n" + "=" * 60)
    print("✓ JSON parameter save/load test complete!")
    print("=" * 60)

    print("\nIMPORTANT NOTES:")
    print("- JSON files contain only hyperparameters (model configuration)")
    print("- To use trained models, you still need .pkl files with fitted weights")
    print("- JSON files are useful for:")
    print("  1. Documentation and version control of model configurations")
    print("  2. Recreating model architecture for retraining")
    print("  3. Human-readable format for model hyperparameters")


if __name__ == "__main__":
    main()
