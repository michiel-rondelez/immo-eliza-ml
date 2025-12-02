"""
Simple configuration presets for preprocessing.
Makes it easy to reuse in APIs and other projects.
"""

from .preprocessing import FeaturePreprocessor


class PreprocessorConfig:
    """Simple presets for common use cases."""

    @staticmethod
    def default():
        """Default config with capping enabled."""
        return FeaturePreprocessor(
            use_capping=True,
            capping_percentiles=(1, 99)
        )

    @staticmethod
    def no_capping():
        """Config without outlier capping."""
        return FeaturePreprocessor(
            use_capping=False
        )

    @staticmethod
    def strict_capping():
        """Stricter outlier capping (0.5%, 99.5%)."""
        return FeaturePreprocessor(
            use_capping=True,
            capping_percentiles=(0.5, 99.5)
        )

    @staticmethod
    def custom(features=None, target="price", use_capping=True,
               capping_percentiles=(1, 99), log_target=True):
        """Custom configuration."""
        return FeaturePreprocessor(
            features=features,
            target=target,
            use_capping=use_capping,
            capping_percentiles=capping_percentiles,
            log_target=log_target
        )

    @staticmethod
    def from_json(path):
        """Load from saved JSON config."""
        return FeaturePreprocessor.load(path)


# Simple API-ready functions
def create_preprocessor(config="default"):
    """
    Create preprocessor with preset config.

    Args:
        config: "default", "no_capping", "strict_capping", or dict with custom params

    Returns:
        FeaturePreprocessor instance

    Usage:
        # In your API
        prep = create_preprocessor("default")
        prep.fit_transform(train_df)
        prep.save("models/preprocessor.json")

        # Later in API
        prep = load_preprocessor("models/preprocessor.json")
        X = prep.transform(new_data)
    """
    if isinstance(config, dict):
        return PreprocessorConfig.custom(**config)
    elif config == "default":
        return PreprocessorConfig.default()
    elif config == "no_capping":
        return PreprocessorConfig.no_capping()
    elif config == "strict_capping":
        return PreprocessorConfig.strict_capping()
    else:
        raise ValueError(f"Unknown config: {config}")


def load_preprocessor(path):
    """
    Load preprocessor from JSON.

    Usage:
        prep = load_preprocessor("models/preprocessor.json")
        X = prep.transform(df)
    """
    return PreprocessorConfig.from_json(path)
