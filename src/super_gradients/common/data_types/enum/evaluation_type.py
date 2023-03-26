from enum import Enum


class EvaluationType(str, Enum):
    """
    EvaluationType

    Passed to Trainer.evaluate(..), and controls which phase callbacks should be triggered (if at all).
    """

    TEST = "TEST"
    """Evaluate on Test set."""

    VALIDATION = "VALIDATION"
    """Evaluate on Validation set."""
