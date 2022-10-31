from enum import Enum


class EvaluationType(str, Enum):
    """
    EvaluationType

    Passed to Trainer.evaluate(..), and controls which phase callbacks should be triggered (if at all).

        Attributes:
            TEST
            VALIDATION

    """
    TEST = 'TEST'
    VALIDATION = 'VALIDATION'
