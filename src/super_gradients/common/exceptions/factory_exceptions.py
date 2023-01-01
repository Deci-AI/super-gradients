from typing import List
from rapidfuzz import process, fuzz


class UnknownTypeException(Exception):
    """Type error with message, followed by type suggestion, chosen by fuzzy matching
     (out of 'choices' arg passed in __init__).

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, unknown_type: str, choices: List, message: str = None):
        message = message or f"Unknown object type: {unknown_type} in configuration. valid types are: {choices}"
        if isinstance(unknown_type, str):
            choice, score, _ = process.extractOne(unknown_type, choices, scorer=fuzz.WRatio)
            if score > 70:
                err_msg_tip = f"\n Did you mean: {choice}?"
        else:
            err_msg_tip = ""
        self.message = message + err_msg_tip
        super().__init__(self.message)
