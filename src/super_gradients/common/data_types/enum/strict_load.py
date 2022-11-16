from enum import Enum


class StrictLoad(Enum):
    """
    Wrapper for adding more functionality to torch's strict_load parameter in load_state_dict().
    Attributes:
        OFF              - Native torch "strict_load = off" behaviour. See nn.Module.load_state_dict() documentation for more details.
        ON               - Native torch "strict_load = on" behaviour. See nn.Module.load_state_dict() documentation for more details.
        NO_KEY_MATCHING  - Allows the usage of SuperGradient's adapt_checkpoint function, which loads a checkpoint by matching each
                           layer's shapes (and bypasses the strict matching of the names of each layer (ie: disregards the state_dict key matching)).
    """

    OFF = False
    ON = True
    NO_KEY_MATCHING = "no_key_matching"
