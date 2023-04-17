from enum import Enum


class StrictLoad(Enum):
    """Wrapper for adding more functionality to torch's strict_load parameter in load_state_dict()."""

    OFF = False
    """Native torch "strict_load = off" behaviour. See nn.Module.load_state_dict() documentation for more details."""

    ON = True
    """Native torch "strict_load = on" behaviour. See nn.Module.load_state_dict() documentation for more details."""

    NO_KEY_MATCHING = "no_key_matching"
    """Allows the usage of SuperGradient's adapt_checkpoint function, which loads a checkpoint by matching each
    layer's shapes (and bypasses the strict matching of the names of each layer (ie: disregards the state_dict key matching)).
    This implementation assumes order of layers in the state_dict and model are the same since it goes layer by layer and as name
    suggest does not use key matching, relying only on index of each weight.
    """

    KEY_MATCHING = "key_matching"
    """Loose load strategy that loads the state dict from checkpoint into model only for common keys and also handling the
    case when shapes of the tensors in the state dict and model are different for the same key (Such layers will be skipped)."""
