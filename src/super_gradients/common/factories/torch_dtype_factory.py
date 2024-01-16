import torch
from super_gradients.common.factories.type_factory import TypeFactory


class TorchDtypeFactory(TypeFactory):
    """
    Factory to return PyTorch dtype from configuration string.
    """

    def __init__(self):
        super().__init__(self._get_torch_dtypes())

    def _get_torch_dtypes(self):
        """
        Get a dictionary of PyTorch dtypes.
        """
        return {
            "float32": torch.float32,
            "float64": torch.float64,
            "float16": torch.float16,
            "int8": torch.int8,
            "int16": torch.int16,
            "int32": torch.int32,
            "int64": torch.int64,
            "uint8": torch.uint8,
            "long": torch.long,
        }
