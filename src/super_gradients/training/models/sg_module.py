from typing import Union

from torch import nn

from super_gradients.training.utils.utils import HpmStruct
from super_gradients.training.utils.sg_trainer_utils import get_callable_param_names


class SgModule(nn.Module):
    def initialize_param_groups(self, lr: float, training_params: HpmStruct) -> list:
        """

        :return: list of dictionaries containing the key 'named_params' with a list of named params
        """
        return [{"named_params": self.named_parameters()}]

    def update_param_groups(self, param_groups: list, lr: float, epoch: int, iter: int, training_params: HpmStruct, total_batch: int) -> list:
        """

        :param param_groups: list of dictionaries containing the params
        :return: list of dictionaries containing the params
        """
        for param_group in param_groups:
            param_group["lr"] = lr
        return param_groups

    def get_include_attributes(self) -> list:
        """
        This function is used by the EMA. When updating the EMA model, some attributes of the main model (used in training)
        are updated to the EMA model along with the model weights.
        By default, all attributes are updated except for private attributes (starting with '_')
        You can either set include_attributes or exclude_attributes. By returning a non empty list from this function,
        you override the default behaviour and only attributes named in this list will be updated.
        Note: This will also override the get_exclude_attributes list.
            :return: list of attributes to update from main model to EMA model
        """
        return []

    def get_exclude_attributes(self) -> list:
        """
        This function is used by the EMA. When updating the EMA model, some attributes of the main model (used in training)
        are updated to the EMA model along with the model weights.
        By default, all attributes are updated except for private attributes (starting with '_')
        You can either set include_attributes or exclude_attributes. By returning a non empty list from this function,
        you override the default behaviour and attributes named in this list will also be excluded from update.
        Note: if get_include_attributes is not empty, it will override this list.
            :return: list of attributes to not update from main model to EMA mode
        """
        return []

    def prep_model_for_conversion(self, input_size: Union[tuple, list] = None, **kwargs):
        """
        Prepare the model to be converted to ONNX or other frameworks.
        Typically, this function will freeze the size of layers which is otherwise flexible, replace some modules
        with convertible substitutes and remove all auxiliary or training related parts.
        :param input_size: [H,W]
        """

    def replace_head(self, **kwargs):
        """
        Replace final layer for pretrained models. Since this varies between architectures, we leave it to the inheriting
        class to implement.
        """

        raise NotImplementedError

    @classmethod
    def load_default_arch_params(cls) -> HpmStruct:  # FIXME: maybe name load_recipe_arch_params ?
        """Placeholder allowing to define default arch_params for every model. By default doenst provide any default arch_params.

        Example of implementation for Unet:
        >>> @classmethod
        >>> def load_default_arch_params(cls) -> HpmStruct:
        >>>     return HpmStruct(**get_arch_params("unet_default_arch_params"))
        """
        return HpmStruct()

    @classmethod
    def require_unpacked_arch_params(cls):
        """Check if the class required a single params names "arch_params" of type HpmStruct, or **kwargs instead."""
        return "arch_params" not in get_callable_param_names(cls.__init__)

    @classmethod
    def from_recipe(
        cls, arch_params: HpmStruct
    ):  # FIXME: not sure if name is good, because if 'load_default_arch_params' not override, it is not really loaded from a recipe...
        """Instantiate the class using recipe as default arch_params, and overriding it with input arch_params.
        :param arch_params: arch_params specified by the user.
        :return: Instance of SgModule
        """
        arch_params_with_default = cls.load_default_arch_params()
        arch_params_with_default.override(**arch_params.to_dict())

        if cls.require_unpacked_arch_params():
            return cls(**arch_params.to_dict(include_schema=False))
        else:
            return cls(arch_params=arch_params)
