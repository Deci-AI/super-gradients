from typing import Union, Optional

from torch import nn

from super_gradients.training.utils.utils import HpmStruct
from super_gradients.training.utils.sg_trainer_utils import get_callable_param_names
from super_gradients.training.models.arch_params_factory import get_arch_params


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
    def require_unpacked_arch_params(cls):
        """Check if the class required a single params names "arch_params" of type HpmStruct, or **kwargs instead."""
        return "arch_params" not in get_callable_param_names(cls.__init__)

    @classmethod
    def get_default_config_name(cls) -> Optional[str]:
        """Return the name of the default config name - i.e. which includes the default arch_params of this architecture.
        If None, it will be assumed that the arch_params of this architecture don't have default values."""
        return None

    @classmethod
    def from_recipe(cls, arch_params: HpmStruct, default_config_name: Optional[str] = None, recipes_dir_path: Optional[str] = None) -> HpmStruct:
        """Instantiate the class using recipe as default arch_params, and overriding it with input arch_params.
        :param arch_params:                 arch_params specified by the user.
        :param default_config_name:         Name of the yaml to use to get default values (e.g. "resnet18_cifar_arch_params")
        :param recipes_dir_path:            Optional. Main directory where every recipe are stored. (e.g. ../super_gradients/recipes)
                                            This directory should include a "arch_params" folder,
                                            which itself should include the config file named after config_name.
        :return: Instance of SgModule

        Example:
            Instantiate MyModel based on arch_params, and using /home/recipes/arch_params/default_mymodel.yaml as default values
            >>> model = MyModel.from_recipe(arch_params=arch_params, default_config_name="default_mymodel", recipes_dir_path='/home/recipes')
        """
        default_config_name = default_config_name or cls.get_default_config_name()

        if default_config_name:
            arch_params = HpmStruct(
                **get_arch_params(config_name=default_config_name, recipes_dir_path=recipes_dir_path, overriding_params=arch_params.to_dict())
            )

        if cls.require_unpacked_arch_params():
            return cls(**arch_params.to_dict(include_schema=False))
        else:
            return cls(arch_params=arch_params)
