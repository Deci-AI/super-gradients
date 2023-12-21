from typing import Tuple, Optional, Union

import torch


class SupportsInputShapeCheck:
    def validate_input_shape(self, input_shape: Union[torch.Size, Tuple[int, ...]]) -> None:
        """
        Validate the input shape of the model.
        If the validation passes, the method should return None.
        If the validation fails, the method should raise a ValueError.

        :param input_shape: (Tuple[int, ...]) Input shape (usually BCHW)
        :return: None.
        """
        input_shape = tuple(input_shape)
        exact_shape = self.get_exact_input_shape_size()
        if exact_shape is not None:
            # Take n last elements of input shape, where n is the number of elements in exact_shape
            shape_to_check = input_shape[-len(exact_shape) :]
            if any(shape_to_check != exact_shape):
                raise ValueError(f"Input shape {input_shape} is not supported. Model requires input with spatial size of {exact_shape}.")

        min_shape = self.get_minimum_input_shape_size()
        if min_shape is not None:
            shape_to_check = input_shape[-len(min_shape) :]
            if any([actual < expected for actual, expected in zip(shape_to_check, min_shape)]):
                raise ValueError(f"Input shape {input_shape} is not supported. Model requires input with spatial size of at least {min_shape}.")

        shape_steps = self.get_input_shape_steps()
        if shape_steps is not None:
            shape_to_check = input_shape[-len(shape_steps) :]
            if any([actual % expected != 0 for actual, expected in zip(shape_to_check, shape_steps)]):
                raise ValueError(f"Input shape {input_shape} is not supported. Model requires input with spatial size that is a multiple of {shape_steps}.")

    def get_exact_input_shape_size(self) -> Optional[Tuple[int, ...]]:
        """
        Get the exact input shape of the model.
        Class should override this method if the model requires an exact input shape.
        Usually true for transformer-based models.
        :return:
        """
        return None

    def get_minimum_input_shape_size(self) -> Optional[Tuple[int, ...]]:
        """
        Get the minimum input shape of the model.
        Class should override this method if the model requires a minimum input shape.
        Usually true for transformer-based models.
        :return:
        """
        return None

    def get_input_shape_steps(self) -> Optional[Tuple[int, ...]]:
        """
        Get the smallest increment between two valid input shapes.
        If the model requires a certain size of input shape, this method should return the smallest increment
        between two valid input shapes.
        For instance, if the model can operate on (256x256), (256+32x256+32), (256+64x256+64), ... this method
        should return (32, 32).
        :return:
        """
        return None
