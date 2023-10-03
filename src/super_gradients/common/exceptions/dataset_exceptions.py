from typing import Tuple, Type


class DatasetValidationException(Exception):
    pass


class ParameterMismatchException(DatasetValidationException):
    pass


class IllegalDatasetParameterException(DatasetValidationException):
    """
    Exception raised illegal dataset param.

    :param desc: Explanation of the error
    """

    def __init__(self, desc: str):
        self.message = "Unsupported dataset parameter format: " + desc
        super().__init__(self.message)


class EmptyDatasetException(DatasetValidationException):
    """
    Exception raised when a dataset does not have any image for a specific config

    :param desc: explanation of the error
    """

    def __init__(self, desc: str):
        self.message = "Empty Dataset: " + desc
        super().__init__(self.message)


class UnsupportedBatchItemsFormat(ValueError):
    """Exception raised illegal batch items returned from data loader.

    :param batch_items: batch items returned from data loader
    """

    def __init__(self, batch_items: tuple):
        self.message = (
            f"The data loader is expected to return 2 to 3 items, but got {len(batch_items)} instead.\n"
            "Items expected:\n"
            "   - inputs = batch_items[0] # model input - The type might depend on the model you are using.\n"
            "   - targets = batch_items[1] # Target that will be used to compute loss/metrics - The type might depend on the function you are using.\n"
            "   - [OPTIONAL] additional_batch_items = batch_items[2] # Dict made of any additional item that you might want to use.\n"
            "To fix this, please change the implementation of your dataset __getitem__ method, so that it would return the items defined above.\n"
        )
        super().__init__(self.message)


class DatasetItemsException(Exception):
    def __init__(self, data_sample: Tuple, collate_type: Type, expected_item_names: Tuple):
        """
        :param data_sample:         item(s) returned by a dataset
        :param collate_type:        type of the collate that caused the exception
        :param expected_item_names: tuple of names of items that are expected by the collate to be returned from the dataset
        """
        collate_type_name = collate_type.__name__
        num_sample_items = len(data_sample) if isinstance(data_sample, tuple) else 1
        error_msg = f"`{collate_type_name}` only supports Datasets that return a tuple {expected_item_names}, but got a tuple of len={num_sample_items}"
        super().__init__(error_msg)
