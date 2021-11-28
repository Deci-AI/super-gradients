
class IllegalDatasetParameterException(Exception):
    """
    Exception raised illegal dataset param.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, desc):
        self.message = "Unsupported dataset parameter format: " + desc
        super().__init__(self.message)


class UnsupportedBatchItemsFormat(ValueError):
    """Exception raised illegal batch items returned from data loader.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self):
        self.message = "Batch items returned by the data loader expected format: \n" \
                       "1. torch.Tensor or tuple, s.t inputs = batch_items[0], targets = batch_items[1] and len(" \
                       "batch_items) = 2 \n" \
                       "2. tuple: (inputs, targets, additional_batch_items)"
        super().__init__(self.message)
