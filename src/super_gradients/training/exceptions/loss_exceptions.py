class IllegalRangeForLossAttributeException(Exception):
    """
    Exception raised illegal value (i.e not in range) for _Loss attribute.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, range_vals: tuple, attr_name: str):
        self.message = attr_name + " must be in range " + str(range_vals)
        super().__init__(self.message)


class RequiredLossComponentReductionException(Exception):
    """
    Exception raised illegal reduction for _Loss component.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, component_name: str, reduction: str, required_reduction: str):
        self.message = component_name + ".reduction must be " + required_reduction + ", got" + reduction
        super().__init__(self.message)
