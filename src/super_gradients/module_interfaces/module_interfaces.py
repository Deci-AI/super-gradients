from typing_extensions import Protocol, runtime_checkable


@runtime_checkable
class HasPreprocessingParams(Protocol):
    """
    Protocol interface for torch datasets that support getting preprocessing params, later to be passed to a model
    that obeys NeedsPreprocessingParams. This interface class serves a purpose of explicitly indicating whether a torch dataset has
    get_dataset_preprocessing_params implemented.

    """

    def get_dataset_preprocessing_params(self):
        ...


@runtime_checkable
class HasPredict(Protocol):
    """
    Protocol class serves a purpose of explicitly indicating whether a torch model has the functionality of ".predict"
    as defined in SG.

    """

    def set_dataset_processing_params(self, *args, **kwargs):
        """Set the processing parameters for the dataset."""
        ...

    def predict(self, images, *args, **kwargs):
        ...

    def predict_webcam(self, *args, **kwargs):
        ...
