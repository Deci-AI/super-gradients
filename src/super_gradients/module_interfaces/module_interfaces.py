from abc import abstractmethod


class HasPreprocessingParams:
    """
    Protocol interface for torch datasets that support getting preprocessing params, later to be passed to a model
    that inherits (multiple inheritance, also from torch.utils.data Dataset) from NeedsPreprocessingParams. This
    interface class serves a purpose of explicitly indicating whether a torch dataset has
    get_dataset_preprocessing_params implemented.

    """

    @abstractmethod
    def get_dataset_preprocessing_params(self):
        pass


class HasPredict:
    """
    Interface class serves a purpose of explicitly indicating whether a torch model has the functionality of ".predict"
    as defined in SG.

    """

    @abstractmethod
    def set_dataset_processing_params(self, *args, **kwargs) -> None:
        """Set the processing parameters for the dataset."""
        pass

    @abstractmethod
    def predict(self, images, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def predict_webcam(self, *args, **kwargs) -> None:
        pass
