import importlib
import inspect


from super_gradients.common.abstractions.abstract_logger import get_logger

logger = get_logger(__name__)

imported_albumentations_failure = None

try:
    from albumentations import BasicTransform, BaseCompose
except (ImportError, NameError, ModuleNotFoundError) as import_err:
    logger.debug("Failed to import albumentations")
    imported_albumentations_failure = import_err

if imported_albumentations_failure is None:
    ALBUMENTATIONS_TRANSFORMS = {
        name: cls for name, cls in inspect.getmembers(importlib.import_module("albumentations"), inspect.isclass) if issubclass(cls, BasicTransform)
    }
    ALBUMENTATIONS_TRANSFORMS.update(
        {name: cls for name, cls in inspect.getmembers(importlib.import_module("albumentations.pytorch"), inspect.isclass) if issubclass(cls, BasicTransform)}
    )

    ALBUMENTATIONS_COMP_TRANSFORMS = {
        name: cls
        for name, cls in inspect.getmembers(importlib.import_module("albumentations.core.composition"), inspect.isclass)
        if issubclass(cls, BaseCompose)
    }
    ALBUMENTATIONS_TRANSFORMS.update(ALBUMENTATIONS_COMP_TRANSFORMS)

else:
    ALBUMENTATIONS_TRANSFORMS = None
    ALBUMENTATIONS_COMP_TRANSFORMS = None
