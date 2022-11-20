import scipy.io
from PIL import Image

from super_gradients.training.datasets.segmentation_datasets.pascal_voc_segmentation import PascalVOC2012SegmentationDataSet


class PascalAUG2012SegmentationDataSet(PascalVOC2012SegmentationDataSet):
    """
    PascalAUG2012SegmentationDataSet - Segmentation Data Set Class for Pascal AUG 2012 Data Set
    """

    def __init__(self, *args, **kwargs):
        self.sample_suffix = '.jpg'
        self.target_suffix = '.mat'
        super().__init__(sample_suffix=self.sample_suffix, target_suffix=self.target_suffix, *args, **kwargs)

    @staticmethod
    def target_loader(target_path: str) -> Image:
        """
        target_loader
            :param target_path: The path to the target data
            :return:            The loaded target
        """
        mat = scipy.io.loadmat(target_path, mat_dtype=True, squeeze_me=True,
                               struct_as_record=False)
        mask = mat['GTcls'].Segmentation
        return Image.fromarray(mask)
