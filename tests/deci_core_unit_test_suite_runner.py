import sys
import unittest

from tests.integration_tests.ema_train_integration_test import EMAIntegrationTest
from tests.unit_tests import (
    ZeroWdForBnBiasTest,
    SaveCkptListUnitTest,
    TestAverageMeter,
    TestRepVgg,
    TestWithoutTrainTest,
    OhemLossTest,
    EarlyStopTest,
    SegmentationTransformsTest,
    TestConvBnRelu,
    FactoriesTest,
    InitializeWithDataloadersTest,
    TrainingParamsTest,
    TrainOptimizerParamsOverride,
    CallTrainTwiceTest,
    ResumeTrainingTest,
    CallTrainAfterTestTest,
    CrashTipTest,
    TestTransforms,
    TestPostPredictionCallback,
    TestModelPredict,
    TestDeprecationDecorator,
    DynamicModelTests,
    TestExportRecipe,
    TestMixedPrecisionDisabled,
    TestClassificationAdapter,
    TestDetectionAdapter,
    TestSegmentationAdapter,
)
from tests.end_to_end_tests import TestTrainer
from tests.unit_tests.depth_estimation_dataset_test import DepthEstimationDatasetTest
from tests.unit_tests.test_convert_recipe_to_code import TestConvertRecipeToCode
from tests.unit_tests.detection_utils_test import TestDetectionUtils
from tests.unit_tests.detection_dataset_test import DetectionDatasetTest, TestParseYoloLabelFile
from tests.unit_tests.export_detection_model_test import TestDetectionModelExport
from tests.unit_tests.export_onnx_test import TestModelsONNXExport
from tests.unit_tests.export_pose_estimation_model_test import TestPoseEstimationModelExport
from tests.unit_tests.extreme_batch_cb_test import ExtremeBatchSanityTest
from tests.unit_tests.load_checkpoint_test import LoadCheckpointTest
from tests.unit_tests.local_ckpt_head_replacement_test import LocalCkptHeadReplacementTest
from tests.unit_tests.max_batches_loop_break_test import MaxBatchesLoopBreakTest
from tests.unit_tests.multiple_ignore_indices_segmentation_metrics_test import TestSegmentationMetricsMultipleIgnored
from tests.unit_tests.pose_estimation_dataset_test import TestPoseEstimationDataset
from tests.unit_tests.pose_estimation_sample_test import PoseEstimationSampleTest
from tests.unit_tests.preprocessing_unit_test import PreprocessingUnitTest
from tests.unit_tests.quantization_utility_tests import QuantizationUtilityTest
from tests.unit_tests.random_erase_test import RandomEraseTest
from tests.unit_tests.replace_head_test import ReplaceHeadUnitTest
from tests.unit_tests.strictload_enum_test import StrictLoadEnumTest
from tests.unit_tests.test_deprecations import DeprecationsUnitTest
from tests.unit_tests.test_depth_estimation_metrics import TestDepthEstimationMetrics
from tests.unit_tests.test_finetune import TestFineTune
from tests.unit_tests.test_min_samples_single_node import TestMinSamplesSingleNode
from tests.unit_tests.test_model_weight_averaging import TestModelWeightAveraging
from tests.unit_tests.test_supports_check_input_shape import TestSupportsInputShapeCheck
from tests.unit_tests.test_train_with_torch_scheduler import TrainWithTorchSchedulerTest
from tests.unit_tests.test_version_check import TestVersionCheck
from tests.unit_tests.test_yolo_nas_pose import YoloNASPoseTests
from tests.unit_tests.train_with_intialized_param_args_test import TrainWithInitializedObjectsTest
from tests.unit_tests.pretrained_models_unit_test import PretrainedModelsUnitTest
from tests.unit_tests.lr_warmup_test import LRWarmupTest
from tests.unit_tests.kd_ema_test import KDEMATest
from tests.unit_tests.kd_trainer_test import KDTrainerTest
from tests.unit_tests.dice_loss_test import DiceLossTest
from tests.unit_tests.iou_loss_test import IoULossTest
from tests.unit_tests.vit_unit_test import TestViT
from tests.unit_tests.yolo_nas_tests import TestYOLONAS
from tests.unit_tests.yolox_unit_test import TestYOLOX
from tests.unit_tests.lr_cooldown_test import LRCooldownTest
from tests.unit_tests.detection_targets_format_transform_test import DetectionTargetsTransformTest
from tests.unit_tests.forward_pass_prep_fn_test import ForwardpassPrepFNTest
from tests.unit_tests.mask_loss_test import MaskAttentionLossTest
from tests.unit_tests.detection_sub_sampling_test import TestDetectionDatasetSubsampling
from tests.unit_tests.detection_sub_classing_test import TestDetectionDatasetSubclassing
from tests.unit_tests.detection_output_adapter_test import TestDetectionOutputAdapter
from tests.unit_tests.multi_scaling_test import MultiScaleTest
from tests.unit_tests.ppyoloe_unit_test import TestPPYOLOE
from tests.unit_tests.bbox_formats_test import BBoxFormatsTest
from tests.unit_tests.config_inspector_test import ConfigInspectTest
from tests.unit_tests.repvgg_block_tests import TestRepVGGBlock
from tests.unit_tests.training_utils_test import TestTrainingUtils
from tests.unit_tests.dekr_loss_test import DEKRLossTest
from tests.unit_tests.pose_estimation_metrics_test import TestPoseEstimationMetrics
from tests.unit_tests.forward_with_sliding_window_test import SlidingWindowTest
from tests.unit_tests.detection_metrics_distance_based_test import TestDetectionMetricsDistanceBased


class CoreUnitTestSuiteRunner:
    def __init__(self):
        self.test_loader = unittest.TestLoader()
        self.unit_tests_suite = unittest.TestSuite()
        self._add_modules_to_unit_tests_suite()
        self.end_to_end_tests_suite = unittest.TestSuite()
        self._add_modules_to_end_to_end_tests_suite()
        self.test_runner = unittest.TextTestRunner(verbosity=3, stream=sys.stdout)

    def _add_modules_to_unit_tests_suite(self):
        """
        _add_modules_to_unit_tests_suite - Adds unit tests to the Unit Tests Test Suite
            :return:
        """
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(CrashTipTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(SaveCkptListUnitTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(ZeroWdForBnBiasTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(TestAverageMeter))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(TestRepVgg))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(TestWithoutTrainTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(StrictLoadEnumTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(TrainWithInitializedObjectsTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(RandomEraseTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(OhemLossTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(EarlyStopTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(SegmentationTransformsTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(PretrainedModelsUnitTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(LRWarmupTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(TestConvBnRelu))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(FactoriesTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(TestDetectionUtils))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(DiceLossTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(TestViT))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(KDEMATest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(KDTrainerTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(TestYOLOX))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(InitializeWithDataloadersTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(LRCooldownTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(DetectionTargetsTransformTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(ForwardpassPrepFNTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(MaskAttentionLossTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(IoULossTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(TestDetectionDatasetSubsampling))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(TestDetectionDatasetSubclassing))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(QuantizationUtilityTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(MultiScaleTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(TrainingParamsTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(CallTrainTwiceTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(TrainOptimizerParamsOverride))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(BBoxFormatsTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(ResumeTrainingTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(CallTrainAfterTestTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(ConfigInspectTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(TestDetectionOutputAdapter))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(TestRepVGGBlock))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(LocalCkptHeadReplacementTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(DetectionDatasetTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(TestParseYoloLabelFile))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(TestModelsONNXExport))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(MaxBatchesLoopBreakTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(TestTrainingUtils))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(TestTransforms))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(TestPPYOLOE))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(DEKRLossTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(TestPoseEstimationMetrics))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(TestPoseEstimationDataset))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(LoadCheckpointTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(ReplaceHeadUnitTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(PreprocessingUnitTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(TestYOLONAS))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(DeprecationsUnitTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(TestMinSamplesSingleNode))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(TestPostPredictionCallback))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(TestSegmentationMetricsMultipleIgnored))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(TrainWithTorchSchedulerTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(ExtremeBatchSanityTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(TestModelPredict))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(TestDetectionModelExport))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(SlidingWindowTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(TestDeprecationDecorator))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(TestPoseEstimationModelExport))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(YoloNASPoseTests))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(PoseEstimationSampleTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(TestExportRecipe))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(TestMixedPrecisionDisabled))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(DynamicModelTests))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(TestConvertRecipeToCode))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(TestVersionCheck))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(TestModelWeightAveraging))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(TestClassificationAdapter))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(TestDetectionAdapter))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(TestSegmentationAdapter))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(TestDetectionMetricsDistanceBased))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(TestFineTune))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(TestDepthEstimationMetrics))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(DepthEstimationDatasetTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(TestSupportsInputShapeCheck))

    def _add_modules_to_end_to_end_tests_suite(self):
        """
        _add_modules_to_end_to_end_tests_suite - Adds end to end tests to the Unit Tests Test Suite
            :return:
        """
        self.end_to_end_tests_suite.addTest(self.test_loader.loadTestsFromModule(TestTrainer))
        self.end_to_end_tests_suite.addTest(self.test_loader.loadTestsFromModule(EMAIntegrationTest))


if __name__ == "__main__":
    unittest.main()
