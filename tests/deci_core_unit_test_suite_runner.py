import sys
import unittest

from tests.integration_tests.ema_train_integration_test import EMAIntegrationTest
from tests.unit_tests import ZeroWdForBnBiasTest, SaveCkptListUnitTest, TestAverageMeter, \
    TestRepVgg, TestWithoutTrainTest, OhemLossTest, EarlyStopTest, SegmentationTransformsTest, \
    TestConvBnRelu, FactoriesTest, InitializeWithDataloadersTest
from tests.end_to_end_tests import TestTrainer
from tests.unit_tests.load_checkpoint_from_direct_path_test import LoadCheckpointFromDirectPathTest
from tests.unit_tests.phase_delegates_test import ContextMethodsTest
from tests.unit_tests.random_erase_test import RandomEraseTest
from tests.unit_tests.strictload_enum_test import StrictLoadEnumTest
from tests.unit_tests.train_with_intialized_param_args_test import TrainWithInitializedObjectsTest
from tests.unit_tests.pretrained_models_unit_test import PretrainedModelsUnitTest
from tests.unit_tests.lr_warmup_test import LRWarmupTest
from tests.unit_tests.kd_ema_test import KDEMATest
from tests.unit_tests.kd_model_test import KDModelTest
from tests.unit_tests.dice_loss_test import DiceLossTest
from tests.unit_tests.update_param_groups_unit_test import UpdateParamGroupsTest
from tests.unit_tests.vit_unit_test import TestViT
from tests.unit_tests.yolox_unit_test import TestYOLOX
from tests.unit_tests.lr_cooldown_test import LRCooldownTest
from tests.unit_tests.detection_targets_format_transform_test import DetectionTargetsTransformTest
from tests.unit_tests.forward_pass_prep_fn_test import ForwardpassPrepFNTest
from tests.unit_tests.mask_loss_test import MaskAttentionLossTest


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
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(SaveCkptListUnitTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(ZeroWdForBnBiasTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(TestAverageMeter))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(TestRepVgg))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(TestWithoutTrainTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(StrictLoadEnumTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(LoadCheckpointFromDirectPathTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(TrainWithInitializedObjectsTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(RandomEraseTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(OhemLossTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(EarlyStopTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(SegmentationTransformsTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(PretrainedModelsUnitTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(LRWarmupTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(TestConvBnRelu))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(FactoriesTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(DiceLossTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(TestViT))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(KDEMATest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(KDModelTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(TestYOLOX))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(InitializeWithDataloadersTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(LRCooldownTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(DetectionTargetsTransformTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(ForwardpassPrepFNTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(ContextMethodsTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(UpdateParamGroupsTest))
        self.unit_tests_suite.addTest(self.test_loader.loadTestsFromModule(MaskAttentionLossTest))

    def _add_modules_to_end_to_end_tests_suite(self):
        """
        _add_modules_to_end_to_end_tests_suite - Adds end to end tests to the Unit Tests Test Suite
            :return:
        """
        self.end_to_end_tests_suite.addTest(self.test_loader.loadTestsFromModule(TestTrainer))
        self.end_to_end_tests_suite.addTest(self.test_loader.loadTestsFromModule(EMAIntegrationTest))


if __name__ == '__main__':
    unittest.main()
