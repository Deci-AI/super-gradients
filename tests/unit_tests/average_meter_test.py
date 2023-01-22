import torch
import unittest
from super_gradients.training.utils.utils import AverageMeter


class TestAverageMeter(unittest.TestCase):
    """Test the behavior of the class is not changed since several parts of the code rely on it"""

    @classmethod
    def setUp(cls):
        cls.avg_float = AverageMeter()
        cls.avg_tuple = AverageMeter()
        cls.avg_list = AverageMeter()
        cls.avg_tensor = AverageMeter()
        cls.left_empty = AverageMeter()
        cls.list_of_avg_meter = [cls.avg_float, cls.avg_tuple, cls.avg_list, cls.avg_tensor]
        cls.score_types = [1.2, (3.0, 4.0), [5.0, 6.0, 7.0], torch.FloatTensor([8.0, 9.0, 10.0])]
        cls.batch_size = 3

    def test_empty_return_0(self):
        self.assertEqual(self.left_empty.average, 0)

    def test_correctness_and_typing(self):
        # VERIFY THE VALUES ARE INITIALIZED TO None & 0 FOR THE ABILITY TO USE ANY TYPE OF value
        self.assertIsNone(self.avg_float._sum)
        self.assertEqual(self.avg_float._count, 0)

        # RUN OVER THE DIFFERENT TYPES OF avg_meters AND VERIFY FOR EACH
        for list_idx, (avg_meter, score) in enumerate(zip(self.list_of_avg_meter, self.score_types)):
            # ADD THE VALUES 3 TIMES
            for repetition in [1, 2, 3]:
                avg_meter.update(score, self.batch_size)
                # VERIFY VALUES ARE CORRECT AND OF THE EXPECTED TYPES
                self.assertEqual(avg_meter._count, self.batch_size * repetition)
                if list_idx == 0:  # FOR avg_float
                    self.assertEqual(avg_meter._count, self.batch_size * repetition)
                    self.assertIsInstance(avg_meter.average, float)
                    self.assertAlmostEqual(avg_meter.average, score)
                else:  # FOR ALL THE OTHERS
                    self.assertListEqual(list(avg_meter._sum), [val * self.batch_size * repetition for val in score])
                    self.assertIsInstance(avg_meter.average, tuple)
                    self.assertListEqual(list(avg_meter.average), list(score))


if __name__ == "__main__":
    unittest.main()
