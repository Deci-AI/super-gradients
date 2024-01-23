import math
import unittest
import torch
import random
from typing import List, Optional, Tuple, Dict

from super_gradients.training.metrics.detection_metrics import DetectionMetricsDistanceBased
from super_gradients.training.utils.detection_utils import EuclideanDistance, ManhattanDistance


class TestDetectionMetricsDistanceBased(unittest.TestCase):
    def setUp(self):
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        random.seed(42)
        self.num_classes = 3
        self.predefined_correct_class = 1  # random.randint(0, self.num_classes - 1)
        self.using_predefined_class = True
        self.verbose = True

        self.distance_thresholds = [5.0]
        self.score_thres = 0.1
        self.img_width = 640
        self.img_height = 480
        # Mock input image tensor (1 batch, 1 image, 640x480 size)
        self.img_tensor = (torch.randint(0, 256, (1, 1, self.img_height, self.img_width), dtype=torch.uint8)).int()

        self.metric = DetectionMetricsDistanceBased(
            num_cls=self.num_classes,
            post_prediction_callback=self.mock_post_prediction_callback,
            distance_thresholds=self.distance_thresholds,
            score_thres=self.score_thres,
            distance_metric=EuclideanDistance(),
        )

    def mock_post_prediction_callback(self, batch_preds: List[torch.Tensor], device="cpu") -> List[torch.Tensor]:
        batch_transformed_preds = []

        for preds in batch_preds:  # Iterate over each image's raw predictions in the batch
            transformed_preds = []
            for i in range(preds.size(0)):  # Iterate over each prediction for the current image
                # Get the bounding box coordinates (cx, cy, w, h) from raw preds
                cx, cy, w, h = preds[i]

                # Generate a random confidence score (for testing)
                confidence = random.uniform(0.1, 0.9)

                # Generate a random class label (for testing)
                if self.using_predefined_class:
                    class_label = self.predefined_correct_class
                else:
                    class_label = random.randint(0, self.num_classes - 1)

                # Calculate absolute coordinates (x1, y1, x2, y2)
                x1 = cx - w / 2.0
                y1 = cy - h / 2.0
                x2 = cx + w / 2.0
                y2 = cy + h / 2.0

                # Store the transformed prediction in a tensor
                transformed_pred = torch.tensor([x1, y1, x2, y2, confidence, class_label], device=device)

                # Append the tensor to the list
                transformed_preds.append(transformed_pred)

            # Convert list of tensors to a single tensor for this image
            transformed_preds = torch.stack(transformed_preds)

            # Add this image's transformed predictions to the batch list
            batch_transformed_preds.append(transformed_preds)

        return batch_transformed_preds

    def validate_results(self, results: Dict, precision, recall, mAP, F1, places=4, verbose=False, description=None):
        if verbose and description:
            test_name = self.id().split(".")[-1]
            print(f"\n{test_name}():")
            print(f"Description: {description}")

        results = dict((k.split("@")[0].lower().replace("distance_based_", ""), v) for k, v in results.items())

        self.assertAlmostEqual(results["precision"], precision, places=places)
        self.assertAlmostEqual(results["recall"], recall, places=places)
        if mAP is not None:
            self.assertAlmostEqual(results["map"], mAP, places=places)
        self.assertAlmostEqual(results["f1"], F1, places=places)

    def generate_targets(self, img_width, img_height, num_classes, num_targets):
        targets = []
        for index in range(num_targets):
            # Generate random coordinates and dimensions for the target
            cx, cy = random.randint(0, img_width - 1), random.randint(0, img_height - 1)
            max_w = min(cx, img_width - cx) * 2
            max_h = min(cy, img_height - cy) * 2
            w = random.randint(1, max_w)
            h = random.randint(1, max_h)

            # Pick label
            if self.using_predefined_class:
                label = self.predefined_correct_class
            else:
                label = random.randint(0, num_classes - 1)

            # Normalize target coordinates and dimensions to [0, 1]
            target_x1, target_y1, target_x2, target_y2 = self.normalize_coordinates(cx, cy, w, h, img_width, img_height)

            target_w = target_x2 - target_x1
            target_h = target_y2 - target_y1

            target_cx = target_x1 + target_w / 2
            target_cy = target_y1 + target_h / 2

            # Append target data in LABEL_CXCYWH format
            targets.append([index, label, target_cx, target_cy, target_w, target_h])

        targets = torch.tensor(targets, dtype=torch.float32).reshape(num_targets, 6)
        return targets

    @staticmethod
    def generate_predictions(distance_thresholds, img_height, img_width, num_correct_preds, num_targets, num_total_predictions, targets):
        predictions = []
        for _ in range(num_correct_preds):
            # Generate predictions close to some targets
            target_idx = random.randint(0, num_targets - 1)
            _, _, x_center, y_center, _, _ = targets[target_idx]
            dist_idx = random.randint(0, len(distance_thresholds) - 1)
            distance = random.randint(0, distance_thresholds[dist_idx])
            angle = random.uniform(0, 2 * math.pi)
            x_center_scalar = int(x_center.item() * img_width)  # As denormalized
            y_center_scalar = int(y_center.item() * img_height)  # As denormalized
            pred_x = int(round(x_center_scalar + distance * math.cos(angle)))
            pred_y = int(round(y_center_scalar + distance * math.sin(angle)))
            max_w = min(pred_x, img_width - pred_x) * 2
            max_h = min(pred_y, img_height - pred_y) * 2
            w = random.randint(1, max_w)
            h = random.randint(1, max_h)

            # Append prediction data in CXCYWH format
            prediction = torch.tensor([pred_x, pred_y, w, h])
            predictions.append(prediction)

        for _ in range(num_total_predictions - num_correct_preds):
            # Generate predictions far from any target
            pred_x, pred_y = random.randint(0, img_width - 1), random.randint(0, img_height - 1)
            max_w = min(pred_x, img_width - pred_x) * 2
            max_h = min(pred_y, img_height - pred_y) * 2
            w = random.randint(1, max_w)
            h = random.randint(1, max_h)

            # Append prediction data in CXCYWH format
            prediction = torch.tensor([pred_x, pred_y, w, h])
            predictions.append(prediction)

        # Convert the list of tensors into a single tensor and reshape it
        predictions = torch.stack(predictions)
        return [predictions]

    @staticmethod
    def normalize_coordinates(cx: int, cy: int, w: int, h: int, img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """
        Normalize coordinates and dimensions to [0, 1] range.

        Args:
            cx (int): Center x-coordinate.
            cy (int): Center y-coordinate.
            w (int): Width.
            h (int): Height.
            img_width (int): Width of the image.
            img_height (int): Height of the image.

        Returns:
            Tuple[float, float, float, float]: Normalized coordinates and dimensions (x1, y1, x2, y2).
        """
        x1 = max(0, (cx - w / 2) / img_width)
        y1 = max(0, (cy - h / 2) / img_height)
        x2 = min(1, (cx + w / 2) / img_width)
        y2 = min(1, (cy + h / 2) / img_height)
        return x1, y1, x2, y2

    @staticmethod
    def generate_mock_data(
        self,
        img_width: int,
        img_height: int,
        num_classes: int,
        num_targets: int,
        distance_thresholds: List[float],
        target_precision: float,
        target_recall: float,
        crowd_targets: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Generate mock data for testing object detection metrics.

        Args:
            img_width (int): Width of the image.
            img_height (int): Height of the image.
            num_classes (int): Number of classes.
            num_targets (int): Number of mock target objects.
            distance_thresholds: (List[float]): List of distance thresholds.
            target_precision (float): Desired precision value (between 0 and 1).
            target_recall (float): Desired recall value (between 0 and 1).
            crowd_targets (bool, optional): Whether to create crowded targets. Default is False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
                - Mock targets with shape (num_targets, 6) in LABEL_CXCYWH format.
                - Mock predictions with shape (num_total_predictions, 4) in CXCYWH format.
                - Crowd targets with shape (num_crowd_targets, 6) in LABEL_CXCYWH format, or None if crowd_targets is False.
        """
        # Generate targets
        targets = self.generate_targets(img_width, img_height, num_classes, num_targets)

        # Calculate TP, FP, and FN
        TP = num_targets
        FP = int((TP / target_precision) - TP)

        # Calculate the total number of predictions you'll need to generate
        num_total_predictions = TP + FP  # Because TP + FP = total predictions
        num_correct_preds = math.ceil(num_total_predictions * target_precision)

        # Generate predictions accordingly to scenario
        predictions = self.generate_predictions(distance_thresholds, img_height, img_width, num_correct_preds, num_targets, num_total_predictions, targets)

        crowd_targets_data = None
        if crowd_targets:
            # Create crowded targets (similar to targets)
            crowd_targets_data = self.generate_targets(img_width, img_height, num_classes, num_targets)

        return targets, predictions, crowd_targets_data

    @staticmethod
    def calculate_expected_metrics(self, precision, recall):
        # Calculate expected mAP (simplified in this context)
        expected_mAP = precision

        # Calculate expected F1-score
        if precision + recall == 0:
            expected_f1 = 0.0
        else:
            expected_f1 = (2 * precision * recall) / (precision + recall)

        return expected_mAP, expected_f1

    # Test Scenario: Single target in the image, single match out of three total predictions.
    # Desired precision: 0.33 (1 out of 3 predictions should match).
    # Desired recall: 0.33 (1 out of 3 targets should be detected).
    # Total predictions: 3 (to meet the precision and recall requirements).
    # Crowd targets: None
    def test_random_case_generation_and_verification(self):
        # Set random seeds for reproducibility (to ensure the seeds even at this level)
        torch.manual_seed(42)
        random.seed(42)

        # Test configuration
        num_targets = 1  # Number of mock target objects
        target_precision = 0.3333
        target_recall = 1
        crowd_targets = False  # Set to True to generate crowd targets

        # Generate mock data
        targets, predictions, crowd_targets_data = self.generate_mock_data(
            self, self.img_width, self.img_height, self.num_classes, num_targets, self.distance_thresholds, target_precision, target_recall, crowd_targets
        )

        # Call the update and compute methods with generated data
        self.metric.update(preds=predictions, target=targets, device="cpu", inputs=self.img_tensor, crowd_targets=None)
        results = self.metric.compute()

        # Calculate expected mAP and F1-score
        expected_mAP, expected_f1 = self.calculate_expected_metrics(self, target_precision, target_recall)

        # Validate the results
        self.validate_results(results, precision=target_precision, recall=target_recall, mAP=None, F1=expected_f1)

    # checks whether a single prediction that matches a single target will yield a perfect score
    # (Precision, Recall, F1 score, and mAP all set to 1.0). Using Manhattan distance.
    def test_distance_based_score_l1_norm_distance_single_target_single_prediction_match(self):
        scenario = "a single prediction that matches a single target using L1 Norm as a metric"
        # Set random seeds for reproducibility (to ensure the seeds even at this level)
        torch.manual_seed(42)
        random.seed(42)

        # Mock raw_preds (model's output) in format cx, cy, w, h; coordinates within image dimensions
        # One prediction is 5px away from the center of the target
        raw_preds = [torch.tensor([[15, 15, 10, 10]])]  # Close to target

        # Define target (unnormalized) coordinates within image dimensions
        target_x1, target_y1, target_x2, target_y2 = 10, 10, 20, 20

        # Normalize target coordinates
        target_x1, target_x2 = target_x1 / self.img_width, target_x2 / self.img_width
        target_y1, target_y2 = target_y1 / self.img_height, target_y2 / self.img_height

        # Calculate normalized width and height
        target_w = target_x2 - target_x1
        target_h = target_y2 - target_y1

        # Calculate normalized center coordinates
        target_cx = target_x1 + target_w / 2
        target_cy = target_y1 + target_h / 2

        # Mock targets
        # Create a single target for one image with shape (1, 6).
        # Format: (index, label, cx, cy, w, h)
        targets = torch.tensor([[0, self.predefined_correct_class, target_cx, target_cy, target_w, target_h]], dtype=torch.float32).reshape(1, 6)

        self.metric = DetectionMetricsDistanceBased(
            num_cls=self.num_classes,
            post_prediction_callback=self.mock_post_prediction_callback,
            distance_thresholds=[5.0],
            score_thres=self.score_thres,
            distance_metric=ManhattanDistance(),
        )

        # Call the update method
        self.metric.update(preds=raw_preds, target=targets, device="cpu", inputs=self.img_tensor, crowd_targets=None)

        # Call the compute method to get the results
        results = self.metric.compute()

        # Validate the results
        self.validate_results(results, precision=1.0, recall=1.0, mAP=1.0, F1=1.0, verbose=self.verbose, description=scenario)

    # checks whether a single prediction that matches a single crowd target will yield a perfect score
    # (Precision, Recall, F1 score, and mAP all set to 1.0). Using Manhattan distance.
    def test_distance_based_score_euclidean_distance_single_crowd_target_single_prediction_target_miss(self):
        scenario = "A single prediction, single target and a single crowd target. Prediction close to crowd target and far from target."

        # Set random seeds for reproducibility (to ensure the seeds even at this level)
        torch.manual_seed(42)
        random.seed(42)

        # Mock raw_preds (model's output) in format cx, cy, w, h; coordinates within image dimensions
        # One prediction is 5px away from the center of the crowd target
        raw_preds = [torch.tensor([[15, 15, 10, 10]])]  # Close to crowd target

        # Define target (unnormalized) coordinates within image dimensions
        target_x1, target_y1, target_x2, target_y2 = 100, 100, 200, 200

        # Normalize target coordinates
        target_x1, target_x2 = target_x1 / self.img_width, target_x2 / self.img_width
        target_y1, target_y2 = target_y1 / self.img_height, target_y2 / self.img_height

        # Calculate normalized width and height
        target_w = target_x2 - target_x1
        target_h = target_y2 - target_y1

        # Calculate normalized center coordinates
        target_cx = target_x1 + target_w / 2
        target_cy = target_y1 + target_h / 2

        # Mock targets
        # Create a single target for one image with shape (1, 6).
        # Format: (index, label, cx, cy, w, h)
        targets = torch.tensor([[0, self.predefined_correct_class, target_cx, target_cy, target_w, target_h]], dtype=torch.float32).reshape(1, 6)

        # Define crowd target (unnormalized) coordinates within image dimensions
        crowd_target_x1, crowd_target_y1, crowd_target_x2, crowd_target_y2 = 10, 10, 20, 20

        # Normalize crowd target coordinates
        crowd_target_x1, crowd_target_x2 = crowd_target_x1 / self.img_width, crowd_target_x2 / self.img_width
        crowd_target_y1, crowd_target_y2 = crowd_target_y1 / self.img_height, crowd_target_y2 / self.img_height

        # Calculate normalized width and height for crowd target
        crowd_target_w = crowd_target_x2 - crowd_target_x1
        crowd_target_h = crowd_target_y2 - crowd_target_y1

        # Calculate normalized center coordinates for crowd target
        crowd_target_cx = crowd_target_x1 + crowd_target_w / 2
        crowd_target_cy = crowd_target_y1 + crowd_target_h / 2

        # Mock crowd targets
        # Create a single crowd target for one image with shape (1, 6).
        # Format: (index, label, cx, cy, w, h)
        crowd_targets = torch.tensor(
            [[0, self.predefined_correct_class, crowd_target_cx, crowd_target_cy, crowd_target_w, crowd_target_h]], dtype=torch.float32
        ).reshape(1, 6)

        self.metric = DetectionMetricsDistanceBased(
            num_cls=self.num_classes,
            post_prediction_callback=self.mock_post_prediction_callback,
            distance_thresholds=[5.0],
            score_thres=self.score_thres,
            distance_metric=ManhattanDistance(),
        )

        # Call the update method
        self.metric.update(preds=raw_preds, target=targets, device="cpu", inputs=self.img_tensor, crowd_targets=crowd_targets)

        # Call the compute method to get the results
        results = self.metric.compute()

        # Validate the results
        self.validate_results(results, precision=0, recall=0, mAP=0, F1=0, verbose=self.verbose, description=scenario)

    def test_distance_based_score_euclidean_distance_single_crowd_target_single_prediction_target_match(self):
        scenario = "A single prediction, single target and a single crowd target. Prediction match target and far from the crowd target."

        # Set random seeds for reproducibility (to ensure the seeds even at this level)
        torch.manual_seed(42)
        random.seed(42)

        # Mock raw_preds (model's output) in format cx, cy, w, h; coordinates within image dimensions
        # One prediction is 5px away from the center of the crowd target
        raw_preds = [torch.tensor([[15, 15, 10, 10]])]  # Close to crowd target

        # Define target (unnormalized) coordinates within image dimensions
        target_x1, target_y1, target_x2, target_y2 = 10, 10, 20, 20

        # Normalize target coordinates
        target_x1, target_x2 = target_x1 / self.img_width, target_x2 / self.img_width
        target_y1, target_y2 = target_y1 / self.img_height, target_y2 / self.img_height

        # Calculate normalized width and height
        target_w = target_x2 - target_x1
        target_h = target_y2 - target_y1

        # Calculate normalized center coordinates
        target_cx = target_x1 + target_w / 2
        target_cy = target_y1 + target_h / 2

        # Mock targets
        # Create a single target for one image with shape (1, 6).
        # Format: (index, label, cx, cy, w, h)
        targets = torch.tensor([[0, self.predefined_correct_class, target_cx, target_cy, target_w, target_h]], dtype=torch.float32).reshape(1, 6)

        # Define crowd target (unnormalized) coordinates within image dimensions
        crowd_target_x1, crowd_target_y1, crowd_target_x2, crowd_target_y2 = 100, 100, 200, 200

        # Normalize crowd target coordinates
        crowd_target_x1, crowd_target_x2 = crowd_target_x1 / self.img_width, crowd_target_x2 / self.img_width
        crowd_target_y1, crowd_target_y2 = crowd_target_y1 / self.img_height, crowd_target_y2 / self.img_height

        # Calculate normalized width and height for crowd target
        crowd_target_w = crowd_target_x2 - crowd_target_x1
        crowd_target_h = crowd_target_y2 - crowd_target_y1

        # Calculate normalized center coordinates for crowd target
        crowd_target_cx = crowd_target_x1 + crowd_target_w / 2
        crowd_target_cy = crowd_target_y1 + crowd_target_h / 2

        # Mock crowd targets
        # Create a single crowd target for one image with shape (1, 6).
        # Format: (index, label, cx, cy, w, h)
        crowd_targets = torch.tensor(
            [[0, self.predefined_correct_class, crowd_target_cx, crowd_target_cy, crowd_target_w, crowd_target_h]], dtype=torch.float32
        ).reshape(1, 6)

        self.metric = DetectionMetricsDistanceBased(
            num_cls=self.num_classes,
            post_prediction_callback=self.mock_post_prediction_callback,
            distance_thresholds=[5.0],
            score_thres=self.score_thres,
            distance_metric=ManhattanDistance(),
        )

        # Call the update method
        self.metric.update(preds=raw_preds, target=targets, device="cpu", inputs=self.img_tensor, crowd_targets=crowd_targets)

        # Call the compute method to get the results
        results = self.metric.compute()

        # Validate the results
        self.validate_results(results, precision=1, recall=1, mAP=1, F1=1, verbose=self.verbose, description=scenario)

    # checks whether a single prediction that matches a single target will yield a perfect score
    # (Precision, Recall, F1 score, and mAP all set to 1.0). Using Euclidean distance.
    def test_distance_based_score_euclidean_distance_single_target_single_prediction_match(self):
        scenario = "a single prediction that matches a single target using Euclidean distance as a metric"

        # Set random seeds for reproducibility (to ensure the seeds even at this level)
        torch.manual_seed(42)
        random.seed(42)

        # Mock raw_preds (model's output) in format cx, cy, w, h; coordinates within image dimensions
        # One prediction is 5px away from the center of the target
        raw_preds = [torch.tensor([[15, 15, 10, 10]])]  # Close to target

        # Define target (unnormalized) coordinates within image dimensions
        target_x1, target_y1, target_x2, target_y2 = 10, 10, 20, 20

        # Normalize target coordinates
        target_x1, target_x2 = target_x1 / self.img_width, target_x2 / self.img_width
        target_y1, target_y2 = target_y1 / self.img_height, target_y2 / self.img_height

        # Calculate normalized width and height
        target_w = target_x2 - target_x1
        target_h = target_y2 - target_y1

        # Calculate normalized center coordinates
        target_cx = target_x1 + target_w / 2
        target_cy = target_y1 + target_h / 2

        # Mock targets
        # Create a single target for one image with shape (1, 6).
        # Format: (index, label, cx, cy, w, h)
        targets = torch.tensor([[0, self.predefined_correct_class, target_cx, target_cy, target_w, target_h]], dtype=torch.float32).reshape(1, 6)

        self.metric = DetectionMetricsDistanceBased(
            num_cls=self.num_classes,
            post_prediction_callback=self.mock_post_prediction_callback,
            distance_thresholds=[5.0],
            score_thres=self.score_thres,
            distance_metric=EuclideanDistance(),
        )

        # Call the update method
        self.metric.update(preds=raw_preds, target=targets, device="cpu", inputs=self.img_tensor, crowd_targets=None)

        # Call the compute method to get the results
        results = self.metric.compute()

        # Validate the results
        self.validate_results(results, precision=1.0, recall=1.0, mAP=1.0, F1=1.0, verbose=self.verbose, description=scenario)

    # checks whether a single prediction that doesn't match the target will yield zero for all metrics.
    def test_distance_based_score_euclidean_distance_single_target_single_prediction_miss(self):
        scenario = "a single prediction that doesn't match the target using Euclidean distance as a metric"

        # Set random seeds for reproducibility (to ensure the seeds even at this level)
        torch.manual_seed(42)
        random.seed(42)

        # Mock raw_preds (model's output) in format cx, cy, w, h; coordinates within image dimensions
        # One prediction is more than 5px away from the center of the target
        raw_preds = [torch.tensor([[15, 15, 10, 10]])]  # Far from target

        # Define target (unnormalized) coordinates within image dimensions
        target_x1, target_y1, target_x2, target_y2 = 40, 40, 80, 80

        # Normalize target coordinates
        target_x1, target_x2 = target_x1 / self.img_width, target_x2 / self.img_width
        target_y1, target_y2 = target_y1 / self.img_height, target_y2 / self.img_height

        # Calculate normalized width and height
        target_w = target_x2 - target_x1
        target_h = target_y2 - target_y1

        # Calculate normalized center coordinates
        target_cx = target_x1 + target_w / 2
        target_cy = target_y1 + target_h / 2

        # Mock targets
        # Create a single target for one image with shape (1, 6).
        # Format: (index, label, cx, cy, w, h)
        targets = torch.tensor([[0, self.predefined_correct_class, target_cx, target_cy, target_w, target_h]], dtype=torch.float32).reshape(1, 6)

        self.metric = DetectionMetricsDistanceBased(
            num_cls=self.num_classes,
            post_prediction_callback=self.mock_post_prediction_callback,
            distance_thresholds=[5.0],
            score_thres=self.score_thres,
            distance_metric=EuclideanDistance(),
        )

        # Call the update method
        self.metric.update(preds=raw_preds, target=targets, device="cpu", inputs=self.img_tensor, crowd_targets=None)

        # Call the compute method to get the results
        results = self.metric.compute()

        # Validate the results
        self.validate_results(results, precision=0, recall=0, mAP=0, F1=0, verbose=self.verbose, description=scenario)

    # checks whether the metrics are calculated correctly when there are multiple
    # predictions but only one matches with a single target.
    def test_distance_based_score_euclidean_distance_single_target_few_predictions(self):
        scenario = "a few predictions 1 - match, 2 - don't, single target using Euclidean distance as a metric"

        # Set random seeds for reproducibility (to ensure the seeds even at this level)
        torch.manual_seed(42)
        random.seed(42)

        # Mock raw_preds (model's output) in format cx, cy, w, h; coordinates within image dimensions
        # One prediction is 5px away from the center of the target
        # The other two predictions are placed randomly
        raw_preds = [torch.tensor([[15, 15, 10, 10], [100, 50, 10, 10], [200, 300, 20, 15]])]  # Close to target  # Randomly placed  # Randomly placed

        # Define target (unnormalized) coordinates within image dimensions
        target_x1, target_y1, target_x2, target_y2 = 10, 10, 20, 20

        # Normalize target coordinates
        target_x1, target_x2 = target_x1 / self.img_width, target_x2 / self.img_width
        target_y1, target_y2 = target_y1 / self.img_height, target_y2 / self.img_height

        # Calculate normalized width and height
        target_w = target_x2 - target_x1
        target_h = target_y2 - target_y1

        # Calculate normalized center coordinates
        target_cx = target_x1 + target_w / 2
        target_cy = target_y1 + target_h / 2

        # Mock targets
        # Create a single target for one image with shape (1, 6).
        # Format: (index, label, cx, cy, w, h)
        targets = torch.tensor([[0, self.predefined_correct_class, target_cx, target_cy, target_w, target_h]], dtype=torch.float32).reshape(1, 6)

        # Call the update method
        self.metric.update(preds=raw_preds, target=targets, device="cpu", inputs=self.img_tensor, crowd_targets=None)

        # Call the compute method to get the results
        results = self.metric.compute()

        # Validate the results
        self.validate_results(results, precision=0.3333, recall=1.0, mAP=1, F1=0.5, verbose=self.verbose, description=scenario)


if __name__ == "__main__":
    unittest.main()
