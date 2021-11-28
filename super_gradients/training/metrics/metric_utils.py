import numpy as np
import torch
from torchmetrics import MetricCollection
from super_gradients.training.metrics.detection_metrics import ap_per_class
from super_gradients.training.utils.utils import AverageMeter


def calc_batch_prediction_detection_metrics_per_class(metrics, dataset_interface, iou_thres, silent_mode, images_counter,
                                                      per_class_verbosity, class_names, test_loss):

    metrics = [np.concatenate(x, 0) for x in list(zip(*metrics))]
    if len(metrics):
        precision, recall, average_precision, f1, ap_class = ap_per_class(*metrics)
        if iou_thres.is_range():
            precision, recall, average_precision, f1 = precision[:, 0], recall[:, 0], average_precision.mean(
                1), average_precision[:, 0]
        mean_precision, mean_recall, map, mf1 = precision.mean(), recall.mean(), average_precision.mean(), f1.mean()
        targets_per_class = np.bincount(metrics[3].astype(np.int64),
                                        minlength=len(dataset_interface.testset.classes))
    else:
        targets_per_class = torch.zeros(1)

    if not silent_mode:
        # PRINT RESULTS
        map_str = 'mAP@%.1f' % iou_thres[0] if not iou_thres.is_range() else 'mAP@%.2f:%.2f' % iou_thres
        print(('%15s' * 7) % ('Class', 'Images', 'Targets', 'Precision', 'Recall', map_str, 'F1'))
        pf = '%15s' + '%15.3g' * 6  # print format
        print(pf % ('all', images_counter, targets_per_class.sum(), mean_precision, mean_recall, map, mf1))

        # PRINT RESULTS PER CLASS
        if len(dataset_interface.testset.classes) > 1 and len(metrics) and per_class_verbosity:
            for i, c in enumerate(ap_class):
                print(pf % (
                    class_names[c], images_counter, targets_per_class[c], precision[i], recall[i],
                    average_precision[i],
                    f1[i]))

    results_tuple = (mean_precision, mean_recall, map, mf1, *test_loss.average)
    return results_tuple


def get_logging_values(loss_loggings: AverageMeter, metrics: MetricCollection, criterion=None):
    """
    @param loss_loggings: AverageMeter running average for the loss items
    @param metrics: MetricCollection object for running user specified metrics
    @param criterion the object loss_loggings average meter is monitoring, when set to None- only the metrics values are
    computed and returned.

    @return: tuple of the computed values
    """
    if criterion is not None:
        loss_loggingg_avg = loss_loggings.average
        if not isinstance(loss_loggingg_avg, tuple):
            loss_loggingg_avg = tuple([loss_loggingg_avg])
        logging_vals = loss_loggingg_avg + get_metrics_results_tuple(metrics)
    else:
        logging_vals = get_metrics_results_tuple(metrics)

    return logging_vals


def get_metrics_titles(metrics_collection: MetricCollection):
    """

    @param metrics_collection: MetricCollection object for running user specified metrics
    @return: list of all the names of the computed values list(str)
    """
    titles = []
    for metric_name, metric in metrics_collection.items():
        if metric_name == "additional_items":
            continue
        elif hasattr(metric, "component_names"):
            titles += metric.component_names
        else:
            titles.append(metric_name)

    return titles


def get_metrics_results_tuple(metrics_collection: MetricCollection):
    """

    @param metrics_collection: metrics collection of the user specified metrics
    @type metrics_collection
    @return: tuple of metrics values
    """
    if metrics_collection is None:
        results_tuple = ()
    else:
        results_tuple = tuple(flatten_metrics_dict(metrics_collection.compute()).values())
    return results_tuple


def flatten_metrics_dict(metrics_dict: dict):
    """
    :param metrics_dict - dictionary of metric values where values can also be dictionaries containing subvalues
    (in the case of compound metrics)

    @return: flattened dict of metric values i.e {metric1_name: metric1_value...}
    """
    flattened = {}
    for metric_name, metric_val in metrics_dict.items():
        if metric_name == "additional_items":
            continue
        # COLLECT ALL OF THE COMPONENTS IN THE CASE OF COMPOUND METRICS
        elif isinstance(metric_val, dict):
            for sub_metric_name, sub_metric_val in metric_val.items():
                flattened[sub_metric_name] = sub_metric_val
        else:
            flattened[metric_name] = metric_val

    return flattened


def get_metrics_dict(metrics_tuple, metrics_collection, loss_logging_item_names):
    """
    Returns a dictionary with the epoch results as values and their names as keys.
    @param metrics_tuple: the result tuple
    @param metrics_collection: MetricsCollection
    @param loss_logging_item_names: loss component's names.
    @return: dict
    """
    keys = loss_logging_item_names + get_metrics_titles(metrics_collection)
    metrics_dict = dict(zip(keys, list(metrics_tuple)))
    return metrics_dict


def get_train_loop_description_dict(metrics_tuple, metrics_collection, loss_logging_item_names, **log_items):
    """
    Returns a dictionary with the epoch's logging items as values and their names as keys, with the purpose of
     passing it as a description to tqdm's progress bar.

    @param metrics_tuple: the result tuple
    @param metrics_collection: MetricsCollection
    @param loss_logging_item_names: loss component's names.
    @param log_items additional logging items to be rendered.
    @return: dict
    """
    log_items.update(get_metrics_dict(metrics_tuple, metrics_collection, loss_logging_item_names))
    for key, value in log_items.items():
        if isinstance(value, torch.Tensor):
            log_items[key] = value.detach().item()

    return log_items
