from super_gradients.training.datasets.dali_datasets.dali_dataloaders import DaliImagenetDataset

try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
    from nvidia.dali.pipeline import pipeline_def
    import nvidia.dali.types as types
    import nvidia.dali.fn as fn
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")
from super_gradients.training import SgModel
from super_gradients.training.datasets.dataset_interfaces import DatasetInterface
from super_gradients.training.metrics.classification_metrics import Accuracy, Top5


@pipeline_def
def create_dali_pipeline(data_dir, crop, size, shard_id, num_shards, dali_cpu=False, is_training=True):
    images, labels = fn.readers.file(file_root=data_dir,
                                     shard_id=shard_id,
                                     num_shards=num_shards,
                                     random_shuffle=is_training,
                                     pad_last_batch=True,
                                     name="Reader")
    dali_device = 'cpu' if dali_cpu else 'gpu'
    decoder_device = 'cpu' if dali_cpu else 'mixed'

    # ask nvJPEG to preallocate memory for the biggest sample in
    # ImageNet for CPU and GPU to avoid reallocations in runtime
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
    # ask HW NVJPEG to allocate memory ahead for the biggest image in the data set to avoid reallocations in runtime
    preallocate_width_hint = 5980 if decoder_device == 'mixed' else 0
    preallocate_height_hint = 6430 if decoder_device == 'mixed' else 0
    if is_training:
        images = fn.decoders.image_random_crop(images,
                                               device=decoder_device, output_type=types.RGB,
                                               device_memory_padding=device_memory_padding,
                                               host_memory_padding=host_memory_padding,
                                               preallocate_width_hint=preallocate_width_hint,
                                               preallocate_height_hint=preallocate_height_hint,
                                               random_aspect_ratio=[0.8, 1.25],
                                               random_area=[0.1, 1.0],
                                               num_attempts=100)
        images = fn.resize(images,
                           device=dali_device,
                           resize_x=crop,
                           resize_y=crop,
                           interp_type=types.INTERP_TRIANGULAR)
        mirror = fn.random.coin_flip(probability=0.5)
    else:
        images = fn.decoders.image(images,
                                   device=decoder_device,
                                   output_type=types.RGB)
        images = fn.resize(images,
                           device=dali_device,
                           size=size,
                           mode="not_smaller",
                           interp_type=types.INTERP_TRIANGULAR)
        mirror = False

    images = fn.crop_mirror_normalize(images.gpu(),
                                      dtype=types.FLOAT,
                                      output_layout="CHW",
                                      crop=(crop, crop),
                                      mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                      std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                                      mirror=mirror)
    labels = labels.gpu()
    return images, labels


pipe = create_dali_pipeline(batch_size=128,
                            num_threads=8,
                            device_id=0,
                            seed=12,
                            data_dir="/data/Imagenet/train/",
                            crop=224,
                            size=256,
                            shard_id=0,
                            num_shards=1,
                            is_training=True,
                            dali_cpu=True)
pipe.build()
train_loader = DALIClassificationIterator(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL,
                                          auto_reset=True)
train_loader = DaliImagenetDataset(train_loader)

pipe = create_dali_pipeline(batch_size=200,
                            num_threads=8,
                            device_id=0,
                            seed=12,
                            data_dir="/data/Imagenet/val/",
                            crop=224,
                            size=256,
                            shard_id=0,
                            num_shards=1,
                            is_training=False,
                            dali_cpu=True)

pipe.build()
val_loader = DALIClassificationIterator(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL,
                                        auto_reset=True)
val_loader = DaliImagenetDataset(val_loader)

di = DatasetInterface(train_loader=train_loader, val_loader=val_loader, classes=list(range(1000)))
trainer = SgModel(experiment_name='dali_imagenet_resnet50_256')
trainer.connect_dataset_interface(di)
trainer.build_model("resnet50")

train_params = {"max_epochs": 110,
                "lr_mode": "step",
                "lr_updates": [30, 60, 90, 100],
                "lr_decay_factor": 0.1,
                "initial_lr": 0.1,
                "loss": "cross_entropy",
                "train_metrics_list": [Accuracy(), Top5()],
                "valid_metrics_list": [Accuracy(), Top5()],

                "loss_logging_items_names": ["Loss"],
                "metric_to_watch": "Accuracy",
                "greater_metric_to_watch_is_better": True}
trainer.train(train_params)
