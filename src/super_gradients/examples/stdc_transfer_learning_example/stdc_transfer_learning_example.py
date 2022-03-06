from super_gradients.training.datasets.dataset_interfaces.dataset_interface import SuperviselyPersonsDatasetInterface
import super_gradients
from super_gradients.training.losses.bce_dice_loss import BCEDiceLoss
from super_gradients.training.sg_model import SgModel
from super_gradients.training.metrics import binary_IOU
from super_gradients.training.utils.segmentation_utils import ColorJitterSeg, RandomFlip, RandomRescale, \
    PadShortToCropSize, CropImageAndMask, ResizeSeg
from torchvision import transforms

# DATA_DIR = './data/CamVid/'
#
# # load repo with data if it is not exists
# if not os.path.exists(DATA_DIR):
#     print('Loading data...')
#     os.system('git clone https://github.com/alexgkendall/SegNet-Tutorial ./data')
#     print('Done!')
#
# x_train_dir = os.path.join(DATA_DIR, 'train')
# y_train_dir = os.path.join(DATA_DIR, 'trainannot')
#
# x_valid_dir = os.path.join(DATA_DIR, 'val')
# y_valid_dir = os.path.join(DATA_DIR, 'valannot')
#
# x_test_dir = os.path.join(DATA_DIR, 'test')
# y_test_dir = os.path.join(DATA_DIR, 'testannot')
#
#
# # helper function for data visualization
# def visualize(**images):
#     """PLot images in one row."""
#     n = len(images)
#     plt.figure(figsize=(16, 5))
#     for i, (name, image) in enumerate(images.items()):
#         plt.subplot(1, n, i + 1)
#         plt.xticks([])
#         plt.yticks([])
#         plt.title(' '.join(name.split('_')).title())
#         plt.imshow(image)
#     plt.show()
#
#
# class Dataset(BaseDataset):
#     """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
#
#     Args:
#         images_dir (str): path to images folder
#         masks_dir (str): path to segmentation masks folder
#         class_values (list): values of classes to extract from segmentation mask
#         augmentation (albumentations.Compose): data transfromation pipeline
#             (e.g. flip, scale, etc.)
#         preprocessing (albumentations.Compose): data preprocessing
#             (e.g. noralization, shape manipulation, etc.)
#
#     """
#
#     CLASSES = ['sky', 'building', 'pole', 'road', 'pavement',
#                'tree', 'signsymbol', 'fence', 'car',
#                'pedestrian', 'bicyclist', 'unlabelled']
#
#     def __init__(
#             self,
#             images_dir,
#             masks_dir,
#             classes=None,
#             augmentation=None,
#             preprocessing=None,
#     ):
#         classes = classes or self.CLASSES
#         self.ids = os.listdir(images_dir)
#         self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
#         self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
#
#         # convert str names to class values on masks
#         self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
#         self.classes = classes
#         self.augmentation = augmentation
#         self.preprocessing = preprocessing
#
#     def __getitem__(self, i):
#
#         # read data
#         image = cv2.imread(self.images_fps[i])
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image = cv2.resize(image, (0, 0), fx=1 + 1 / 3, fy=1 + 1 / 3)
#         image = np.moveaxis(image, -1, 0)
#         image = image.astype(np.float32)
#         mask = cv2.imread(self.masks_fps[i], 0)
#         mask = cv2.resize(mask, (0, 0), fx=1 + 1 / 3, fy=1 + 1 / 3)
#
#         # extract certain classes from mask (e.g. cars)
#         masks = [(mask == v) for v in self.class_values]
#         mask = np.stack(masks, axis=-1).astype(np.float32)
#         if mask.shape[-1] == 1:
#             mask = mask[:, :, 0]
#             mask = mask[np.newaxis, :, :]
#
#         # apply augmentations
#         if self.augmentation:
#             sample = self.augmentation(image=image, mask=mask)
#             image, mask = sample['image'], sample['mask']
#
#         # apply preprocessing
#         if self.preprocessing:
#             sample = self.preprocessing(image=image, mask=mask)
#             image, mask = sample['image'], sample['mask']
#
#         return image, mask
#
#     def __len__(self):
#         return len(self.ids)


# Lets look at data we have

# trainset = Dataset(x_train_dir, y_train_dir, classes=['car'])
# train_loader = torch.utils.data.DataLoader(trainset,
#                                            batch_size=32,
#                                            shuffle=True,
#                                            num_workers=8,
#                                            pin_memory=True,
#                                            drop_last=True)
#
# validset = Dataset(x_valid_dir, y_valid_dir, classes=['car'])
# valid_loader = torch.utils.data.DataLoader(validset,
#                                            batch_size=32,
#                                            shuffle=True,
#                                            num_workers=8,
#                                            pin_memory=True,
#                                            drop_last=True)


# class BinaryDiceLoss(torch.nn.Module):
#     """
#     Binary Dice Loss
#
#     Attributes:
#         logits: Whether to apply sigmoid on the network's output.
#     """
#
#     def __init__(self, logits: bool = True):
#         super(BinaryDiceLoss, self).__init__()
#         self.logits = logits
#
#     def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         """
#
#         @param input: Network's output shaped (N,1,H,W) or (N,H,W)
#         @param target: Ground truth shaped (N,H,W)
#         """
#         if self.logits:
#             input = torch.sigmoid(input)
#
#         smooth = 1.
#         iflat = input.view(-1)
#         tflat = target.view(-1)
#         intersection = (iflat * tflat).sum()
#
#         return 1 - ((2. * intersection + smooth) /
#                     (iflat.sum() + tflat.sum() + smooth))


dataset_params = {
    "image_mask_transforms_aug": transforms.Compose([ColorJitterSeg(brightness=0.5, contrast=0.5, saturation=0.5),
                                                     RandomFlip(),
                                                     RandomRescale(scales=[0.25, 1.]),
                                                     PadShortToCropSize([320, 480]),
                                                     CropImageAndMask(crop_size=[320, 480],
                                                                      mode="random")]),
    "image_mask_transforms": transforms.Compose([ResizeSeg(h=480, w=320)])
}

# dataset_interface = DatasetInterface(train_loader=train_loader, val_loader=valid_loader, classes=['car'])

dataset_interface = SuperviselyPersonsDatasetInterface(dataset_params)

super_gradients.init_trainer()
# criterion_params = {"num_classes": 2,
#                     "mining_percent": 1.,  # full mining i.e regular cross entropy
#                     "weights": [1., 0.6, 0.4, 1.],
#                     "threshold": 0}

# train_params = {"max_epochs": 200,
#                 "lr_mode": "poly",
#                 "initial_lr": 0.001,  # for batch_size=16
#                 "lr_warmup_epochs": 0,
#                 "multiply_head_lr": 1.,
#                 "optimizer": "SGD",
#                 "optimizer_params": {"momentum": 0.9,
#                                      "weight_decay": 0},
#                 # "loss": "stdc_loss",
#                 # "criterion_params": {"num_classes": 1,
#                 #                      "ignore_index": 0,
#                 #                      "mining_percent": 1.,  # full mining i.e regular cross entropy
#                 #                      "weights": [1., 0.6, 0.4, 1.],
#                 #                      "threshold": 0},
#                 "loss": STDCLoss(**criterion_params),
#
#                 "ema": True,
#
#                 "zero_weight_decay_on_bias_and_bn": True,
#                 "average_best_models": True,
#                 "mixed_precision": False,
#                 "metric_to_watch": "mIOU",
#                 "greater_metric_to_watch_is_better": True,
#                 "train_metrics_list": [mIOU(num_classes=2, bg_index=0), PixelAccuracy()],
#                 "valid_metrics_list": [mIOU(num_classes=2, bg_index=0), PixelAccuracy()],
#                 "loss_logging_items_names": ["main_loss", "aux_loss1", "aux_loss2", "detail_loss", "loss"]
#                 }
#
# arch_params = {"num_classes": 2,
#                "use_aux_heads": True,
#                "sync_bn": True,
#                "external_checkpoint_path": "/home/shay.aharon/stdc_backbones/stdc1_imagenet_pretrained.pth",
#                "load_backbone": True,
#                "load_weights_only": True,
#                "strict_load": "no_key_matching"}

# dataset_params = {"batch_size": 32,  # batch size for trainset in DatasetInterface
#                   "val_batch_size": 32,  # batch size for valset in DatasetInterface
#                   "crop_size": [320, 480],  # crop size (size of net's input)
#                   "img_size": 600,
#                   "train_loader_drop_last": True,
#                   "color_jitter": 0.5,
#                   "random_scales": [0.25, 1.],
#                   "image_mask_transforms_aug": transforms.Compose([
#                       ColorJitterSeg(0.5, 0.5, 0.5),
#                       RandomFlip(),
#                       RandomRescale(scales=[0.25, 1.]),
#                       PadShortToCropSize([320, 480]),
#                       CropImageAndMask(crop_size=[320, 480], mode="random")
#                   ]),  # train
#                   "image_mask_transforms": transforms.Compose([
#                       Rescale(scale_factor=0.6),
#                       CropImageAndMask(crop_size=[320, 480], mode="center")])  # validation
#                   }
# inds = None
# for i, (img, mask) in enumerate(dataset):
#     if inds is None:
#         inds = np.unique(mask)
#     else:
#         inds = np.unique(np.concatenate([inds, np.unique(mask)]))
#
#
# image, mask = dataset[4] # get some sample
# visualize(
#     image=image,
#     cars_mask=mask.squeeze(),
# )
model = SgModel("regseg48_transfer_learning_old_dice")

# CONNECTING THE DATASET INTERFACE WILL SET SGMODEL'S CLASSES ATTRIBUTE ACCORDING TO PASCAL VOC
model.connect_dataset_interface(dataset_interface)

# THIS IS WHERE THE MAGIC HAPPENS- SINCE SGMODEL'S CLASSES ATTRIBUTE WAS SET TO BE DIFFERENT FROM COCO'S, AFTER
# LOADING THE PRETRAINED YOLO_V5M, IT WILL CALL IT'S REPLACE_HEAD METHOD AND CHANGE IT'S DETECT LAYER ACCORDING
# TO PASCAL VOC CLASSES
model.build_model("regseg48", arch_params={"pretrained_weights": "cityscapes"})

train_params = {"max_epochs": 20,
                "lr_mode": "poly",
                "initial_lr": 0.001,  # for batch_size=16
                "lr_warmup_epochs": 0,
                "optimizer": "SGD",
                "optimizer_params": {"momentum": 0.9,
                                     "weight_decay": 0},
                "loss": BCEDiceLoss(),
                "ema": True,
                "zero_weight_decay_on_bias_and_bn": True,
                "average_best_models": True,
                "mixed_precision": False,
                "metric_to_watch": "binary_IOU",
                "greater_metric_to_watch_is_better": True,
                "train_metrics_list": [binary_IOU()],
                "valid_metrics_list": [binary_IOU()],
                "loss_logging_items_names": ["loss"]
                }

model.train(train_params)
