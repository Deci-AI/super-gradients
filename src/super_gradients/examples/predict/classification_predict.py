from super_gradients.training import dataloaders


data_dir = "/path/to/coco_dataset_dir"
train_dataloader = dataloaders.get(name="coco2017_train", dataset_params={"data_dir": data_dir})
val_dataloader = dataloaders.get(name="coco2017_val", dataset_params={"data_dir": data_dir})
