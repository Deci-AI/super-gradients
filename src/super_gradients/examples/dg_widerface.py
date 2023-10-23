from data_gradients.managers.detection_manager import DetectionAnalysisManager
from super_gradients.training.dataloaders.dataloaders import widerface_train, widerface_val

if __name__ == "__main__":

    train_loader = widerface_train()
    val_loader = widerface_val()

    analyzer = DetectionAnalysisManager(
        report_title="WiderfaceDL",
        train_data=train_loader,
        val_data=val_loader,
        class_names=train_loader.dataset.classes,
        log_dir="/home/shay.aharon/",
        # batches_early_stop=20,
        use_cache=True,  # With this we will be asked about the dataset information only once
    )

    analyzer.run()
