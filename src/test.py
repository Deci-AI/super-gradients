from super_gradients.training.losses import SegKDLoss


crit = SegKDLoss(kd_loss="cross_entropy", ce_loss="cross_entropy", weights=[1.0], kd_loss_weights=[1, 1])

print(crit)
