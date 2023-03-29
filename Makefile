cells-uyemf:
#	CUDA_VISIBLE_DEVICES=0 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=cells-uyemf dataset_params=roboflow_detection_v2_dataset_params training_hyperparams.lr_warmup_epochs=3 training_hyperparams.warmup_mode=linear_epoch_step training_hyperparams.initial_lr=5e-4 training_hyperparams.ema=False experiment_suffix=0  &
#	CUDA_VISIBLE_DEVICES=1 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=cells-uyemf dataset_params=roboflow_detection_v2_dataset_params training_hyperparams.lr_warmup_epochs=3 training_hyperparams.warmup_mode=linear_epoch_step training_hyperparams.initial_lr=4e-4 training_hyperparams.ema=False experiment_suffix=1  &
#	CUDA_VISIBLE_DEVICES=2 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=cells-uyemf dataset_params=roboflow_detection_v2_dataset_params training_hyperparams.lr_warmup_epochs=3 training_hyperparams.warmup_mode=linear_epoch_step training_hyperparams.initial_lr=3e-4 training_hyperparams.ema=False experiment_suffix=2  &
#	CUDA_VISIBLE_DEVICES=3 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=cells-uyemf dataset_params=roboflow_detection_v2_dataset_params training_hyperparams.lr_warmup_epochs=3 training_hyperparams.warmup_mode=linear_epoch_step training_hyperparams.initial_lr=2e-4 training_hyperparams.ema=False experiment_suffix=3  &
#	CUDA_VISIBLE_DEVICES=4 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=cells-uyemf dataset_params=roboflow_detection_v2_dataset_params training_hyperparams.lr_warmup_epochs=3 training_hyperparams.warmup_mode=linear_epoch_step training_hyperparams.initial_lr=1e-4 training_hyperparams.ema=False experiment_suffix=4  &
#	CUDA_VISIBLE_DEVICES=5 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=cells-uyemf dataset_params=roboflow_detection_v2_dataset_params training_hyperparams.lr_warmup_epochs=3 training_hyperparams.warmup_mode=linear_epoch_step training_hyperparams.initial_lr=9e-5 training_hyperparams.ema=False experiment_suffix=5  &
#	CUDA_VISIBLE_DEVICES=6 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=cells-uyemf dataset_params=roboflow_detection_v2_dataset_params training_hyperparams.lr_warmup_epochs=3 training_hyperparams.warmup_mode=linear_epoch_step training_hyperparams.initial_lr=8e-5 training_hyperparams.ema=False experiment_suffix=6  &
#	CUDA_VISIBLE_DEVICES=7 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=cells-uyemf dataset_params=roboflow_detection_v2_dataset_params training_hyperparams.lr_warmup_epochs=3 training_hyperparams.warmup_mode=linear_epoch_step training_hyperparams.initial_lr=7e-5 training_hyperparams.ema=False experiment_suffix=7  &

	CUDA_VISIBLE_DEVICES=0 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=cells-uyemf dataset_params=roboflow_detection_v2_dataset_params training_hyperparams.lr_warmup_epochs=3 training_hyperparams.warmup_mode=linear_epoch_step training_hyperparams.initial_lr=5e-4 experiment_suffix=0  &
	CUDA_VISIBLE_DEVICES=1 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=cells-uyemf dataset_params=roboflow_detection_v2_dataset_params training_hyperparams.lr_warmup_epochs=3 training_hyperparams.warmup_mode=linear_epoch_step training_hyperparams.initial_lr=4e-4 experiment_suffix=1  &
	CUDA_VISIBLE_DEVICES=2 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=cells-uyemf dataset_params=roboflow_detection_v2_dataset_params training_hyperparams.lr_warmup_epochs=3 training_hyperparams.warmup_mode=linear_epoch_step training_hyperparams.initial_lr=3e-4 experiment_suffix=2  &
	CUDA_VISIBLE_DEVICES=3 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=cells-uyemf dataset_params=roboflow_detection_v2_dataset_params training_hyperparams.lr_warmup_epochs=3 training_hyperparams.warmup_mode=linear_epoch_step training_hyperparams.initial_lr=2e-4 experiment_suffix=3  &
	CUDA_VISIBLE_DEVICES=4 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=cells-uyemf dataset_params=roboflow_detection_v2_dataset_params training_hyperparams.lr_warmup_epochs=3 training_hyperparams.warmup_mode=linear_epoch_step training_hyperparams.initial_lr=1e-4 experiment_suffix=4  &
	CUDA_VISIBLE_DEVICES=5 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=cells-uyemf dataset_params=roboflow_detection_v2_dataset_params training_hyperparams.lr_warmup_epochs=3 training_hyperparams.warmup_mode=linear_epoch_step training_hyperparams.initial_lr=9e-5 experiment_suffix=5  &
	CUDA_VISIBLE_DEVICES=6 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=cells-uyemf dataset_params=roboflow_detection_v2_dataset_params training_hyperparams.lr_warmup_epochs=3 training_hyperparams.warmup_mode=linear_epoch_step training_hyperparams.initial_lr=8e-5 experiment_suffix=6  &
	CUDA_VISIBLE_DEVICES=7 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=cells-uyemf dataset_params=roboflow_detection_v2_dataset_params training_hyperparams.lr_warmup_epochs=3 training_hyperparams.warmup_mode=linear_epoch_step training_hyperparams.initial_lr=7e-5 experiment_suffix=7  &

bacteria-ptywi:
#	CUDA_VISIBLE_DEVICES=0 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=bacteria-ptywi dataset_params=roboflow_detection_v2_dataset_params training_hyperparams.lr_warmup_epochs=3 training_hyperparams.warmup_mode=linear_epoch_step training_hyperparams.initial_lr=5e-4 experiment_suffix=0  &
#	CUDA_VISIBLE_DEVICES=1 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=bacteria-ptywi dataset_params=roboflow_detection_v2_dataset_params training_hyperparams.lr_warmup_epochs=3 training_hyperparams.warmup_mode=linear_epoch_step training_hyperparams.initial_lr=4e-4 experiment_suffix=1  &
#	CUDA_VISIBLE_DEVICES=2 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=bacteria-ptywi dataset_params=roboflow_detection_v2_dataset_params training_hyperparams.lr_warmup_epochs=3 training_hyperparams.warmup_mode=linear_epoch_step training_hyperparams.initial_lr=3e-4 experiment_suffix=2  &
#	CUDA_VISIBLE_DEVICES=3 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=bacteria-ptywi dataset_params=roboflow_detection_v2_dataset_params training_hyperparams.lr_warmup_epochs=3 training_hyperparams.warmup_mode=linear_epoch_step training_hyperparams.initial_lr=2e-4 experiment_suffix=3  &
#	CUDA_VISIBLE_DEVICES=4 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=bacteria-ptywi dataset_params=roboflow_detection_v2_dataset_params training_hyperparams.lr_warmup_epochs=3 training_hyperparams.warmup_mode=linear_epoch_step training_hyperparams.initial_lr=1e-4 experiment_suffix=4  &
#	CUDA_VISIBLE_DEVICES=5 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=bacteria-ptywi dataset_params=roboflow_detection_v2_dataset_params training_hyperparams.lr_warmup_epochs=3 training_hyperparams.warmup_mode=linear_epoch_step training_hyperparams.initial_lr=9e-5 experiment_suffix=5  &
#	CUDA_VISIBLE_DEVICES=6 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=bacteria-ptywi dataset_params=roboflow_detection_v2_dataset_params training_hyperparams.lr_warmup_epochs=3 training_hyperparams.warmup_mode=linear_epoch_step training_hyperparams.initial_lr=8e-5 experiment_suffix=6  &
#	CUDA_VISIBLE_DEVICES=7 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=bacteria-ptywi dataset_params=roboflow_detection_v2_dataset_params training_hyperparams.lr_warmup_epochs=3 training_hyperparams.warmup_mode=linear_epoch_step training_hyperparams.initial_lr=7e-5 experiment_suffix=7  &

	CUDA_VISIBLE_DEVICES=0 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=bacteria-ptywi dataset_params=roboflow_detection_v2_dataset_params training_hyperparams.lr_warmup_epochs=3 training_hyperparams.warmup_mode=linear_epoch_step training_hyperparams.initial_lr=5e-4 training_hyperparams.ema=False  experiment_suffix=0  &
	CUDA_VISIBLE_DEVICES=1 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=bacteria-ptywi dataset_params=roboflow_detection_v2_dataset_params training_hyperparams.lr_warmup_epochs=3 training_hyperparams.warmup_mode=linear_epoch_step training_hyperparams.initial_lr=4e-4 training_hyperparams.ema=False  experiment_suffix=1  &
	CUDA_VISIBLE_DEVICES=2 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=bacteria-ptywi dataset_params=roboflow_detection_v2_dataset_params training_hyperparams.lr_warmup_epochs=3 training_hyperparams.warmup_mode=linear_epoch_step training_hyperparams.initial_lr=3e-4 training_hyperparams.ema=False  experiment_suffix=2  &
	CUDA_VISIBLE_DEVICES=3 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=bacteria-ptywi dataset_params=roboflow_detection_v2_dataset_params training_hyperparams.lr_warmup_epochs=3 training_hyperparams.warmup_mode=linear_epoch_step training_hyperparams.initial_lr=2e-4 training_hyperparams.ema=False  experiment_suffix=3  &
	CUDA_VISIBLE_DEVICES=4 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=bacteria-ptywi dataset_params=roboflow_detection_v2_dataset_params training_hyperparams.lr_warmup_epochs=3 training_hyperparams.warmup_mode=linear_epoch_step training_hyperparams.initial_lr=1e-4 training_hyperparams.ema=False  experiment_suffix=4  &
	CUDA_VISIBLE_DEVICES=5 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=bacteria-ptywi dataset_params=roboflow_detection_v2_dataset_params training_hyperparams.lr_warmup_epochs=3 training_hyperparams.warmup_mode=linear_epoch_step training_hyperparams.initial_lr=9e-5 training_hyperparams.ema=False  experiment_suffix=5  &
	CUDA_VISIBLE_DEVICES=6 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=bacteria-ptywi dataset_params=roboflow_detection_v2_dataset_params training_hyperparams.lr_warmup_epochs=3 training_hyperparams.warmup_mode=linear_epoch_step training_hyperparams.initial_lr=8e-5 training_hyperparams.ema=False  experiment_suffix=6  &
#	CUDA_VISIBLE_DEVICES=7 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=bacteria-ptywi dataset_params=roboflow_detection_v2_dataset_params training_hyperparams.lr_warmup_epochs=3 training_hyperparams.warmup_mode=linear_epoch_step training_hyperparams.initial_lr=7e-5 training_hyperparams.ema=False  experiment_suffix=7  &

with-min_samples:
	#CUDA_VISIBLE_DEVICES=1 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m_with_min_samples dataset_name=bacteria-ptywi dataset_params=roboflow_detection_v2_dataset_params training_hyperparams.lr_warmup_epochs=3 training_hyperparams.warmup_mode=linear_epoch_step training_hyperparams.initial_lr=5e-4 experiment_suffix=1  &
	#CUDA_VISIBLE_DEVICES=0 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m_with_min_samples dataset_name=cells-uyemf    dataset_params=roboflow_detection_v2_dataset_params training_hyperparams.lr_warmup_epochs=3 training_hyperparams.warmup_mode=linear_epoch_step training_hyperparams.initial_lr=5e-4 experiment_suffix=0  &

	CUDA_VISIBLE_DEVICES=2 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m_with_min_samples dataset_name=cells-uyemf    dataset_params=roboflow_detection_v2_dataset_params training_hyperparams.lr_warmup_epochs=3 training_hyperparams.warmup_mode=linear_epoch_step training_hyperparams.initial_lr=5e-4 experiment_suffix=0  &
	CUDA_VISIBLE_DEVICES=3 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m_with_min_samples dataset_name=cells-uyemf    dataset_params=roboflow_detection_v2_dataset_params training_hyperparams.lr_warmup_epochs=3 training_hyperparams.warmup_mode=linear_epoch_step training_hyperparams.initial_lr=5e-4 experiment_suffix=0  &
	CUDA_VISIBLE_DEVICES=4 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m_with_min_samples dataset_name=cells-uyemf    dataset_params=roboflow_detection_v2_dataset_params training_hyperparams.lr_warmup_epochs=3 training_hyperparams.warmup_mode=linear_epoch_step training_hyperparams.initial_lr=5e-4 experiment_suffix=0  &
	CUDA_VISIBLE_DEVICES=5 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m_with_min_samples dataset_name=cells-uyemf    dataset_params=roboflow_detection_v2_dataset_params training_hyperparams.lr_warmup_epochs=3 training_hyperparams.warmup_mode=linear_epoch_step training_hyperparams.initial_lr=5e-4 experiment_suffix=0  &
	CUDA_VISIBLE_DEVICES=6 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m_with_min_samples dataset_name=cells-uyemf    dataset_params=roboflow_detection_v2_dataset_params training_hyperparams.lr_warmup_epochs=3 training_hyperparams.warmup_mode=linear_epoch_step training_hyperparams.initial_lr=5e-4 experiment_suffix=0  &
	CUDA_VISIBLE_DEVICES=7 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m_with_min_samples dataset_name=cells-uyemf    dataset_params=roboflow_detection_v2_dataset_params training_hyperparams.lr_warmup_epochs=3 training_hyperparams.warmup_mode=linear_epoch_step training_hyperparams.initial_lr=5e-4 experiment_suffix=0  &


# First machine

tzag_8_batch_0:
	CUDA_VISIBLE_DEVICES=0 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=aerial-pool
	CUDA_VISIBLE_DEVICES=0 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=aquarium-qlnqy
	CUDA_VISIBLE_DEVICES=0 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=insects-mytwu
	CUDA_VISIBLE_DEVICES=0 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=robomasters-285km
	CUDA_VISIBLE_DEVICES=0 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=solar-panels-taxvb
	CUDA_VISIBLE_DEVICES=0 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=truck-movement
	CUDA_VISIBLE_DEVICES=0 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=valentines-chocolate
	CUDA_VISIBLE_DEVICES=0 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=wall-damage

tzag_8_batch_2:
	CUDA_VISIBLE_DEVICES=2 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=aerial-spheres
	CUDA_VISIBLE_DEVICES=2 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=bccd-ouzjz
	CUDA_VISIBLE_DEVICES=2 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=digits-t2eg6
	CUDA_VISIBLE_DEVICES=2 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=excavators-czvg9
	CUDA_VISIBLE_DEVICES=2 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=furniture-ngpea
	CUDA_VISIBLE_DEVICES=2 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=lettuce-pallets
	CUDA_VISIBLE_DEVICES=2 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=thermal-cheetah-my4dp
	CUDA_VISIBLE_DEVICES=2 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=underwater-objects-5v7p8

tzag_8_batch_3:
	CUDA_VISIBLE_DEVICES=3 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=cable-damage
	CUDA_VISIBLE_DEVICES=3 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=chess-pieces-mjzgj
	CUDA_VISIBLE_DEVICES=3 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=coral-lwptl
	CUDA_VISIBLE_DEVICES=3 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=parasites-1s07h
	CUDA_VISIBLE_DEVICES=3 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=shark-teeth-5atku
	CUDA_VISIBLE_DEVICES=3 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=soccer-players-5fuqs
	CUDA_VISIBLE_DEVICES=3 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=vehicles-q0x2v
	CUDA_VISIBLE_DEVICES=3 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=x-ray-rheumatology

tzag_8_batch_4:
	CUDA_VISIBLE_DEVICES=4 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=acl-x-ray
	CUDA_VISIBLE_DEVICES=4 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=animals-ij5d2
	CUDA_VISIBLE_DEVICES=4 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=bees-jt5in
	CUDA_VISIBLE_DEVICES=4 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=cells-uyemf
	CUDA_VISIBLE_DEVICES=4 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=peanuts-sd4kf
	CUDA_VISIBLE_DEVICES=4 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=road-signs-6ih4y
	CUDA_VISIBLE_DEVICES=4 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=tweeter-posts
	CUDA_VISIBLE_DEVICES=4 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=weed-crop-aerial

tzag_8_batch_5:
	CUDA_VISIBLE_DEVICES=5 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=4-fold-defect
	CUDA_VISIBLE_DEVICES=5 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=abdomen-mri
	CUDA_VISIBLE_DEVICES=5 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=aerial-cows
	CUDA_VISIBLE_DEVICES=5 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=coins-1apki
	CUDA_VISIBLE_DEVICES=5 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=printed-circuit-board
	CUDA_VISIBLE_DEVICES=5 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=sedimentary-features-9eosf
	CUDA_VISIBLE_DEVICES=5 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=street-work
	CUDA_VISIBLE_DEVICES=5 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=tweeter-profile

tzag_8_batch_6:
	CUDA_VISIBLE_DEVICES=6 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=apples-fvpl5
	CUDA_VISIBLE_DEVICES=6 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=axial-mri
	CUDA_VISIBLE_DEVICES=6 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=bacteria-ptywi
	CUDA_VISIBLE_DEVICES=6 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=cloud-types
	CUDA_VISIBLE_DEVICES=6 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=construction-safety-gsnvb
	CUDA_VISIBLE_DEVICES=6 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=currency-v4f8j
	CUDA_VISIBLE_DEVICES=6 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=document-parts
	CUDA_VISIBLE_DEVICES=6 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=pests-2xlvx

tzag_8_batch_7:
	CUDA_VISIBLE_DEVICES=7 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=activity-diagrams-qdobr
	CUDA_VISIBLE_DEVICES=7 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=apex-videogame
	CUDA_VISIBLE_DEVICES=7 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=cavity-rs0uf
	CUDA_VISIBLE_DEVICES=7 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=cotton-plant-disease
	CUDA_VISIBLE_DEVICES=7 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=gynecology-mri
	CUDA_VISIBLE_DEVICES=7 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=people-in-paintings
	CUDA_VISIBLE_DEVICES=7 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=road-traffic
	CUDA_VISIBLE_DEVICES=7 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=trail-camera

# Second machine
tzag_5_batch_0:
	CUDA_VISIBLE_DEVICES=0 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=corrosion-bi3q3
	CUDA_VISIBLE_DEVICES=0 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=cotton-20xz5
	CUDA_VISIBLE_DEVICES=0 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=fish-market-ggjso
	CUDA_VISIBLE_DEVICES=0 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=peixos-fish
	CUDA_VISIBLE_DEVICES=0 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=radio-signal
	CUDA_VISIBLE_DEVICES=0 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=secondary-chains
	CUDA_VISIBLE_DEVICES=0 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=stomata-cells
	CUDA_VISIBLE_DEVICES=0 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=wine-labels

tzag_5_batch_1:
	CUDA_VISIBLE_DEVICES=1 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=brain-tumor-m2pbp
	CUDA_VISIBLE_DEVICES=1 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=cell-towers
	CUDA_VISIBLE_DEVICES=1 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=circuit-voltages
	CUDA_VISIBLE_DEVICES=1 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=farcry6-videogame
	CUDA_VISIBLE_DEVICES=1 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=leaf-disease-nsdsr
	CUDA_VISIBLE_DEVICES=1 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=mitosis-gjs3g
	CUDA_VISIBLE_DEVICES=1 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=signatures-xc8up
	CUDA_VISIBLE_DEVICES=1 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=washroom-rf1fa

tzag_5_batch_2:
	CUDA_VISIBLE_DEVICES=2 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=bone-fracture-7fylg
	CUDA_VISIBLE_DEVICES=2 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=circuit-elements
	CUDA_VISIBLE_DEVICES=2 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=csgo-videogame
	CUDA_VISIBLE_DEVICES=2 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=marbles
	CUDA_VISIBLE_DEVICES=2 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=paper-parts
	CUDA_VISIBLE_DEVICES=2 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=phages
	CUDA_VISIBLE_DEVICES=2 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=uno-deck

tzag_5_batch_4:
	CUDA_VISIBLE_DEVICES=4 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=avatar-recognition-nuexe
	CUDA_VISIBLE_DEVICES=4 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=flir-camera-objects
	CUDA_VISIBLE_DEVICES=4 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=grass-weeds
	CUDA_VISIBLE_DEVICES=4 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=liver-disease
	CUDA_VISIBLE_DEVICES=4 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=mask-wearing-608pr
	CUDA_VISIBLE_DEVICES=4 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=pills-sxdht
	CUDA_VISIBLE_DEVICES=4 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=tabular-data-wf9uh

tzag_5_batch_5:
	CUDA_VISIBLE_DEVICES=5 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=asbestos
	CUDA_VISIBLE_DEVICES=5 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=cables-nl42k
	CUDA_VISIBLE_DEVICES=5 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=gauge-u2lwv
	CUDA_VISIBLE_DEVICES=5 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=halo-infinite-angel-videogame
	CUDA_VISIBLE_DEVICES=5 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=hand-gestures-jps7z
	CUDA_VISIBLE_DEVICES=5 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=paragraphs-co84b
	CUDA_VISIBLE_DEVICES=5 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=poker-cards-cxcvz

tzag_5_batch_6:
	CUDA_VISIBLE_DEVICES=6 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=number-ops
	CUDA_VISIBLE_DEVICES=6 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=sign-language-sokdr
	CUDA_VISIBLE_DEVICES=6 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=smoke-uvylj
	CUDA_VISIBLE_DEVICES=6 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=soda-bottles
	CUDA_VISIBLE_DEVICES=6 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=team-fight-tactics
	CUDA_VISIBLE_DEVICES=6 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=thermal-dogs-and-people-x6ejw
	CUDA_VISIBLE_DEVICES=6 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_m dataset_name=underwater-pipes-4ng4t
