#28               coins-1apki       real world   0.932   0.977   0.966     0.962727     0.877273            0.088727
#5                aerial-pool           aerial   0.513   0.791   0.744     0.583063     0.665535            0.078465
#4                aerial-cows           aerial   0.854   0.568   0.826     0.767991     0.774649            0.051351
#7              animals-ij5d2       real world   0.761   0.342   0.835     0.802735     0.787668            0.047332
#39         farcry6-videogame       videogames   0.619   0.216   0.709     0.694128     0.661946            0.047054
#87            truck-movement       real world   0.786   0.846   0.858     0.805831     0.811271            0.046729
#67              radio-signal  electromagnetic   0.673   0.653   0.695     0.658062     0.657557            0.037443
#90  underwater-objects-5v7p8       underwater   0.693   0.453   0.728     0.719367     0.692381            0.035619
#51             liver-disease      microscopic   0.592   0.583   0.576     0.534186     0.540495            0.035505
#58           parasites-1s07h      microscopic   0.848   0.889   0.861     0.851866     0.833404            0.027596
#
#
#Top worst performing datasets on deci_yolo_s
#         dataset_name         category  yolov5  yolov7  yolov8  deci_yolo_s  deci_yolo_m  deciyolo_s_from_v8
#5         aerial-pool           aerial   0.513   0.791   0.744     0.583063     0.665535            0.160937
#4         aerial-cows           aerial   0.854   0.568   0.826     0.767991     0.774649            0.058009
#87     truck-movement       real world   0.786   0.846   0.858     0.805831     0.811271            0.052169
#57   paragraphs-co84b        documents   0.626   0.610   0.653     0.609180     0.652811            0.043820
#51      liver-disease      microscopic   0.592   0.583   0.576     0.534186     0.540495            0.041814
#67       radio-signal  electromagnetic   0.673   0.653   0.695     0.658062     0.657557            0.036938
#70  robomasters-285km       videogames   0.816   0.772   0.814     0.780455     0.807799            0.033545
#7       animals-ij5d2       real world   0.761   0.342   0.835     0.802735     0.787668            0.032265
#11           asbestos      microscopic   0.596   0.611   0.631     0.602092     0.606690            0.028908
#94     vehicles-q0x2v       real world   0.454   0.464   0.472     0.444800     0.460311            0.027200

cells-uyemf:
	CUDA_VISIBLE_DEVICES=0 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=coins-1apki &
	CUDA_VISIBLE_DEVICES=1 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=aerial-pool &
	CUDA_VISIBLE_DEVICES=2 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=aerial-cows &
	CUDA_VISIBLE_DEVICES=3 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=animals-ij5d2 &
	CUDA_VISIBLE_DEVICES=4 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=truck-movement &
	CUDA_VISIBLE_DEVICES=5 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=farcry6-videogame &
	CUDA_VISIBLE_DEVICES=6 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=radio-signal &
	CUDA_VISIBLE_DEVICES=7 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=underwater-objects-5v7p8 &

	CUDA_VISIBLE_DEVICES=0 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=parasites-1s07h &
	CUDA_VISIBLE_DEVICES=1 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=liver-disease &
	CUDA_VISIBLE_DEVICES=2 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=vehicles-q0x2v &
	CUDA_VISIBLE_DEVICES=3 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=robomasters-285km &
	CUDA_VISIBLE_DEVICES=4 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=paragraphs-co84b &
	CUDA_VISIBLE_DEVICES=5 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=asbestos &
