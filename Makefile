batch_0:
	CUDA_VISIBLE_DEVICES=0 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=aerial-pool
	CUDA_VISIBLE_DEVICES=0 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=aquarium-qlnqy
	CUDA_VISIBLE_DEVICES=0 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=insects-mytwu
	CUDA_VISIBLE_DEVICES=0 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=robomasters-285km
	CUDA_VISIBLE_DEVICES=0 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=solar-panels-taxvb
	CUDA_VISIBLE_DEVICES=0 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=truck-movement
	CUDA_VISIBLE_DEVICES=0 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=valentines-chocolate
	CUDA_VISIBLE_DEVICES=0 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=wall-damage

batch_1:
	CUDA_VISIBLE_DEVICES=1 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=corrosion-bi3q3
	CUDA_VISIBLE_DEVICES=1 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=cotton-20xz5
	CUDA_VISIBLE_DEVICES=1 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=fish-market-ggjso
	CUDA_VISIBLE_DEVICES=1 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=peixos-fish
	CUDA_VISIBLE_DEVICES=1 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=radio-signal
	CUDA_VISIBLE_DEVICES=1 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=secondary-chains
	CUDA_VISIBLE_DEVICES=1 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=stomata-cells
	CUDA_VISIBLE_DEVICES=1 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=wine-labels

batch_2:
	CUDA_VISIBLE_DEVICES=2 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=aerial-spheres
	CUDA_VISIBLE_DEVICES=2 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=bccd-ouzjz
	CUDA_VISIBLE_DEVICES=2 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=digits-t2eg6
	CUDA_VISIBLE_DEVICES=2 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=excavators-czvg9
	CUDA_VISIBLE_DEVICES=2 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=furniture-ngpea
	CUDA_VISIBLE_DEVICES=2 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=lettuce-pallets
	CUDA_VISIBLE_DEVICES=2 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=thermal-cheetah-my4dp
	CUDA_VISIBLE_DEVICES=2 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=underwater-objects-5v7p8

batch_3:
	CUDA_VISIBLE_DEVICES=3 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=cable-damage
	CUDA_VISIBLE_DEVICES=3 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=chess-pieces-mjzgj
	CUDA_VISIBLE_DEVICES=3 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=coral-lwptl
	CUDA_VISIBLE_DEVICES=3 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=parasites-1s07h
	CUDA_VISIBLE_DEVICES=3 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=shark-teeth-5atku
	CUDA_VISIBLE_DEVICES=3 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=soccer-players-5fuqs
	CUDA_VISIBLE_DEVICES=3 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=vehicles-q0x2v
	CUDA_VISIBLE_DEVICES=3 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=x-ray-rheumatology

batch_4:
	CUDA_VISIBLE_DEVICES=4 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=acl-x-ray
	CUDA_VISIBLE_DEVICES=4 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=animals-ij5d2
	CUDA_VISIBLE_DEVICES=4 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=bees-jt5in
	CUDA_VISIBLE_DEVICES=4 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=cells-uyemf
	CUDA_VISIBLE_DEVICES=4 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=peanuts-sd4kf
	CUDA_VISIBLE_DEVICES=4 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=road-signs-6ih4y
	CUDA_VISIBLE_DEVICES=4 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=tweeter-posts
	CUDA_VISIBLE_DEVICES=4 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=weed-crop-aerial

batch_5:
	CUDA_VISIBLE_DEVICES=5 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=4-fold-defect
	CUDA_VISIBLE_DEVICES=5 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=abdomen-mri
	CUDA_VISIBLE_DEVICES=5 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=aerial-cows
	CUDA_VISIBLE_DEVICES=5 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=coins-1apki
	CUDA_VISIBLE_DEVICES=5 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=printed-circuit-board
	CUDA_VISIBLE_DEVICES=5 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=sedimentary-features-9eosf
	CUDA_VISIBLE_DEVICES=5 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=street-work
	CUDA_VISIBLE_DEVICES=5 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=tweeter-profile

batch_6:
	CUDA_VISIBLE_DEVICES=6 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=apples-fvpl5
	CUDA_VISIBLE_DEVICES=6 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=axial-mri
	CUDA_VISIBLE_DEVICES=6 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=bacteria-ptywi
	CUDA_VISIBLE_DEVICES=6 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=cloud-types
	CUDA_VISIBLE_DEVICES=6 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=construction-safety-gsnvb
	CUDA_VISIBLE_DEVICES=6 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=currency-v4f8j
	CUDA_VISIBLE_DEVICES=6 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=document-parts
	CUDA_VISIBLE_DEVICES=6 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=pests-2xlvx

batch_7:
	CUDA_VISIBLE_DEVICES=7 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=activity-diagrams-qdobr
	CUDA_VISIBLE_DEVICES=7 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=apex-videogame
	CUDA_VISIBLE_DEVICES=7 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=cavity-rs0uf
	CUDA_VISIBLE_DEVICES=7 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=cotton-plant-disease
	CUDA_VISIBLE_DEVICES=7 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=gynecology-mri
	CUDA_VISIBLE_DEVICES=7 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=people-in-paintings
	CUDA_VISIBLE_DEVICES=7 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=road-traffic
	CUDA_VISIBLE_DEVICES=7 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=trail-camera

batch_8:
	CUDA_VISIBLE_DEVICES=8 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=brain-tumor-m2pbp
	CUDA_VISIBLE_DEVICES=8 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=cell-towers
	CUDA_VISIBLE_DEVICES=8 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=circuit-voltages
	CUDA_VISIBLE_DEVICES=8 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=farcry6-videogame
	CUDA_VISIBLE_DEVICES=8 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=leaf-disease-nsdsr
	CUDA_VISIBLE_DEVICES=8 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=mitosis-gjs3g
	CUDA_VISIBLE_DEVICES=8 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=signatures-xc8up
	CUDA_VISIBLE_DEVICES=8 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=washroom-rf1fa

batch_9:
	CUDA_VISIBLE_DEVICES=9 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=bone-fracture-7fylg
	CUDA_VISIBLE_DEVICES=9 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=circuit-elements
	CUDA_VISIBLE_DEVICES=9 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=csgo-videogame
	CUDA_VISIBLE_DEVICES=9 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=marbles
	CUDA_VISIBLE_DEVICES=9 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=paper-parts
	CUDA_VISIBLE_DEVICES=9 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=phages
	CUDA_VISIBLE_DEVICES=9 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=uno-deck

batch_10:
	CUDA_VISIBLE_DEVICES=10 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=avatar-recognition-nuexe
	CUDA_VISIBLE_DEVICES=10 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=flir-camera-objects
	CUDA_VISIBLE_DEVICES=10 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=grass-weeds
	CUDA_VISIBLE_DEVICES=10 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=liver-disease
	CUDA_VISIBLE_DEVICES=10 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=mask-wearing-608pr
	CUDA_VISIBLE_DEVICES=10 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=pills-sxdht
	CUDA_VISIBLE_DEVICES=10 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=tabular-data-wf9uh

batch_11:
	CUDA_VISIBLE_DEVICES=11 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=asbestos
	CUDA_VISIBLE_DEVICES=11 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=cables-nl42k
	CUDA_VISIBLE_DEVICES=11 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=gauge-u2lwv
	CUDA_VISIBLE_DEVICES=11 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=halo-infinite-angel-videogame
	CUDA_VISIBLE_DEVICES=11 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=hand-gestures-jps7z
	CUDA_VISIBLE_DEVICES=11 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=paragraphs-co84b
	CUDA_VISIBLE_DEVICES=11 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=poker-cards-cxcvz

batch_12:
	CUDA_VISIBLE_DEVICES=12 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=number-ops
	CUDA_VISIBLE_DEVICES=12 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=sign-language-sokdr
	CUDA_VISIBLE_DEVICES=12 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=smoke-uvylj
	CUDA_VISIBLE_DEVICES=12 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=soda-bottles
	CUDA_VISIBLE_DEVICES=12 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=team-fight-tactics
	CUDA_VISIBLE_DEVICES=12 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=thermal-dogs-and-people-x6ejw
	CUDA_VISIBLE_DEVICES=12 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=underwater-pipes-4ng4t
