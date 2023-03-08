docker:
	docker build --build-arg CACHEBUST=$(date +%s) -t supergradients_debug_workers .

test: docker
	docker run --gpus '"device=0,1,2,3"' --rm --shm-size=64gb -v /data/coco/:/data/coco/:ro supergradients_debug_workers /bin/bash -c "python -m super_gradients.examples.train_from_recipe_example.train_from_recipe --config-name=coco2017_yolox_fast_det"
