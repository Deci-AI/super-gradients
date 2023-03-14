import importlib
import sys

from omegaconf import OmegaConf

from super_gradients.common.environment.checkpoints_dir_utils import get_checkpoints_dir_path


def get_cls(cls_path: str):
    """
    A resolver for Hydra/OmegaConf to allow getting a class instead on an instance.
    usage:
    class_of_optimizer: ${class:torch.optim.Adam}
    """
    module = ".".join(cls_path.split(".")[:-1])
    name = cls_path.split(".")[-1]
    importlib.import_module(module)
    return getattr(sys.modules[module], name)


def hydra_output_dir_resolver(ckpt_root_dir: str, experiment_name: str) -> str:
    return get_checkpoints_dir_path(experiment_name=experiment_name, ckpt_root_dir=ckpt_root_dir)


ROBOFLOW_DATASETS = {
    "aerial-pool": {"expected": 7, "found": 6},
    "secondary-chains": {"expected": 1, "found": 2},
    "aerial-spheres": {"expected": 6, "found": 7},
    "soccer-players-5fuqs": {"expected": 3, "found": 4},
    "weed-crop-aerial": {"expected": 2, "found": 3},
    "aerial-cows": {"expected": 1, "found": 2},
    "cloud-types": {"expected": 4, "found": 5},
    "apex-videogame": {"expected": 2, "found": 3},
    "farcry6-videogame": {"expected": 11, "found": 12},
    "csgo-videogame": {"expected": 2, "found": 3},
    "avatar-recognition-nuexe": {"expected": 1, "found": 2},
    "halo-infinite-angel-videogame": {"expected": 4, "found": 5},
    "team-fight-tactics": {"expected": 59, "found": 60},
    "robomasters-285km": {"expected": 9, "found": 10},
    "stomata-cells": {"expected": 2, "found": 3},
    "bccd-ouzjz": {"expected": 3, "found": 4},
    "parasites-1s07h": {"expected": 8, "found": 9},
    "cells-uyemf": {"expected": 1, "found": 2},
    "4-fold-defect": {"expected": 1, "found": 2},
    "bacteria-ptywi": {"expected": 1, "found": 2},
    "cotton-plant-disease": {"expected": 1, "found": 2},
    "mitosis-gjs3g": {"expected": 1, "found": 2},
    "phages": {"expected": 2, "found": 3},
    "liver-disease": {"expected": 4, "found": 5},
    "asbestos": {"expected": 4, "found": 5},
    "underwater-pipes-4ng4t": {"expected": 1, "found": 2},
    "aquarium-qlnqy": {"expected": 7, "found": 8},
    "peixos-fish": {"expected": 12, "found": 3},
    "underwater-objects-5v7p8": {"expected": 5, "found": 6},
    "coral-lwptl": {"expected": 14, "found": 15},
    "tweeter-posts": {"expected": 2, "found": 3},
    "tweeter-profile": {"expected": 1, "found": None},
    "document-parts": {"expected": 2, "found": 3},
    "activity-diagrams-qdobr": {"expected": 19, "found": 20},
    "signatures-xc8up": {"expected": 1, "found": 2},
    "paper-parts": {"expected": 46, "found": 20},
    "tabular-data-wf9uh": {"expected": 12, "found": 13},
    "paragraphs-co84b": {"expected": 7, "found": 8},
    "thermal-dogs-and-people-x6ejw": {"expected": 2, "found": 3},
    "solar-panels-taxvb": {"expected": 5, "found": 6},
    "radio-signal": {"expected": 2, "found": 3},
    "thermal-cheetah-my4dp": {"expected": 2, "found": 3},
    "x-ray-rheumatology": {"expected": 12, "found": 13},
    "acl-x-ray": {"expected": 1, "found": 2},
    "abdomen-mri": {"expected": 1, "found": 2},
    "axial-mri": {"expected": 2, "found": 3},
    "gynecology-mri": {"expected": 3, "found": 4},
    "brain-tumor-m2pbp": {"expected": 3, "found": 4},
    "bone-fracture-7fylg": {"expected": 4, "found": 5},
    "flir-camera-objects": {"expected": 4, "found": 5},
    "hand-gestures-jps7z": {"expected": 14, "found": 15},
    "smoke-uvylj": {"expected": 1, "found": 2},
    "wall-damage": {"expected": 3, "found": 4},
    "corrosion-bi3q3": {"expected": 3, "found": 4},
    "excavators-czvg9": {"expected": 3, "found": 4},
    "chess-pieces-mjzgj": {"expected": 13, "found": 14},
    "road-signs-6ih4y": {"expected": 21, "found": 22},
    "street-work": {"expected": 11, "found": 9},
    "construction-safety-gsnvb": {"expected": 5, "found": 6},
    "road-traffic": {"expected": 12, "found": 8},
    "washroom-rf1fa": {"expected": 10, "found": 11},
    "circuit-elements": {"expected": 46, "found": 32},
    "mask-wearing-608pr": {"expected": 2, "found": 3},
    "cables-nl42k": {"expected": 11, "found": 12},
    "soda-bottles": {"expected": 6, "found": 4},
    "truck-movement": {"expected": 7, "found": 6},
    "wine-labels": {"expected": 12, "found": 13},
    "digits-t2eg6": {"expected": 10, "found": 11},
    "vehicles-q0x2v": {"expected": 12, "found": 13},
    "peanuts-sd4kf": {"expected": 2, "found": 3},
    "printed-circuit-board": {"expected": 34, "found": 24},
    "pests-2xlvx": {"expected": 28, "found": 29},
    "cavity-rs0uf": {"expected": 2, "found": 3},
    "leaf-disease-nsdsr": {"expected": 3, "found": 4},
    "marbles": {"expected": 2, "found": 3},
    "pills-sxdht": {"expected": 8, "found": 9},
    "poker-cards-cxcvz": {"expected": 53, "found": 54},
    "number-ops": {"expected": 15, "found": 16},
    "insects-mytwu": {"expected": 10, "found": 11},
    "cotton-20xz5": {"expected": 4, "found": 5},
    "furniture-ngpea": {"expected": 3, "found": 4},
    "cable-damage": {"expected": 2, "found": 3},
    "animals-ij5d2": {"expected": 10, "found": 11},
    "coins-1apki": {"expected": 4, "found": 5},
    "apples-fvpl5": {"expected": 2, "found": 3},
    "people-in-paintings": {"expected": 1, "found": 2},
    "circuit-voltages": {"expected": 6, "found": 7},
    "uno-deck": {"expected": 15, "found": 16},
    "grass-weeds": {"expected": 1, "found": 2},
    "gauge-u2lwv": {"expected": 2, "found": 3},
    "sign-language-sokdr": {"expected": 26, "found": 27},
    "valentines-chocolate": {"expected": 22, "found": 23},
    "fish-market-ggjso": {"expected": 21, "found": 20},
    "lettuce-pallets": {"expected": 5, "found": 6},
    "shark-teeth-5atku": {"expected": 4, "found": 5},
    "bees-jt5in": {"expected": 1, "found": 2},
    "sedimentary-features-9eosf": {"expected": 5, "found": 6},
    "currency-v4f8j": {"expected": 10, "found": 11},
    "trail-camera": {"expected": 2, "found": 3},
    "cell-towers": {"expected": 2, "found": 3},
}


def register_hydra_resolvers():
    """Register all the hydra resolvers required for the super-gradients recipes."""
    OmegaConf.register_new_resolver("hydra_output_dir", hydra_output_dir_resolver, replace=True)
    OmegaConf.register_new_resolver("class", lambda *args: get_cls(*args), replace=True)
    OmegaConf.register_new_resolver("add", lambda *args: sum(args), replace=True)
    OmegaConf.register_new_resolver("cond", lambda boolean, x, y: x if boolean else y, replace=True)
    OmegaConf.register_new_resolver("getitem", lambda container, key: container[key], replace=True)  # get item from a container (list, dict...)
    OmegaConf.register_new_resolver("first", lambda lst: lst[0], replace=True)  # get the first item from a list
    OmegaConf.register_new_resolver("last", lambda lst: lst[-1], replace=True)  # get the last item from a list

    def get_n(dataset_name):
        found = ROBOFLOW_DATASETS[dataset_name]["found"]
        return found

    OmegaConf.register_new_resolver("roboflow_dataset_num_classes", get_n, replace=True)  # get the last item from a list
