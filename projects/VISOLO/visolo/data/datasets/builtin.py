# -*- coding: utf-8 -*-

import os

from .ytvis import (
    register_ytvis_instances,
    _get_ytvis_2019_instances_meta,
    _get_ytvis_2021_instances_meta,
    _get_ovis_instances_meta,
)
from detectron2.data.datasets.coco import register_coco_instances

# ==== Predefined splits for YTVIS 2019 ===========
_PREDEFINED_SPLITS_YTVIS_2019 = {
    "ytvis_2019_train": ("ytvis_2019/train/JPEGImages",
                         "ytvis_2019/train.json"),
    "ytvis_2019_val": ("ytvis_2019/valid/JPEGImages",
                       "ytvis_2019/valid.json"),
    "ytvis_2019_test": ("ytvis_2019/test/JPEGImages",
                        "ytvis_2019/test.json"),
    "ytvis_2019_dev": ("ytvis_2019/train/JPEGImages",
                       "ytvis_2019/instances_train_sub.json"),
    "ytvis_2019_all_val": ("ytvis_2019/valid_all_frames/JPEGImages",
                            "ytvis_2019/valid_all.json"),
}

_PREDEFINED_SPLITS_CUSTOM_COCO = {
    "coco_ytvis_2019_train": ("coco/train2017", "coco/annotations/coco_to_ytvis2019.json", 0),
    "coco_ytvis_2021_train": ("coco/train2017", "coco/annotations/coco_to_ytvis2021.json", 1),
    "coco_ovis_train": ("coco/train2017", "coco/annotations/coco_to_ovis.json", 2)
}

# ==== Predefined splits for YTVIS 2021 ===========
_PREDEFINED_SPLITS_YTVIS_2021 = {
    "ytvis_2021_train": ("ytvis_2021/train/JPEGImages",
                         "ytvis_2021/train.json"),
    "ytvis_2021_val": ("ytvis_2021/valid/JPEGImages",
                       "ytvis_2021/valid.json"),
    "ytvis_2021_test": ("ytvis_2021/test/JPEGImages",
                        "ytvis_2021/test.json"),
    "ytvis_2021_dev": ("ytvis_2021/train/JPEGImages",
                       "ytvis_2021/instances_train_sub.json"),
}


# ====    Predefined splits for OVIS    ===========
_PREDEFINED_SPLITS_OVIS = {
    "ovis_train": ("ovis/train",
                   "ovis/annotations/train.json"),
    "ovis_val": ("ovis/valid",
                 "ovis/annotations/valid.json"),
    "ovis_test": ("ovis/test",
                  "ovis/annotations/test.json"),
}


def register_all_ytvis_2019(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2019.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2019_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


def register_all_custom_coco(root):
    for key, (image_root, json_file, id) in _PREDEFINED_SPLITS_CUSTOM_COCO.items():
        if id == 0:
            register_coco_instances(
                key,
                _get_ytvis_2019_instances_meta(),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )
        elif id == 1:
            register_coco_instances(
                key,
                _get_ytvis_2021_instances_meta(),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )
        elif id == 2:
            register_coco_instances(
                key,
                _get_ovis_instances_meta(),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )


def register_all_ytvis_2021(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2021.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2021_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


def register_all_ovis(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_OVIS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ovis_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_ytvis_2019(_root)
    register_all_custom_coco(_root)
    register_all_ytvis_2021(_root)
    register_all_ovis(_root)
