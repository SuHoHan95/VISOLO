# -*- coding: utf-8 -*-
from detectron2.config import CfgNode as CN


def add_visolo_config(cfg):
    """
    Add config for VISOLO.
    """
    cfg.MODEL.VISOLO = CN()
    cfg.MODEL.VISOLO.NUM_CLASSES = 80

    cfg.DATASETS.DATASET_RATIO = [0.75, 0.25]

    # DataLoader
    cfg.INPUT.SAMPLING_FRAME_NUM = 3
    cfg.INPUT.SAMPLING_FRAME_RANGE = 20
    cfg.INPUT.SAMPLING_FRAME_SHUFFLE = True
    cfg.INPUT.SAMPLING_EPS = 0.2

    # "brightness", "contrast", "saturation", "rotation"
    cfg.INPUT.AUGMENTATIONS = ["brightness", "contrast", "saturation", "rotation"]
    cfg.INPUT.RANDOM_ROTATION = True

    # LOSS
    cfg.MODEL.VISOLO.MASK_WEIGHT = 3.0
    cfg.MODEL.VISOLO.FOCAL_LOSS_ALPHA = 0.25
    cfg.MODEL.VISOLO.FOCAL_LOSS_GAMMA = 2.
    cfg.MODEL.VISOLO.DICE_LOSS_EPS = 1e-7

    # ModelStructure
    cfg.MODEL.VISOLO.GRID_NUM = (12, 21)
    cfg.MODEL.VISOLO.INDIM = 1024
    cfg.MODEL.VISOLO.OUTDIM = 256
    cfg.MODEL.VISOLO.NORM = "GN"

    # Evaluation
    cfg.MODEL.VISOLO.TRACKING_THR = 0.1
    cfg.MODEL.VISOLO.SCORE_THR = 0.1
    cfg.MODEL.VISOLO.MASK_THR = 0.5
    cfg.MODEL.VISOLO.UPDATE_THR = 0.05
    cfg.MODEL.VISOLO.KERNEL = 'gaussian'
    cfg.MODEL.VISOLO.SIGMA = 2.
    cfg.MODEL.VISOLO.NMS_PRE = 500

    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1
