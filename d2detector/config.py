from detectron2 import model_zoo
import os
from detectron2.config import get_cfg
import torch


def add_d2trainer_config(cfg):
    """
    Add config for d2detector
    """
    cfg.merge_from_file(model_zoo.get_config_file(
        'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml')

    # Data augmentation
    cfg.INPUT.MIN_SIZE_TRAIN = (240,)
    cfg.INPUT.MAX_SIZE_TRAIN = 480
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    cfg.INPUT.MIN_SIZE_TEST = 240
    cfg.INPUT.MAX_SIZE_TEST = 480

    cfg.INPUT.RANDOM_FLIP_PROBABILITY = 0.5
    cfg.INPUT.RANDOM_FLIP = "horizontal"

    cfg.INPUT.RANDOM_LIGHTNING = 0.7
    cfg.INPUT.RANDOM_BRIGHTNESS_RANGE = (0.8, 1.2)
    cfg.INPUT.RANDOM_CONTRAST_RANGE = (0.8, 1.2)
    cfg.INPUT.RANDOM_SATURATION_RANGE = (0.7, 1.3)
    cfg.INPUT.RANDOM_ROTATE_RANGE = [-10, 10]

    # Dataloader
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"

    # Model detail
    cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    # cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE = [True, True, True, True]
    # cfg.MODEL.RESNETS.DEFORM_MODULATED = True
    cfg.MODEL.BACKBONE.FREEZE_AT = 2
    cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA = 0.1
    cfg.MODEL.ROI_BOX_HEAD.NAME = "FastRCNNConvFCHead"
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3", "p4", "p5"]
    cfg.MODEL.RPN.IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]
    # cfg.MODEL.FPN.NORM = "GN"

    # Solver
    cfg.SOLVER.IMS_PER_BATCH = 32
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 2000
    cfg.SOLVER.STEPS = (1000,)
    cfg.SOLVER.WARMUP_ITERS = 500
    cfg.SOLVER.GAMMA = 0.05

    cfg.TEST.EVAL_PERIOD = 2000
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)


def add_d2test_config(cfg):
    """
    Add config for testing
    """
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    return cfg
