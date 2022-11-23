import os
import logging
import copy

import torch

import detectron2.data.transforms as T
from detectron2.data.transforms.augmentation import AugInput
from detectron2.data.transforms.augmentation_impl import Resize
from d2detector.defaults import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader, build_detection_test_loader, DatasetMapper
from detectron2.evaluation import COCOEvaluator
from detectron2.data import detection_utils as utils
from d2detector.mapper import DatasetMapperCustom


class D2Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        transform_list = [
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN, 
            cfg.INPUT.MAX_SIZE_TRAIN, 
            cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING),
        T.RandomBrightness(
            cfg.INPUT.RANDOM_BRIGHTNESS_RANGE[0],
            cfg.INPUT.RANDOM_BRIGHTNESS_RANGE[1]
        ),
        T.RandomContrast(
            cfg.INPUT.RANDOM_CONTRAST_RANGE[0], 
            cfg.INPUT.RANDOM_CONTRAST_RANGE[1]
            ),
        T.RandomSaturation(
            cfg.INPUT.RANDOM_SATURATION_RANGE[0],
            cfg.INPUT.RANDOM_SATURATION_RANGE[1]
            ),
        T.RandomRotation(angle=cfg.INPUT.RANDOM_ROTATE_RANGE),
        T.RandomLighting(cfg.INPUT.RANDOM_LIGHTNING),
        T.RandomFlip(
            prob=cfg.INPUT.RANDOM_FLIP_PROBABILITY, 
            horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal", 
            vertical=cfg.INPUT.RANDOM_FLIP == "vertical"),
        ]
        mapper = DatasetMapperCustom(cfg, is_train=True, augmentations=transform_list)
        return build_detection_train_loader(cfg, mapper=mapper)