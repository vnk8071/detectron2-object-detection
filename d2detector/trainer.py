import os
from detectron2.data.transforms.augmentation import AugInput
from detectron2.data.transforms.augmentation_impl import Resize
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader, build_detection_test_loader, DatasetMapper
from detectron2.evaluation import COCOEvaluator
import detectron2.data.transforms as T
from detectron2.data import detection_utils as utils
import copy
import torch


def build_train_aug(dataset_dict):
    # Get image and bounding box from dataset
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict["file_name"])
    boxes = [obj["bbox"] for obj in dataset_dict["annotations"]]

    # Data augmentation
    transform_list = T.AugmentationList([
        T.ResizeShortestEdge((480, ), 720, sample_style='choice'),
        T.RandomFlip(prob=0.8, horizontal=True),
        T.RandomCrop(crop_type='relative', crop_size=[0.9, 0.9]),
    ])
    aug_input = AugInput(image, boxes=boxes)
    transforms = transform_list(aug_input)
    image, boxes = aug_input.image, aug_input.boxes
    dataset_dict["image"] = torch.as_tensor(
        image.transpose(2, 0, 1).astype("float32"))
    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict["annotations"]
        if obj.get("iscrowd", 0) == 0
    ]

    # Convert annotations to instances
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict


class D2Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    # @classmethod
    # def build_train_loader(cls, cfg):
    #     return build_detection_train_loader(cfg, mapper=build_train_aug)
