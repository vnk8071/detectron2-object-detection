from detectron2.config import get_cfg
from detectron2.engine import launch
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets import coco as cc
from d2detector.config import add_d2trainer_config
from d2detector.trainer import D2Trainer

from pycocotools.coco import COCO
import argparse

# Handling arguments
parser = argparse.ArgumentParser()
parser.add_argument('--datapath_train', help='Path to train data',
                    default='dataset/train.json')
parser.add_argument('--datapath_test', help='Path to test data',
                    default='dataset/test.json')
parser.add_argument('--imagepath', help='Path to images',
                    default='dataset/images')
args = parser.parse_args()

annpath_train = args.datapath_train
annpath_test = args.datapath_test
imagepath = args.imagepath

coco = COCO(annpath_train)
cats = coco.loadCats(coco.getCatIds())


def setup():
    """
    Create configs and perform basic setups
    """
    # Dataset metadata
    register_coco_instances("train", {},
                            annpath_train,
                            imagepath)
    register_coco_instances("test", {},
                            annpath_test,
                            imagepath)
    train_metadata = MetadataCatalog.get("train")
    test_metadata = MetadataCatalog.get("test")

    # Display dataset
    print(train_metadata)
    print(test_metadata)
    # Config
    cfg = get_cfg()
    add_d2trainer_config(cfg)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(cats) - 1  # Remove background
    cfg.DATASETS.TRAIN = ("train",)
    # no metrics implemented for this dataset
    cfg.DATASETS.TEST = ("test",)
    return cfg


def main():
    cfg = setup()
    trainer = D2Trainer(cfg)
    trainer.resume_or_load(resume=False)
    return trainer.train()


if __name__ == '__main__':
    num_gpus_per_machine = 1
    launch(main, num_gpus_per_machine)
