from cv2 import threshold
import torch
import detectron2.data.transforms as T
from detectron2.data import build_detection_test_loader, detection_utils
from detectron2.utils.env import TORCH_VERSION
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import GeneralizedRCNN, build_model
from detectron2.utils.file_io import PathManager
import argparse
from detectron2.export import (dump_torchscript_IR, TracingAdapter)
import os
from config import add_d2test_config
from detectron2.config import get_cfg

# source: https://github.com/facebookresearch/detectron2/blob/main/tools/deploy/export_model.py

parser = argparse.ArgumentParser(
    description="Export a model for deployment.")
parser.add_argument(
    "--format",
    type=str,
    help="output format",
    default="torchscript",
)
parser.add_argument(
    "--exportmethod",
    type=str,
    help="Method to export models",
    default="scripting",
)
parser.add_argument(
    "--output_path", help="output directory for the converted model", default='./output/')
parser.add_argument('--model_path', help='Path to model weight',
                    default='./output/model_final.pth')
parser.add_argument('--sample_image', type=str,
                    default='./sample/978.jpg')
parser.add_argument('--threshold', type=float,
                    default=0.7)
args = parser.parse_args()
modelpath = args.model_path
exportmethod = args.exportmethod
format = args.format
output = args.output_path
sample_image = args.sample_image
threshold = args.threshold


def setup():
    # Load predictor
    cfg = get_cfg()
    add_d2test_config(cfg)
    cfg.MODEL.WEIGHTS = modelpath
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.DEVICE = "cpu"
    return cfg


def export_scripting(torch_model, inputs):
    assert TORCH_VERSION >= (1, 7)
    image = inputs[0]["image"]
    inputs = [{"image": image}]  # remove other unused keys

    if isinstance(torch_model, GeneralizedRCNN):

        def inference(model, inputs):
            # use do_postprocess=False so it returns ROI mask
            inst = model.inference(inputs, do_postprocess=False)[0]
            return [{"instances": inst}]

    else:
        inference = None  # assume that we just call the model directly

    traceable_model = TracingAdapter(torch_model, inputs, inference)

    ts_model = torch.jit.trace(traceable_model, (image,))
    with PathManager.open(os.path.join(output, "model_tracing.ts"), "wb") as f:
        torch.jit.save(ts_model, f)
    dump_torchscript_IR(ts_model, output)


def get_sample_inputs():

    if sample_image is None:
        # get a first batch from dataset
        data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
        first_batch = next(iter(data_loader))
        return first_batch
    else:
        # get a sample data
        original_image = detection_utils.read_image(
            sample_image, format=cfg.INPUT.FORMAT)
        # Do same preprocessing as DefaultPredictor
        aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        height, width = original_image.shape[:2]
        image = aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        inputs = {"image": image, "height": height, "width": width}

        # Sample ready
        sample_inputs = [inputs]
        return sample_inputs


if __name__ == '__main__':
    cfg = setup()
    torch_model = build_model(cfg)
    DetectionCheckpointer(torch_model).resume_or_load(cfg.MODEL.WEIGHTS)
    torch_model.eval()
    sample_inputs = get_sample_inputs()
    export_scripting(torch_model, sample_inputs)
    print('Finished export!')
