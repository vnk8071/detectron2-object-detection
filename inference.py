import cv2
from PIL import Image
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode
from detectron2.engine import DefaultPredictor
from d2detector.config import add_d2test_config
from detectron2 import model_zoo
import os
import numpy as np
import time
from tqdm import tqdm
import copy
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import argparse


CLASSES = ['no_mask', 'mask', 'incorrect_mask']
DEVICE = 'gpu'

parser = argparse.ArgumentParser(description='Predict bounding box for video')
parser.add_argument('-i', dest='input_path', type=str,
                    default="./sample/", help='Path of video to predict')
parser.add_argument('-m', dest='model_zoo', type=str,
                    default="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", help='Path of model to predict')
parser.add_argument('-w', dest='weight', type=str,
                    default="./output/model_path.pth", help='Path weight trained of model')
parser.add_argument('--threshold', dest='threshold', type=float, default=0.7,
                    help='Threshold to bounding box display')
parser.add_argument('--fps', dest='fps_predict', type=int, default=0,
                    help='Number of frame per second (default is 0) to predict video')
parser.add_argument('-o', dest='output_path', type=str,
                    default="./inference/", help='Path of video to save')

args = parser.parse_args()
input_path = args.input_path
output_path = args.output_path
fps_predict = args.fps_predict
version = args.version
model = args.model_zoo
weight = args.weight
threshold = args.threshold


def setup():
    # Load predictor
    cfg = get_cfg()
    add_d2test_config(cfg)
    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.MODEL.WEIGHTS = weight
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASSES)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.DEVICE = DEVICE
    predictor = DefaultPredictor(cfg)

    # get meta data
    metadata = MetadataCatalog.get("test")
    metadata.set(thing_classes=CLASSES)

    return predictor, metadata


def visualize_(outputs, image, metadata):
    v = Visualizer(image[:, :, ::-1],
                   metadata=metadata,
                   scale=1,
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
                   )
    v = v.draw_instance_predictions(outputs["instances"].to('cpu'))
    image_out = v.get_image()
    image_out = Image.fromarray(image_out)
    return image_out


def display_predict(args):
    # Create config
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        args.model_zoo))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.threshold
    cfg.MODEL.WEIGHTS = args.weight
    cfg.MODEL.DEVICE = "gpu"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASSES)
    metadata = MetadataCatalog.get("test")
    metadata.set(thing_classes=CLASSES)

    # get image
    im = mpimg.imread(args.input_path)
    # Create predictor
    predictor = DefaultPredictor(cfg)

    # Make prediction
    start = time.perf_counter()
    outputs = predictor(im)
    print(outputs)
    end = time.perf_counter()
    print('Total time: {} (s)'.format(end-start))
    v = Visualizer(im[:, :, ::-1], metadata=metadata,
                   scale=1)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    print("V:", v)
    plt.imshow(v.get_image()[:, :, ::-1])
    plt.show()


def segment4video(input_path, output_path, predictor, metadata):
    print("\n--------- VideoProcessing ---------")
    cap = cv2.VideoCapture(input_path)

    if cap.isOpened():
        width = cap.get(3)
        height = cap.get(4)
        fps = cap.get(5)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # TODO: try 'MP4V'
        out = cv2.VideoWriter(output_path, fourcc, fps,
                              (int(width), int(height)))

        print("Running...", input_path)
        print("Video:", output_path)
        print(f"Metadata | W: {width} | H: {height} | fps: {round(fps)}")
        print("----------------------------------")
        for i in tqdm(range(num_frames)):
            ret, frame = cap.read()
            if ret:
                outputs = predictor(frame)
                image_out = visualize_(outputs, frame, metadata)
                image_out = cv2.cvtColor(
                    np.array(image_out), cv2.COLOR_BGR2RGB)
                out.write(np.array(image_out))
            else:
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    else:
        print("Cannot open ", input_path)


def segment4video_fps(input_path, output_path, predictor, metadata, fps_predict):
    print("\n--------- VideoProcessing ---------")
    cap = cv2.VideoCapture(input_path)

    if cap.isOpened():
        width = cap.get(3)
        height = cap.get(4)
        fps = cap.get(5)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # TODO: try 'MP4V'
        out = cv2.VideoWriter(output_path, fourcc, fps,
                              (int(width), int(height)))

        print("Running...", input_path)
        print("Video:", output_path)
        print(
            f"Metadata | W: {width} | H: {height} | fps: {round(fps)} | fps: {fps_predict}")
        print("----------------------------------")
        for index in tqdm(range(num_frames)):
            ret, frame = cap.read()
            if ret and index % (round(fps/fps_predict)) == 0:
                outputs = predictor(frame)
                image_out = visualize_(outputs, frame, metadata)
                image_out = cv2.cvtColor(
                    np.array(image_out), cv2.COLOR_BGR2RGB)
            elif ret and index % (round(fps/fps_predict)) != 0:
                image_out = visualize_(outputs_prev, frame, metadata)
                image_out = cv2.cvtColor(
                    np.array(image_out), cv2.COLOR_BGR2RGB)
            else:
                break
            out.write(np.array(image_out))
            outputs_prev = copy.deepcopy(outputs)
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    else:
        print("Cannot open ", input_path)


if __name__ == '__main__':
    predictor, metadata = setup()
    start = time.perf_counter()
    start_predict = time.monotonic()
    if fps_predict == 0:
        display_predict(args)
    elif fps_predict == 1:
        segment4video(
            input_path, output_path, predictor, metadata)
    elif fps_predict > 1:
        segment4video_fps(input_path, output_path,
                          predictor, metadata, fps_predict)
    else:
        print('Please check the number of frame per image (fps)')
    end_predict = time.monotonic()
    elapsed_time = int(end_predict - start_predict)

    print(f'Elapsed time: {round(elapsed_time/60,2)} mins')
