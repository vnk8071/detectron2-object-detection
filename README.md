# Predict mask detection with detectron2

The FSOFT organizing committee provides a dataset of 790 pictures labeled with bounding boxes in 3 classes: wearing masks, not wearing masks and wearing masks incorrectly.

## Create virtual environment
```bash
conda create -n d2detector python=3.8
conda activate d2detector
```

## Install requirements
```bash
pip install -r requirements.txt
```
Include Detectron2: Follow CUDA and Torch version to run this project.

Link detail: https://detectron2.readthedocs.io/en/latest/tutorials/install.html#install-pre-built-detectron2-linux-only

## Get data
```bash
bash setup_data.sh
```
Include: 1 folder images (790 images) and 1 annotations coco format

## Split train test
```bash
python d2detector/cocosplit.py --having-annotations -s 0.9 dataset/annotations.json dataset/train.json dataset/test.json
```

## Train
```bash
python train.py \
    --datapath_train 'dataset/train.json' \
    --datapath_test 'dataset/test.json' \
    --imagepath 'dataset/images'

# or simply
python train.py
```

## Inference
```bash
python inference.py \
    --input_path 'sample/978.jpg' \
    --model_zoo 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml' \
    --weight 'output/model_path.pth' \
    --threshold 0.7 \
    --fps 0 \
    --output_path 'inference/' 

# or simply
python inference.py
```

*With fps*:
- 0 is image
- 1 is video
- greater than 1 is predict frame per second on video

**Check image or video in inference folder**

## License
```bash
@misc{wu2019detectron2,
  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                  Wan-Yen Lo and Ross Girshick},
  title =        {Detectron2},
  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
  year =         {2019}
}
```
