# Predict mask detection with detectron2

The FSOFT organizing committee provides a dataset of 790 pictures labeled with bounding boxes in 3 classes: wearing masks, not wearing masks and wearing masks incorrectly.

## Create virtual environment
```bash
conda create -n d2detector python=3.8
conda activate d2detector
```

## Install requirements
### Torch CUDA and Detectron2
```bash
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
```
Include Detectron2: Follow CUDA and Torch version to run this project.

Link detail: https://detectron2.readthedocs.io/en/latest/tutorials/install.html#install-pre-built-detectron2-linux-only

### Related packages
```bash
pip install -r requirements.txt
```

## Dataset
- mask-fpt-ai.zip: Include 1 folder images (790 images) and 1 annotations coco format
- annotations.json: Annotations of above dataset
- train.json: Annotations for training (100%)
- test.json: Annotations for testing (20%)

## Get data (Sample)
```bash
bash setup_data.sh
```
Include: 1 folder images (790 images) and 1 annotations coco format

## Deal video dataset
I have already define some functions to convert dataset video to images.
```
python d2detector/video2frame.py --data_paths /path/to/dataset/video --output dataset/images
```
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
    --weight 'output/model_final.pth' \
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

## Output
![evaluate](https://github.com/vnk8071/detectron2-object-detection/blob/master/image/output.jpg)

**Sample**
![sample](https://github.com/vnk8071/detectron2-object-detection/blob/master/image/sample.jpg)

## Export model to tracing
```bash
python d2detector/export_model_tracing.py \
    --sample_image 'sample/978.jpg' \
    --model_path 'output/model_final.pth' \
    --threshold 0.7 \
    --output_path 'output/' 
```
After have model tracing, you can follow my repository about inference Detectron2 by C++.

Link repo: https://github.com/vnk8071/AI-on-Cpp/tree/main/Detectron2-Cpp
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
