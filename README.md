# detectron2-object-detection

## Install requirements
```bash
pip install -r requirements.txt
```

## Split train test
```bash
python d2detector/cocosplit.py --having-annotations -s 0.9 dataset/annotations.json dataset/train.json dataset/test.json
```

## Train
```bash
python train.py \
    --datapath_train 'data/train.json' \
    --datapath_test 'data/test.json' \
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

With fps:
- 0 is image
- 1 is video
- >1 is predict frame per second on video

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