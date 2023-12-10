# InterCutMix

Interaction-aware Scene Debiasing Method for Action Recognition.

# Steps

## A. Preparation

1. Clone this repository and the submodules.

```shell
git clone --recursive https://github.com/rendicahya/intercutmix.git
cd intercutmix
```

2. Download the UCF101 dataset.

```shell
mkdir -p data/ucf101
cd data/ucf101
wget https://www.crcv.ucf.edu/datasets/human-actions/ucf101/UCF101.rar --no-check-certificate
wget https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip --no-check-certificate
unrar x UCF101.rar
rm UCF101.rar
mv UCF-101 videos
unzip UCF101TrainTestSplits-RecognitionTask.zip
rm UCF101TrainTestSplits-RecognitionTask.zip
mv ucfTrainTestlist annotations
cd ../..
```

3. Prepare virtual environments.

```shell
python -m venv venv
source venv/bin/activate
pip install -U pip
```

## B. Generate scene mask images

This process uses manually-created annotations in xgtf format available in the `xgtf` directory to generate mask images. These images will be used in the next step to create scene videos.

1. Install packages.

```shell
pip install beautifulsoup4 lxml opencv-python tqdm av decord moviepy
```

2. Generate mask images. The results will be stored in `data/ucf101/xgtf-mask`.

```shell
python xgtf_to_mask.py
```

## C. Generate scene videos

1. Enter submodule.

```shell
cd E2FGVI
```

2. Install packages.

```shell
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install openmim gdown matplotlib av decord moviepy
mim install mmcv-full
```

3. Download pretrained model.

```shell
gdown 10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3 -O release_model/
```

4. Generate videos. This step will take several hours and the resulting videos will be stored in `data/ucf101/scene-xgtf`.

```shell
python batch.py
```

5. Make a list of the generated scene videos. The list will be stored in `data/ucf101/scene-xgtf.json`.

```shell
cd ..
python make_file_list_json.py
```

## D. Relevancy

This step generates relevancy scores between UCF101 action names and object names used by the UniDet object detector.

1. Install packages.

```shell
pip install sentence-transformers
```

2. Generate relevancy lists. This will create subdirectories `relevancy/unidet-relevant-ids` and `relevancy/unidet-relevant-names` and generate relevancy files in JSON format.

```shell
python relevancy.py
```

## E. Object detection

1. Enter submodule.

```shell
cd UniDet
```

2. Install packages.

```shell
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
pip install pillow==9.5.0 numpy==1.23.5
```

3. Download pretrained object detection model.

```shell
mkdir models
gdown 1HvUv399Vie69dIOQX0gnjkCM0JUI9dqI -O models/
```

3. Run object detection. This step will detect all objects with a confidence threshold of (by default) 0.5. The detection results will be saved in a JSON file for each video in `data/ucf101/unidet-json`.

```shell
python batch-detect.py
```

4. Filter object detection. This will select only relevant objects based on the relevancy between the video class names and the detected object names.

```shell
python batch-select.py
```
