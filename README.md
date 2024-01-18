# InterCutMix

Interaction-aware Scene Debiasing Method for Action Recognition.

# Steps

## A. Preparation

1. Clone this repository and the submodules.

```shell
git clone --recursive https://github.com/rendicahya/intercutmix.git
cd intercutmix
```

2. Create a virtual environment.

```shell
python -m venv venv
source venv/bin/activate
pip install -U pip
```

## B. Download datasets

### a. UCF101

```shell
mkdir -p data/ucf101
cd data/ucf101
```

```shell
wget https://www.crcv.ucf.edu/datasets/human-actions/ucf101/UCF101.rar --no-check-certificate
wget https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip --no-check-certificate
```

```shell
unrar x UCF101.rar -idq
rm UCF101.rar
mv UCF-101 videos
unzip UCF101TrainTestSplits-RecognitionTask.zip
rm UCF101TrainTestSplits-RecognitionTask.zip
mv ucfTrainTestlist annotations
cd ../..
```

### b. HMDB51

```shell
mkdir -p data/hmdb51/videos
cd data/hmdb51/videos
```

```shell
wget http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar
# wget http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_sta.rar
```

```shell
unrar x hmdb51_org.rar
rm hmdb51_org.rar
for file in *.rar; do unrar x "$file"; done
rm *.rar
cd ../../..
```

## D. Generate scene mask images

This process uses bounding boxes in xgtf format to generate mask images that will be used in the next step to create scene videos.

### a. UCF101

1. Install packages.

```shell
pip install beautifulsoup4 lxml opencv-python tqdm av decord moviepy
```

2. Download bounding boxes.
```shell
cd data/ucf101
```

```shell
wget http://crcv.ucf.edu/ICCV13-Action-Workshop/index.files/UCF101_24Action_Detection_Annotations.zip --no-check-certificate
```

```shell
unzip -q UCF101_24Action_Detection_Annotations.zip
mv UCF101_24Action_Detection_Annotations/UCF101_24_Annotations xgtf
rmdir UCF101_24Action_Detection_Annotations
rm UCF101_24Action_Detection_Annotations.zip readme.txt
cd ../..
```

2. Generate mask images. This will take a few minutes and the results will be stored in `data/ucf101/xgtf-mask`.

```shell
python xgtf_to_mask.py
```

### b. HMDB51

1. Install packages.
```shell
pip install scipy
```

## E. Generate scene videos

This step generates scene-only videos using the [E<sup>2</sup>FGVI](https://github.com/MCG-NKU/E2FGVI) method ([Li et al., 2022](https://arxiv.org/abs/2204.02663)) and the mask images generated in [step D](#d-generate-scene-mask-images). Therefore, make sure that step D has been successfully completed.

1. Enter submodule.

```shell
cd E2FGVI
```

2. Install packages. Use the correct PyTorch version according to your system.

```shell
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install openmim gdown matplotlib av decord moviepy
mim install mmcv-full
```

3. Download the E<sup>2</sup>FGVI checkpoint `E2FGVI-HQ-CVPR22.pth`.

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

## F. Relevancy

This step generates relevancy scores between UCF101 action names and object names covered in the UniDet object detector.

1. Install packages.

```shell
pip install sentence-transformers
```

2. Generate relevancy lists. This will generate relevancy files in JSON format saved in `relevancy/unidet-relevant-ids` and `relevancy/unidet-relevant-names`.

```shell
python relevancy.py
```

## G. Object detection

This step uses the [UniDet](https://github.com/xingyizhou/UniDet) method ([Zhou et al., 2022](http://arxiv.org/abs/2102.13086)).

1. Enter submodule.

```shell
cd UniDet
```

2. Install packages.

```shell
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
pip install pillow==9.5.0 numpy==1.23.5
```

3. Download object detection checkpoint `Unified_learned_OCIM_RS200_6x+2x.pth`.

```shell
gdown 1HvUv399Vie69dIOQX0gnjkCM0JUI9dqI -O models/
```

4. Run object detection.

This step detects all objects with a confidence threshold of (by default) 0.5. The detection results for each video will be saved in a JSON file in `data/ucf101/unidet-json` and the generated videos will be saved in `data/ucf101/unidet`. If you want to speed up the process by generating only the JSON files without generating videos, modify `config.py` and set `unidet.detect.output.video.generate` to `false`.

```shell
python batch-detect.py
```

5. Filter object detection.

This will select only relevant objects based on the relevancy between the video class names and the detected object names. The output of this step is mask images stored in `data/ucf101/unidet-actor-mask` and the generated videos will be saved in `data/ucf101/unidet-actor`. Therefore, make sure that `batch-detect.py` and step D (relevancy) have been successfully completed. If you want to speed up the process by generating only the JSON files without generating videos, modify `config.py` and set `unidet.select.output.video.generate` to `false`.

```shell
python batch-select.py
```

There are two modes of this script: `actorcutmix` (default) and `intercutmix`. This can be configured in `config.json` at `unidet.select.mode`. If you change the mode, make sure to change `unidet.select.output.video.path` and `unidet.select.output.mask.path` as well.
