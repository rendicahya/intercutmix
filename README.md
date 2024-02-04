# InterCutMix

Interaction-aware Scene Debiasing Method for Action Recognition.

# Steps

## A. Preparation

1. Clone this repository and the submodules.

```shell
git clone --recursive https://github.com/rendicahya/intercutmix.git
cd intercutmix
```

2. Create virtual environment.

```shell
python -m venv ~/venv/intercutmix
source ~/venv/intercutmix/bin/activate
pip install -U pip
```

## B. Download datasets

Make sure that you are in the `intercutmix` directory.

### a. UCF101

1. Download videos.

```shell
mkdir -p data/ucf101
cd data/ucf101
wget https://www.crcv.ucf.edu/datasets/human-actions/ucf101/UCF101.rar --no-check-certificate
unrar x UCF101.rar -idq
rm UCF101.rar
mv UCF-101 videos

# Symlink:
# ln -s /nas.dbms/randy/datasets/ucf101/videos/avi data/ucf101/videos
```

2. Download annotations.

```shell
wget https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip --no-check-certificate
unzip UCF101TrainTestSplits-RecognitionTask.zip
rm UCF101TrainTestSplits-RecognitionTask.zip
mv ucfTrainTestlist annotations
cd ../..
```

### b. HMDB51

Download videos.

```shell
mkdir -p data/hmdb51/videos
cd data/hmdb51/videos
wget --no-check-certificate http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar
unrar x hmdb51_org.rar
rm hmdb51_org.rar
for file in *.rar; do unrar x "$file"; done
rm *.rar
cd ../../..

# Symlink:
# ln -s /nas.dbms/randy/datasets/hmdb51/videos /nas.dbms/randy/projects/intercutmix/data/hmdb51/videos
```

## C. Generate mask images

Install packages.

```shell
pip install beautifulsoup4 lxml tqdm scipy gdown opencv-python av decord moviepy
```

### a. UCF101

1. Download .xgtf files.

```shell
cd data/ucf101
wget http://crcv.ucf.edu/ICCV13-Action-Workshop/index.files/UCF101_24Action_Detection_Annotations.zip --no-check-certificate
mkdir xgtf
unzip -q UCF101_24Action_Detection_Annotations.zip
mv UCF101_24Action_Detection_Annotations/UCF101_24_Annotations xgtf/files
rmdir UCF101_24Action_Detection_Annotations
rm UCF101_24Action_Detection_Annotations.zip
```

2. Correct file name.

```shell
cd xgtf/files/RopeClimbing
mv v_RopeClimbing_g02_C01.xgtf v_RopeClimbing_g02_c01.xgtf
cd ../../../..
```

3. Convert .xgtf files into mask images.

The results will be stored in `data/ucf101/xgtf/mask`.

```shell
python xgtf2mask.py
```

### b. HMDB51

1. Download .mat files.

```shell
mkdir -p data/hmdb51/mat/files
cd data/hmdb51/mat/files
gdown 1qwarqC8O6XU5CKyMLub6qPpjw2pvVrfg
tar -xzf hmdb51-mask.tar.gz
rm hmdb51-mask.tar.gz
```

2. Convert .mat files into mask images.

```shell
python mat2mask.py
```

The results will be stored in `data/hmdb51/mat/mask`.

## E. Generate scene-only videos

This step generates scene-only videos using the [E<sup>2</sup>FGVI](https://github.com/MCG-NKU/E2FGVI) method ([Li et al., 2022](https://arxiv.org/abs/2204.02663)) and the mask images generated in step C.

1. Enter submodule.

```shell
cd E2FGVI
```

2. Install packages.

Use the correct PyTorch version according to your system.

```shell
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install openmim matplotlib
mim install mmcv-full
```

3. Download the E<sup>2</sup>FGVI checkpoint `E2FGVI-HQ-CVPR22.pth`.

```shell
gdown 10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3 -O release_model/
```

4. Generate videos.

This step will take several hours and the resulting videos will be stored in `data/{dataset}/xgtf/scene`.

```shell
python batch.py

# Symlink: ln -s /nas.dbms/randy/datasets/ucf101/xgtf/scene ../data/ucf101/xgtf/scene
```

5. Make a list of the generated scene videos.

The output will be stored in `data/{dataset}/xgtf/scene-list.json`.

```shell
cd ..
python list-scene.py
```

## F. Relevancy

This step generates relevancy scores between UCF101 action names and object names covered in the UniDet object detector.

1. Install packages.

```shell
pip install sentence-transformers
```

2. Generate relevancy lists. This will generate relevancy files in JSON format saved in `data/relevancy/UniDet/ids` and `data/relevancy/UniDet/names`.

```shell
python relevancy.py
```

## G. Object detection

This step uses [Unified Detector (UniDet)](https://github.com/xingyizhou/UniDet) ([Zhou et al., 2022](http://arxiv.org/abs/2102.13086)).

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

This step detects all objects with a confidence threshold of 0.5 (configurable in `unidet.detect.confidence`). The detection results will be saved in JSON files in `data/{dataset}/UniDet/detect/json`.

```shell
python batch-detect.py

# Symlink:
# mkdir -p data/ucf101/UniDet/detect
# ln -s /nas.dbms/randy/datasets/ucf101/UniDet/detect/json data/ucf101/UniDet/detect/json
```

Output videos can be generated by setting `unidet.detect.output.video.generate` to `true`. Then the generated videos will be saved in `data/{dataset}/UniDet/detect/videos`.

5. Filter object detection.

This will select only relevant objects based on the relevancy between the video class names and the detected object names (step F). The output are `.pckl` files stored in `data/{dataset}/UniDet/select/inter/dump`.

```shell
python batch-select.py

# Symlink:
# mkdir data/ucf101/UniDet/select/actor data/ucf101/UniDet/select/inter
# ln -s /nas.dbms/randy/datasets/ucf101/UniDet/select/actor/dump data/ucf101/UniDet/select/actor/dump
# ln -s /nas.dbms/randy/datasets/ucf101/UniDet/select/inter/dump data/ucf101/UniDet/select/inter/dump
```

There are two modes: `actorcutmix` and `intercutmix` (default) configurable in `unidet.select.mode`. If you change it, make sure to change `unidet.select.output.video.path`, `unidet.select.output.mask.path`, and `unidet.select.output.dump.path` accordingly. Videos and mask images can also be generated by setting `conf.unidet.select.output.video.generate` and `conf.unidet.select.output.mask.generate` to true.

6. Quit submodule.

```shell
cd ..
```

## H. Detection Post-processing

This step uses [Robust and efficient post-processing for video object detection (REPP)](https://github.com/AlbertoSabater/Robust-and-efficient-post-processing-for-video-object-detection) ([Sabater et al., 2020](https://arxiv.org/abs/2009.11050)) to refine the object detection results.

1. Enter submodule.

```shell
cd REPP
```

2. Run REPP.

This will post-process the `.pckl` files and save the resulting mask files in `data/{dataset}/UniDet/REPP/inter/mask` and (optionally) the resulting videos in `data/{dataset}/UniDet/REPP/inter/videos`.

```shell
python batch.py

# Symlink:
# mkdir -p ../data/ucf101/UniDet/REPP/inter
# ln -s /nas.dbms/randy/datasets/ucf101/REPP/inter/mask ../data/ucf101/UniDet/REPP/inter/mask
```

## I. CutMix

Videos will be mixed with scene-only videos. By default, 10 scene-only videos will be randomly picked from different actions and each input video will be mixed with them. Thus, the resulting mixed videos will be 10 times as many as the original videos.

1. Run script.

```shell
python cutmix.py
```

## J. Training

1. Install packages.

```shell
pip install 
```
