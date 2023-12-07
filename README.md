# InterCutMix

Interaction-aware Scene Debiasing Method for Action Recognition

# Steps

## Preparation

1. Clone this repository and the submodules

```shell
git clone --recursive https://github.com/rendicahya/intercutmix.git
```

2. Download the UCF101 dataset

```shell
mkdir data
wget -P data https://www.crcv.ucf.edu/datasets/human-actions/ucf101/UCF101.rar --no-check-certificate
unrar x UCF101.rar data
mv data/UCF-101 data/ucf101
```

3. Prepare virtual environments

```shell
mkdir venv
python -m venv venv/venv1
source venv/venv1/bin/activate
pip install -U pip
```

## Relevancy

This step generates relevancy scores between UCF101 action names and object names used by the UniDet object detector.

1. Install packages

```shell
pip install sentence-transformers pandas tqdm
```

2. Generate relevancy lists. This will create subdirectories `relevancy/unidet-relevant-ids` and `relevancy/unidet-relevant-names` and generate relevancy files in JSON format.

```shell
python relevancy.py
```

## Generate scene videos

This process uses manually-created annotations in XGTF format available in the `xgtf` directory.

1. Install packages

```shell
pip install beautifulsoup4 lxml
```

2. Convert XGTF annotation files into mask images. This will generate the images in `data/ucf101-xgtf-mask`.

```shell
python xgtf_to_mask.py
```

3. Enter submodule

```shell
cd E2FGVI
```

4. Generate videos

## Object detection

1. Enter submodule

```shell
cd UniDet
```

2. Install packages
```shell
pip install gdown
```

3. Download pretrained object detection model

```shell
mkdir models
gdown 1HvUv399Vie69dIOQX0gnjkCM0JUI9dqI -O models
```

3. Install packages

```shell
# pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
# pip install detectron2 https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
pip install -r requirements.txt
```