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
python3 -m venv ~/venv/intercutmix
source ~/venv/intercutmix/bin/activate
pip install -U pip
```

## B. Download datasets

### a. UCF101

1. Download videos.

```shell
mkdir -p data/ucf101 && cd "$_"
wget https://www.crcv.ucf.edu/datasets/human-actions/ucf101/UCF101.rar --no-check-certificate
unrar x UCF101.rar -idq
rm UCF101.rar
mv UCF-101 videos
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
mkdir -p data/hmdb51/videos && cd "$_"
wget --no-check-certificate http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar
unrar x hmdb51_org.rar
rm hmdb51_org.rar
for file in *.rar; do unrar x "$file"; done
rm *.rar
cd ../../..
```

## C. Generate mask images

Install packages.

```shell
pip install beautifulsoup4 lxml tqdm opencv-python av decord moviepy scipy gdown
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
cd ../../../../..
```

3. Convert .xgtf files into mask images.

The results will be stored in `data/ucf101/xgtf/mask`.

```shell
python3 xgtf2mask.py
```

### b. HMDB51

1. Download .mat files.

```shell
mkdir -p data/hmdb51/mat/files && cd "$_"
gdown 1qwarqC8O6XU5CKyMLub6qPpjw2pvVrfg
tar -xzf hmdb51-mask.tar.gz
rm hmdb51-mask.tar.gz
cd ../../../..
```

2. Convert .mat files into mask images.

```shell
python3 mat2mask.py
```

The results will be stored in `data/hmdb51/mat/mask`.

## D. Generate scene-only videos

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

3. Download model checkpoint `E2FGVI-HQ-CVPR22.pth`.

```shell
gdown 10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3 -O release_model/
```

4. Generate videos.

This step will take several hours and the resulting videos will be stored in `data/{dataset}/xgtf/scene`.

```shell
python3 batch.py
cd ..
```

Alternatively, download the result:

```shell
mkdir -p ../data/ucf101/xgtf/scene && cd "$_"
gdown 1F53RbTXaWW-M6W5I7JDhRU73szm2GvWi
tar -xzf ucf101-scene.tar.gz
rm ucf101-scene.tar.gz
cd ../../../../
```

5. Make a list of the generated scene videos.

The output will be stored in `data/{dataset}/xgtf/scene-list.json`.

```shell
cd ..
python3 list-scene.py
```

## E. Relevancy

This step generates relevancy scores between dataset action names and object names covered in the UniDet object detector (step F).

1. Create virtual environment.

```shell
deactivate
python3 -m venv ~/venv/sentence-transformers
source ~/venv/sentence-transformers/bin/activate
pip install -U pip
```

2. Install packages.

```shell
pip install sentence-transformers pandas pyarrow
```

3. Generate relevancy lists. This will generate relevancy files in JSON format in `data/relevancy/UniDet/ids` and `data/relevancy/UniDet/names`.

```shell
python3 relevancy.py
```

## F. Object detection

This step uses [Unified Detector (UniDet)](https://github.com/xingyizhou/UniDet) ([Zhou et al., 2022](http://arxiv.org/abs/2102.13086)).

1. Enter submodule.

```shell
cd UniDet
```

2. Switch environment.

```shell
deactivate
source ~/venv/intercutmix/bin/activate
```

3. Install packages.

```shell
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
pip install pillow==9.5.0 numpy==1.23.5
```

4. Download model checkpoint `Unified_learned_OCIM_RS200_6x+2x.pth`.

```shell
gdown 1HvUv399Vie69dIOQX0gnjkCM0JUI9dqI -O models/
```

5. Run object detection.

This step detects all objects (regardless of the actorcutmix/intercutmix mode) with a confidence threshold of 0.5 (configurable in `unidet.detect.confidence`). The detection results will be saved in JSON files in `data/{dataset}/UniDet/detect/json`.

```shell
python3 batch-detect.py
```

Output videos can be generated by setting `unidet.detect.output.video.generate` to `true`. The generated videos will be saved in `data/{dataset}/UniDet/detect/videos`.

Alternatively, download the result:
```shell
mkdir -p ../data/ucf101/UniDet/detect/json && cd "$_"
gdown 1DPIVkNk36wn2fScwH02QHA3z-s9fc8hR
tar -xzf ucf101-UniDet-json.tar
rm ucf101-UniDet-json.tar
cd ../../../../..
```

6. Filter object detection.

This will select only relevant objects based on the relevancy between the video class names and the detected object names (step F). The output are `.pckl` files stored in `data/{dataset}/UniDet/select/inter/dump`.

```shell
python3 batch-select.py
cd ..
```

There are two modes: `actorcutmix` and `intercutmix` (default) configurable in `unidet.select.mode`. If you change it, make sure to change `unidet.select.output.video.path`, `unidet.select.output.mask.path`, and `unidet.select.output.dump.path` accordingly. Videos and mask images can also be generated by setting `conf.unidet.select.output.video.generate` and `conf.unidet.select.output.mask.generate` to true.

Alternatively, download the results:
```shell
mkdir -p data/ucf101/UniDet/select/actor/dump && cd "$_"
gdown 1BW6AXE6glxkrdhucyTJAD9MYVTG2ZhU7
tar -xzf ucf101-UniDet-dump-actor.tar
rm ucf101-UniDet-dump-actor.tar
mkdir -p ../../inter/dump && cd "$_"
gdown 1rAhSitJpva-e3nkxOZAilkk4T6XHJhGS
tar -xzf ucf101-UniDet-dump-inter.tar
rm ucf101-UniDet-dump-inter.tar
cd ../../../../../..
```

## G. Detection Post-processing

This step uses [Robust and efficient post-processing for video object detection (REPP)](https://github.com/AlbertoSabater/Robust-and-efficient-post-processing-for-video-object-detection) ([Sabater et al., 2020](https://arxiv.org/abs/2009.11050)) to refine the object detection results.

1. Enter submodule.

```shell
cd REPP
```

2. Install packages.

```shell
pip install scikit-learn
```

3. Run REPP.

This will post-process the `.pckl` files and save the resulting mask files in `data/{dataset}/UniDet/REPP/{mode}/mask` and (optionally) the resulting videos in `data/{dataset}/UniDet/REPP/{mode}/videos`. Available options for `{mode}` are` `actor` and `inter` (default) configurable in `conf.repp.input.dump.path`: `"data/ucf101/UniDet/select/**<inter/actor>**/dump"`.

```shell
python3 batch.py
```

Alternatively, download the result:
```shell
# Actor:
mkdir -p ../data/ucf101/REPP/actor/mask && cd "$_"
gdown 1Qhb64pXJPy8nUUfWSK0SHqTCSJLwsXcD
tar -xzf ucf101-REPP-mask-actor.tar && rm "$_"

# Inter:
mkdir -p ../../inter/mask && cd "$_"
gdown 1oz6sos92T5fMxja1kgkvKTgyBXq8WO1P
tar -xzf ucf101-REPP-mask-inter.tar && rm "$_"
```

## H. CutMix

Videos will be mixed with scene-only videos. By default, 10 scene-only videos will be randomly picked from different actions and each input video will be mixed with them. Thus, the resulting mixed videos will be 10 times as many as the original videos.

1. Run script.

```shell
python3 cutmix.py
```

## J. Classification

1. Install packages.

```shell
mim install mmengine mmcv
```

2. Install mmaction2.

```shell
pip install -v -e mmaction2
```

3. Generate file list.

```shell
python3 mmaction2/tools/data/build_file_list.py ucf101 data/ucf101/videos --level 2 --format videos --shuffle
```

4. Train.

```shell
python3 mmaction2/tools/train.py mmaction2/configs/<config>.py --work-dir mmaction2/work_dirs/<dir> --amp --auto-scale-lr
```

5. Test.

```shell
python3 mmaction2/tools/test.py mmaction2/configs/<config>.py work_dirs/<checkpoint>.pth 
```

# Citations

If you find our code useful for your research, please consider citing our paper:
 ```bibtex
 @inproceedings{randy2024intercutmix,
    title={InterCutMix: Interaction-aware Scene Debiasing Method for Action Recognition},
    author={Wihandika, Randy Cahya and Mendonça, Israel and Aritsugi, Masayoshi},
    booktitle={European Conference on Computer Vision (ECCV)},
    year={2024}
 }
 ```
