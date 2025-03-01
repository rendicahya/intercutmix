# InterCutMix

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
unrar x UCF101.rar && rm "$_"
mv UCF-101 videos
```

2. Download annotations.

```shell
wget https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip --no-check-certificate
unzip -q UCF101TrainTestSplits-RecognitionTask.zip && rm "$_"
mv ucfTrainTestlist annotations
cd ../../
```

3. Check structure.

```shell
intercutmix/data/ucf101/
├── annotations/
│   ├── classInd.txt
│   ├── testlist01.txt
│   ├── testlist02.txt
│   ├── testlist03.txt
│   ├── trainlist01.txt
│   ├── trainlist02.txt
│   └── trainlist03.txt
└── videos/
    ├── ApplyEyeMakeup/
    │   ├── v_ApplyEyeMakeup_g01_c01.avi
    │   ├── v_ApplyEyeMakeup_g01_c02.avi
    │   ├── v_ApplyEyeMakeup_g01_c03.avi
    |   └── ...
    ├── ApplyLipstick/
    │   ├── v_ApplyLipstick_g01_c01.avi
    │   ├── v_ApplyLipstick_g01_c02.avi
    │   ├── v_ApplyLipstick_g01_c03.avi
    |   └── ...
    └── ...
```

### b. HMDB51

1. Download videos.

```shell
mkdir -p data/hmdb51/videos && cd "$_"
wget --no-check-certificate http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar
unrar x hmdb51_org.rar && rm "$_"
for file in *.rar; do unrar x "$file"; done
rm *.rar
cd ../
```

2. Download annotations.

```shell
wget http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar
unrar x test_train_splits.rar && rm "$_"
cd ../../
```

3. Check structure.

```shell
intercutmix/data/hmdb51/
├── testTrainMulti_7030_splits/
│   ├── brush_hair_test_split1.txt
│   ├── brush_hair_test_split2.txt
│   ├── brush_hair_test_split3.txt
|   └── ...
└── videos/
    ├── brush_hair/
    │   ├── April_09_brush_hair_u_nm_np1_ba_goo_0.avi
    │   ├── April_09_brush_hair_u_nm_np1_ba_goo_1.avi
    │   ├── April_09_brush_hair_u_nm_np1_ba_goo_2.avi
    |   └── ...
    ├── cartwheel/
    │   ├── (Rad)Schlag_die_Bank!_cartwheel_f_cm_np1_le_med_0.avi
    │   ├── Acrobacias_de_un_fenomeno_cartwheel_f_cm_np1_ba_bad_8.avi
    │   ├── Acrobacias_de_un_fenomeno_cartwheel_f_cm_np1_fr_bad_3.avi
    |   └── ...
    └── ...
```

c. Kinetics-100

1. Download Kinetics-400 using the provided `k400_downloader.sh` and `k400_extractor.sh` in https://github.com/cvdfoundation/kinetics-dataset.git.

2. Make Kinetics-400 file list.

```shell
python3 k400-list.py
```

3. Build Kinetics-100 by creating symlinks from Kinetics-400.

```shell
python3 k100-make.py
```

## C. Generate mask images

Install packages.

```shell
pip install beautifulsoup4 lxml tqdm opencv-python av decord moviepy scipy gdown dynaconf click
```

### a. UCF101

1. Download .xgtf files.

```shell
cd data/ucf101
wget http://crcv.ucf.edu/ICCV13-Action-Workshop/index.files/UCF101_24Action_Detection_Annotations.zip --no-check-certificate
mkdir xgtf
unzip -q UCF101_24Action_Detection_Annotations.zip && rm "$_"
mv UCF101_24Action_Detection_Annotations/UCF101_24_Annotations xgtf/files
rmdir UCF101_24Action_Detection_Annotations
```

2. Correct file name.

```shell
cd xgtf/files/RopeClimbing
mv v_RopeClimbing_g02_C01.xgtf v_RopeClimbing_g02_c01.xgtf
cd ../../../../..
```

3. Generate mask images from .xgtf files.

```shell
python3 xgtf2mask.py
```

The results will be stored in `data/ucf101/xgtf/mask`.

4. Check structure.

```shell
intercutmix/data/hmdb51/
├── annotations/...
├── videos/...
└── xgtf/
    ├── files/
    │   ├── Basketball/
    |   |   ├── v_Basketball_g01_c01.xgtf
    |   |   ├── v_Basketball_g01_c02.xgtf
    |   |   └── ...
    │   ├── BasketballDunk/
    |   |   ├── v_BasketballDunk_g01_c01.xgtf
    |   |   ├── v_BasketballDunk_g01_c02.xgtf
    |   |   └── ...
    |   └── ...
    ├── mask/
    │   ├── Basketball/
    |   |   ├── v_Basketball_g01_c01/
    |   |   |   ├── 0000.png
    |   |   |   ├── 0001.png
    |   |   |   └── ...
    |   |   ├── v_Basketball_g01_c02/
    |   |   |   ├── 0000.png
    |   |   |   ├── 0001.png
    |   |   |   └── ...
    |   |   └── ...
    │   ├── BasketballDunk/
    |   |   ├── v_BasketballDunk_g01_c01/
    |   |   |   ├── 0000.png
    |   |   |   ├── 0001.png
    |   |   |   └── ...
    |   |   ├── v_Basketball_g01_c02/
    |   |   |   ├── 0000.png
    |   |   |   ├── 0001.png
    |   |   |   └── ...
    |   |   └── ...
    |   └── ...
    └── ...
```

### b. HMDB51

1. Download .mat files.

```shell
mkdir -p data/hmdb51/mat/files && cd "$_"
gdown 1qwarqC8O6XU5CKyMLub6qPpjw2pvVrfg
tar -xzf hmdb51-mask.tar.gz && rm "$_"
```

2. Fix file locations and names.

```shell
rm climb_stairs/Stadium_Plyometric_Workout_climb_stairs_f_cm_np1_ba_bad_5.mat
rm catch/LearnToShootFromTheMaster_catch_*
rm throw/Faith_Rewarded_throw_f_nm_np1_le_bad_51.mat
mv catch/Goalkeeper_Training_Day_@_7_catch_f_cm_np1_ri_med_0.mat catch/Goalkeeper_Training_Day_#_7_catch_f_cm_np1_ri_med_0.mat
mv clap/@20_Rhythm_clap_u_nm_np1_fr_goo_0.mat clap/#20_Rhythm_clap_u_nm_np1_fr_goo_0.mat
mv clap/@20_Rhythm_clap_u_nm_np1_fr_goo_1.mat clap/#20_Rhythm_clap_u_nm_np1_fr_goo_1.mat
mv clap/@20_Rhythm_clap_u_nm_np1_le_goo_3.mat clap/#20_Rhythm_clap_u_nm_np1_le_goo_3.mat
mv golf/Golf_Swing_@6Iron_golf_f_cm_np1_fr_med_0.mat golf/Golf_Swing_#6Iron_golf_f_cm_np1_fr_med_0.mat
mv golf/Golf_Swing_@6Iron_golf_f_cm_np1_fr_med_1.mat golf/Golf_Swing_#6Iron_golf_f_cm_np1_fr_med_1.mat
mv golf/Golf_Swing_@6Iron_golf_f_cm_np1_fr_med_2.mat golf/Golf_Swing_#6Iron_golf_f_cm_np1_fr_med_2.mat
mv golf/Golf_Swing_@6Iron_golf_f_cm_np1_ri_med_3.mat golf/Golf_Swing_#6Iron_golf_f_cm_np1_ri_med_3.mat
mv golf/Golf_Swing_@6Iron_golf_f_cm_np1_ri_med_4.mat golf/Golf_Swing_#6Iron_golf_f_cm_np1_ri_med_4.mat
mv golf/Golf_Swing_@6Iron_golf_f_cm_np1_ri_med_5.mat golf/Golf_Swing_#6Iron_golf_f_cm_np1_ri_med_5.mat
mv kick_ball/Goal_1_@_2_kick_ball_f_cm_np1_fr_goo_2.mat "kick_ball/Goal_1_&_2_kick_ball_f_cm_np1_fr_goo_2.mat"
mv kick_ball/Amazing_Soccer_@2_kick_ball_u_cm_np1_ba_bad_0.mat kick_ball/Amazing_Soccer_#2_kick_ball_u_cm_np1_ba_bad_0.mat
mv kick_ball/Amazing_Soccer_@2_kick_ball_f_cm_np1_le_bad_2.mat kick_ball/Amazing_Soccer_#2_kick_ball_f_cm_np1_le_bad_2.mat
mv pour/Drink_@18_-_Apple_martini_pour_u_nm_np1_fr_goo_0.mat pour/Drink_#18_-_Apple_martini_pour_u_nm_np1_fr_goo_0.mat
mv pour/Drink_@18_-_Apple_martini_pour_u_nm_np1_fr_goo_1.mat pour/Drink_#18_-_Apple_martini_pour_u_nm_np1_fr_goo_1.mat
mv pour/Drink_@18_-_Apple_martini_pour_u_nm_np1_fr_goo_2.mat pour/Drink_#18_-_Apple_martini_pour_u_nm_np1_fr_goo_2.mat
cd ../../../..
```

3. Split videos into frames.

```shell
python3 hmdb51-frames.py
```

The results will be stored in `data/hmdb51/frames`.

4. Generate mask images from the .mat files.

```shell
python3 mat2mask.py
```

The results will be stored in `data/hmdb51/mat/mask`.

### c. Kinetics-100

1. Download AVA-Kinetics.

```shell
cd data/kinetics100
wget https://storage.googleapis.com/deepmind-media/Datasets/ava_kinetics_v1_0.tar.gz
tar -xzf ava_kinetics_v1_0.tar.gz && rm "$_"
```

2. Generate mask images.

```shell
# TODO
```

The results will be stored in `data/kinetics100/ava/mask`.

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

This step will take several hours and the resulting videos will be stored in `data/{dataset}/{dir}/scene`.

```shell
python3 batch.py
```

Alternatively, download the finished result:

```shell
mkdir -p ../data/ucf101/xgtf/scene && cd "$_"
gdown 1F53RbTXaWW-M6W5I7JDhRU73szm2GvWi
tar -xzf ucf101-scene.tar.gz && rm "$_"
mkdir -p ../../../hmdb51/mat/scene && cd "$_"
gdown 1QqNGenFtKJzNu_xNCN5IpwknYYNPfQbD
tar -xzf hmdb51-scene.tar.gz && rm "$_"
cd ../../../../
```

5. Make a list of the generated scene videos.

The output will be stored in `data/{dataset}/{dir}/scene-list.json`.

```shell
cd ..
python3 list-scene.py
```

6. Switch to another dataset (UCF101/HMDB51) by changing `conf.active.dataset` then rerunning step 4 and step 5.

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
pip install sentence-transformers pandas pyarrow click dynaconf
```

3. Generate relevancy lists. This will generate relevancy files in JSON format in `data/relevancy/{detector}/{dataset}/ids` and `data/relevancy/{detector}/{dataset}/names`.

```shell
python3 relevancy.py
```

4. Run with other detectors and datasets by changing `conf.active.detector` and `conf.active.dataset` then rerunning step 3.

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

Output videos can be generated by setting `unidet.detect.generate_videos` to `true`. The generated videos will be saved in `data/{dataset}/UniDet/detect/videos`.

Alternatively, download the result:

```shell
mkdir -p ../data/ucf101/UniDet/detect/json && cd "$_"
gdown 1DPIVkNk36wn2fScwH02QHA3z-s9fc8hR
tar -xzf ucf101-UniDet-json.tar.gz.tar && rm "$_"
mkdir -p ../../../../hmdb51/UniDet/detect/json && cd "$_"
gdown 1NTbVpps2CSs2RFxdpeNtr7xWF_sJSMLu
tar -xzf hmdb51-UniDet-json.tar.gz && rm "$_"
cd ../../../../
```

6. Filter object detection.

This will select only relevant objects based on the relevancy between the video class names and the detected object names (step F). The output are `.pckl` files stored in `data/{dataset}/UniDet/select/{mode}/dump`.

```shell
python3 batch-select.py
cd ..
```

There are two modes: `actorcutmix` and `intercutmix` (default) configurable in `unidet.active.mode`. Videos and mask images can also be generated by setting `conf.unidet.select.output.video` and `conf.unidet.select.output.mask` to true.

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

This will post-process the `.pckl` files and save the resulting mask files in `data/{dataset}/UniDet/REPP/{mode}/mask` and (optionally) the resulting videos in `data/{dataset}/UniDet/REPP/{mode}/videos`.

```shell
python3 batch.py
```

## H. CutMix

Videos will be mixed with scene-only videos. By default, 10 scene-only videos will be randomly picked from different actions and each input video will be mixed with them. Thus, the resulting mixed videos will be 10 times as many as the original videos.

1. Run script.

This will take several hours and, by default, this will generate mixed videos using the InterCutMix mode.

```shell
python3 cutmix.py
```

2. Optionally, change the mode to ActorCutMix (`conf.cutmix.input.mask.path` and `conf.cutmix.output.path`) and rerun the above command.

3. Make a list of the generated videos.

```shell
python3 list-videos.py
```

4. Obtain the mask ratio from all videos.

```shell
python3 mask-ratio.py
```

## J. Classification

1. Install packages.

```shell
mim install mmengine mmcv
```

2. Install mmaction2.

```shell
pip install -v -e mmaction2/
```

3. Generate file list.

```shell
python3 mmaction2/tools/data/build_file_list.py ucf101 data/ucf101/videos --format videos --shuffle --seed 0
```

4. Train.

```shell
python3 mmaction2/tools/train.py mmaction2/configs/<config>.py --work-dir mmaction2/work_dirs/<dir> --amp --auto-scale-lr
```

5. Test.

```shell
python3 mmaction2/tools/test.py mmaction2/configs/<config>.py work_dirs/<checkpoint>.pth
```

<!-- # Citation

If you find our code useful for your research, please consider citing our paper:

```bibtex
@inproceedings{wihandika2024intercutmix,
   title={InterCutMix: Interaction-aware Scene Debiasing Method for Action Recognition},
   author={Wihandika, Randy Cahya and Mendonça, Israel and Aritsugi, Masayoshi},
   booktitle={},
   year={2025}
}
``` -->
