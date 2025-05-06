# InteractionCutMix

Interaction-aware Scene Debiasing Method for Action Recognition

## 1. Preparation

1. Clone this repository.

```shell
git clone --recursive https://github.com/rendicahya/intercutmix.git
cd intercutmix/
```

2. Create virtual environment.

```shell
python3 -m venv ~/venv/intercutmix/
source ~/venv/intercutmix/bin/activate
pip install -U pip
```

3. Install MMAction2.

```shell
pip install openmim
mim install mmengine mmcv
pip install -v -e mmaction2/
```

## 2. Download datasets

Link the project's `data/` directory with MMAction2's `data/` directory.

```shell
cd mmaction2/
ln -s ../data/ ./
cd -
```

### a. UCF101

1. Download videos.

```shell
cd mmaction2/tools/data/ucf101/
bash download_videos.sh
```

This downloads `UCF101.rar` and unrars it to `data/ucf101/videos`.

2. Verify the number of videos.

```shell
find videos/ -type f | wc -l
```

Expected: `13,320`.

3. Download annotations.

```shell
bash download_annotations.sh
cd -
```

This creates the `annotations/` directory and puts files in it.

4. Generate splits.

```shell
python3 mmaction2/tools/data/build_file_list.py ucf101 data/ucf101/videos/ --format videos --shuffle --seed 0
```

This creates:

- `ucf101_train_split_{1-3}_videos.txt`
- `ucf101_val_split_{1-3}_videos.txt`

5. Verify files and directories.

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
├── videos/
│   ├── ApplyEyeMakeup/
│   │   ├── v_ApplyEyeMakeup_g01_c01.avi
│   │   ├── v_ApplyEyeMakeup_g01_c02.avi
│   │   ├── v_ApplyEyeMakeup_g01_c03.avi
│   │   └── ...
│   └── ...
├── ucf101_train_split_1_videos.txt
├── ucf101_train_split_2_videos.txt
├── ucf101_train_split_3_videos.txt
├── ucf101_val_split_1_videos.txt
├── ucf101_val_split_2_videos.txt
└── ucf101_val_split_3_videos.txt
```

### b. HMDB51

1. Download videos.

```shell
cd mmaction2/tools/data/hmdb51/
bash download_videos.sh
```

This downloads `hmdb51_org.rar` and unrars it to `data/hmdb51/videos`.

2. Verify the number of videos.

```shell
find videos/ -type f | wc -l
```

Expected: `6,766`.

3. Download annotations.

```shell
bash download_annotations.sh
cd -
```

This creates the `annotations/` directory and puts files in it.

4. Generate splits.

```shell
python3 mmaction2/tools/data/build_file_list.py hmdb51 data/hmdb51/videos/ --format videos --shuffle --seed 0
```

This creates:

- `hmdb51_train_split_{1-3}_videos.txt`
- `hmdb51_val_split_{1-3}_videos.txt`

5. Verify files and directories.

```shell
intercutmix/data/hmdb51/
├── annotations/
│   ├── brush_hair_test_split1.txt
│   ├── brush_hair_test_split2.txt
│   ├── brush_hair_test_split3.txt
│   └── ...
├── frames/
│   ├── brush_hair/
│   │   ├── April_09_brush_hair_u_nm_np1_ba_goo_0/
│   │   │   ├── img_00001.png
│   │   │   ├── img_00002.png
│   │   │   ├── img_00003.png
│   │   │   └── ...
│   │   └── ...
│   └── ...
├─── videos/
│   ├── brush_hair/
│   │   ├── April_09_brush_hair_u_nm_np1_ba_goo_0.avi
│   │   ├── April_09_brush_hair_u_nm_np1_ba_goo_1.avi
│   │   ├── April_09_brush_hair_u_nm_np1_ba_goo_2.avi
│   │   └── ...
│   └── ...
├── hmdb51_train_split_1_videos.txt
├── hmdb51_train_split_2_videos.txt
├── hmdb51_train_split_3_videos.txt
├── hmdb51_val_split_1_videos.txt
├── hmdb51_val_split_2_videos.txt
└── hmdb51_val_split_3_videos.txt
```

### c. Kinetics100

1. Download videos.

```shell
mkdir -p data/kinetics100
cd data/kinetics100
gdown 1_gPSZDo_yasyEbtfs0m2edZAzhsr8dU4
tar xzf kinetics100.tar.gz
mv kinetics_100/ videos/
rm kinetics100.tar.gz
```

This downloads `kinetics100.tar.gz` and extracts it to `data/kinetics100/videos`.

2. Verify the number of videos.

```shell
find videos/ -type f | wc -l
```

Expected: `9,999`.

3. Open `settings.json` and set `active.dataset: kinetics100`.

4. Generate annotations.

```shell
python3 tools/data/classind.py
```

This creates `data/kinetics100/annotations/classInd.txt`.

5. Generate splits.

```shell
python3 tools/data/split.py
```

This creates:
- `data/kinetics100/kinetics100_train_split_1_videos.txt`
- `data/kinetics100/kinetics100_val_split_1_videos.txt`

## 3. Inference

1. Download checkpoints.

```shell
pip install gdown
mkdir checkpoints/
gdown -O checkpoints/ <download-key>
```

Refer the following table for `<download-key>`.

2. Run inference.

```shell
python3 mmaction/tools/test.py <config-path> <checkpoint-path>
```

Refer the following table for `<config-path>` and `<checkpoint-path>`.

| **Dataset** | **Top-1** | **Top-5** | **Config Path**                                                                                                                   | **Checkpoint Path**                        | **Download Key**                  |
|-------------|-----------|-----------|-----------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------|-----------------------------------|
| UCF101      | 87.84%    | 97.12%    | mmaction2/configs/recognition/c3d-ucf101-soft/c3d_sports1m-pretrained_8xb64-16x1x1-100e_ucf101-rgb-intercutmix-p0.5-mmr0.05-a2.py | checkpoints/ucf101-icm-p0.5-mmr0.05-a2.pth | 1Aynmc64VpLJEXBeNe-Bq_u787h5pX2aq |
| HMDB51      | 55.75%    | 85.49%    | mmaction2/configs/recognition/c3d-hmdb51-soft/c3d_sports1m-pretrained_8xb64-16x1x1-100e_hmdb51-rgb-intercutmix-p0.5-mmr0.05-a2.py | checkpoints/hmdb51-icm-p0.5-mmr0.05.pth    | 1cb3gfG3qJUAAsrMSXcXm0LKHLSuuDGd6 |
| Kinetics100 |           |           |                                                                                                                                   |                                            |                                   |