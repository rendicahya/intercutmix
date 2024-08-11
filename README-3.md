# InteractionCutMix

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

Link project's `data/` directory with MMAction2's `data/` directory.

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

4. Generate splits.

```shell
python3 mmaction2/tools/data/build_file_list.py ucf101 data/ucf101/videos/ --format videos --shuffle --seed 0
```

5. Verify structure.

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

4. Generate splits.

```shell
python3 mmaction2/tools/data/build_file_list.py hmdb51 data/hmdb51/videos/ --format videos --shuffle --seed 0
```

5. Verify structure.

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

## 3. Inference

1. Download checkpoints.

```shell
pip install gdown
mkdir checkpoints/
gdown -O checkpoints/ <download-key>
```

| **Dataset** | **Top-1** | **Top-5** | **Download Key** |
|-------------|-----------|-----------|------------------|
| UCF101      | xxxx%     | xxxx%     | xxxx             |
| HMDB51      | xxxx%     | xxxx%     | xxxx             |

2. Run inference.

```shell
python <config> <checkpoint>
```

| **Dataset** | **Config**                                                                                                                        | **Checkpoint**                             |
|-------------|-----------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------|
| UCF101      | mmaction2/configs/recognition/c3d-ucf101-soft/c3d_sports1m-pretrained_8xb64-16x1x1-100e_ucf101-rgb-intercutmix-p0.5-mmr0.05-a2.py | checkpoints/ucf101-icm-0.5-mmr-0.05-a2.pth |
| HMDB51      | mmaction2/configs/recognition/c3d-hmdb51-soft/c3d_sports1m-pretrained_8xb64-16x1x1-100e_hmdb51-rgb-intercutmix-p0.5-mmr0.05-a2.py | checkpoints/hmdb51-icm-0.5-mmr-0.05-a2.pth |
