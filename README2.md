# InterCutMix

## A. Preparation

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

## B. Install mmaction2

1. Install dependencies.

```shell
pip install openmim
mim install mmengine mmcv
```

2. Install MMAction2.

```shell
pip install -v -e mmaction2/
```

## C. Download datasets

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

2. Verify.

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

2. Download annotations.

```shell
bash download_annotations.sh
cd -
```

3. Extract frames.

```shell
python mmaction2/tools/data/build_rawframes.py data/hmdb51/videos/ data/hmdb51/frames/ --task rgb --num-worker 16 --out-format png --use-opencv
```

4. Generate file list.

```shell
python3 mmaction2/tools/data/build_file_list.py hmdb51 data/hmdb51/videos/ --format videos --shuffle --seed 0
```

5. Check structure.

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

### c. Kinetics-100

1. Download videos.

```shell
bash kinetics-dataset/k400_downloader.sh
bash kinetics-dataset/k400_extractor.sh
```

2. Remove .tar.gz files.

```shell
rm -rf data/kinetics400/targz/
```

- Symlink
```shell
ln -s /nas.dbms/randy/datasets/kinetics400/train/ /nas.dbms/randy/projects/intercutmix/data/kinetics400
ln -s /nas.dbms/randy/datasets/kinetics400/val/ /nas.dbms/randy/projects/intercutmix/data/kinetics400
ln -s /nas.dbms/randy/datasets/kinetics400/test/ /nas.dbms/randy/projects/intercutmix/data/kinetics400
ln -s /nas.dbms/randy/datasets/kinetics400/replacement/replacement_for_corrupted_k400/ /nas.dbms/randy/projects/intercutmix/data/kinetics400/replacement
```

3. Build Kinetics-100 from Kinetics-400.

```shell
python3 tools/data/kinetics100/make.py
```
