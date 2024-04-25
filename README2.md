# InterCutMix

# Steps

## A. Preparation

1. Clone this repository and the submodules.

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

3. Install mmaction2

1. Install dependencies.

```shell
pip install openmim
mim install mmengine mmcv
```

2. Install mmaction2.

```shell
pip install -v -e mmaction2/
```

## B. Download datasets

### a. UCF101

1. Download videos.

```shell
mkdir -p data/ucf101/ && cd "$_"
wget https://www.crcv.ucf.edu/datasets/human-actions/ucf101/UCF101.rar --no-check-certificate
unrar x UCF101.rar && rm "$_"
mv UCF-101 videos
```

2. Download annotations.

```shell
wget https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip --no-check-certificate
unzip -q UCF101TrainTestSplits-RecognitionTask.zip && rm "$_"
mv ucfTrainTestlist/ annotations/
cd ../../
```

### b. HMDB51

1. Download videos.

```shell
mkdir -p data/hmdb51/videos/ && cd "$_"
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

3. Extract frames.

```shell
python mmaction2/tools/data/build_rawframes.py data/hmdb51/videos/ data/hmdb51/frames/ --task rgb --num-worker 16 --out-format png --use-opencv
```

3. Check structure.

```shell
intercutmix/data/hmdb51/
├── frames/
|   ├── brush_hair/
|   │   ├── April_09_brush_hair_u_nm_np1_ba_goo_0
|   │   │   ├── img_00001.png
|   │   │   ├── img_00002.png
|   │   │   ├── img_00003.png
|   │   │   └── ...
|   │   ├── April_09_brush_hair_u_nm_np1_ba_goo_1
|   │   │   ├── img_00001.png
|   │   │   ├── img_00002.png
|   │   │   ├── img_00003.png
|   │   │   └── ...
|   |   └── ...
|   ├── cartwheel/(Rad)Schlag_die_Bank!_cartwheel_f_cm_np1_le_med_0
|   │   ├── (Rad)Schlag_die_Bank!_cartwheel_f_cm_np1_le_med_0
|   │   │   ├── img_00001.png
|   │   │   ├── img_00002.png
|   │   │   ├── img_00003.png
|   │   │   └── ...
|   │   ├── Acrobacias_de_un_fenomeno_cartwheel_f_cm_np1_ba_bad_8
|   │   │   ├── img_00001.png
|   │   │   ├── img_00002.png
|   │   │   ├── img_00003.png
|   │   │   └── ...
|   |   └── ...
|   └── ...
├── testTrainMulti_7030_splits/
│   ├── brush_hair_test_split1.txt
│   ├── brush_hair_test_split2.txt
│   ├── brush_hair_test_split3.txt
|   └── ...
└─── videos/
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