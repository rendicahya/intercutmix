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
```

```shell
wget https://www.crcv.ucf.edu/datasets/human-actions/ucf101/UCF101.rar https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip --no-check-certificate
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

2. Generate mask images. This will take a few minutes and the results will be stored in `data/ucf101/xgtf-mask`.

```shell
python xgtf_to_mask.py
```

## C. Generate scene videos

This step uses mask images generated in step B. Therefore, make sure that step B has been successfully completed.

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

Alternatively, in case the above download fails:

```shell
wget https://download847.mediafire.com/ou5x8bq0q9sgku70mNh31V5epldWxIMWhR7n2ZU7vhIoJqAg-QwZEFMqXQ3Y9gckOviT5ItorlxGBJRFg6WYuxHmkkkirUJNefaB9OdExmXDVUaZc_Gwua1BRanev3ONCDwvk1jbc5KcKuZMblIBvG6UyFoqxxzK29ejxXK3GMWOyw/mrd06il310cklxh/E2FGVI-HQ-CVPR22.pth -P release_model/
```

4. Generate videos. This step will take several hours and the resulting videos will be stored in `data/ucf101/scene-xgtf`.

```shell
python batch.py
```

5. Make a list of the generated scene videos. Do this step only after the video generation step has been completed. The list will be stored in `data/ucf101/scene-xgtf.json`.

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

2. Generate relevancy lists. This will generate relevancy files in JSON format and save them in `relevancy/unidet-relevant-ids` and `relevancy/unidet-relevant-names`.

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
gdown 1HvUv399Vie69dIOQX0gnjkCM0JUI9dqI -O models/
```

Alternatively, in case the above download fails:

```shell
wget https://download1649.mediafire.com/jjyqufty4b1gXPpH0tUaoqp-MK0xgi-89SKBJqYjH1TLSjrDqufwW_LIXF0OeiiH8tx2BxZ71cm0S_dg7xpkb0Y_sWdGD9Ca0b8eyrU32VF8ZVSUc8IKibOi_wb6DkDSR3I3cRfIVKqArhw0U_JJEpewtkgHXjdl3FCNSJ4Kv4y53Q/wdxfkp1wyc0ccxl/Unified_learned_OCIM_RS200_6x%2B2x.pth -P models/
```

4. Run object detection. This step detects all objects with a confidence threshold of (by default) 0.5. The detection results for each video will be saved in a JSON file in `data/ucf101/unidet-json` and the generated videos will be saved in `data/ucf101/unidet`. If you want to speed up the process by generating only the JSON files without generating videos, modify `config.py` and set `unidet.detect.output.video.generate` to `false`.

```shell
python batch-detect.py
```

5. Filter object detection. This will select only relevant objects based on the relevancy between the video class names and the detected object names. The output of this step is mask images stored in `data/ucf101/unidet-actor-mask` and the generated videos will be saved in `data/ucf101/unidet-actor`. Therefore, make sure that `batch-detect.py` and step D (relevancy) have been successfully completed. If you want to speed up the process by generating only the JSON files without generating videos, modify `config.py` and set `unidet.select.output.video.generate` to `false`.

```shell
python batch-select.py
```
