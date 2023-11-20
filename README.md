# InterCutMix

Interaction-aware Scene Debiasing Method for Action Recognition

# Steps

1. Clone this repository and the required submodules.

```shell
git clone --recursive https://github.com/rendicahya/intercutmix.git
```

2. Install packages

```shell
# pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
# pip install detectron2 https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
pip install -r requirements.txt
```

3. Edit `conf.json`.

```json
{
  "ucf101": {
    "path": "/your/ucf101/path"
  }
}
```

4. Generate relevancy lists. This will create subdirectories `relevancy/unidet-relevant-ids` and `relevancy/unidet-relevant-names` and generate several JSON files.

```shell
python relevancy.py
```
