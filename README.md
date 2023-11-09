# InterCutMix

Interaction-aware Scene Debiasing Method for Action Recognition

# Steps

1. Clone this repository.

```shell
git clone https://github.com/rendicahya/intercutmix.git
```

2. Install packages

```shell
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

4. Generate relevancy matrices. This will create a subdirectory `relevancy/unidet-matrix` and generate several CSV files.

```shell
python relevancy_matrix.py
```

5. Generate relevancy lists. This will create subdirectories `relevancy/unidet-relevant-ids` and `relevancy/unidet-relevant-names` and generate several JSON files.

```shell
python relevancy_list.py
```
