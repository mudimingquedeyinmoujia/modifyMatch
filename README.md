# modifyMatch
Modified Colmap image matching algorithom, for accelerating matching of image pairs (usually 1000+).
Compared to exhaustive matching (Colmap default), this way can save nearly x20+ time.


# Setup
Please install [Colmap](https://colmap.github.io/install.html) first.
When you are setting, please **deactivate conda** environment to avoid potential bugs.
- numpy
- scipy
The dataset path should be like:
```
path/to/your/dataset
|---dataset
|------input
|---------img_01.png
|---------img_02.png
......
```


# Usage
#### 1. use exhaustive matching (default of colmap)
```shell
python run_modifyMatch.py -s path/to/your/dataset --match_mode E --threads 64 --gpu_id 3
```

#### 2. use vocab tree matching (provided of colmap)
```shell
python run_modifyMatch.py -s path/to/your/dataset --match_mode T --threads 64 --gpu_id 3
```

#### 3. use sparse matching
```shell
python run_modifyMatch.py -s path/to/your/dataset --match_mode O_neighborJump --threads 64 --gpu_id 3
```
##### Params of sparse matching
- `--match_mode`: O_neighborJump, O_neighbor, O_jump
- `--arg_r`: neighbor number to match (radius), default 5
- `--arg_k`: jump interval, default 10
- `--arg_w`: concentric annulus width, default 1


# Special Case

- if BA failed, change `--camera`: default `OPENCV`, can be `SIMPLE_PINHOLE`
- or change input params of colmap BA, in run_modifyMatch.py