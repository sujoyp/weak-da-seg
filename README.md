# Domain Adaptive Semantic Segmentation Using Weak Labels

> **Domain Adaptive Semantic Segmentation Using Weak Labels**
> Sujoy Paul, Yi-Hsuan Tsai, Samuel Schulter, Amit K. Roy-Chowdhury, Manmohan Chandraker
> ECCV 2020
> [arXiv](https://arxiv.org/abs/2007.15176), [project](https://www.nec-labs.com/~mas/WeakSegDA/)

<p align="center"><img src='docs/teaser.jpg' align="center" width="640px"></p>


## Highlights

* Improve domain adaptive semantic segmentation with weak labels
* Weak labels can be pseudo labels (unsupervised domain adaptation) or human-annotated (weakly-supervised domain adaptation)
* Weak labels can be image-level tags or point annotations
* 56.4% mIoU on GTA->CityScapes when using point annotations (45 seconds annotation time per image)

<p align="center"><img src='docs/result.jpg' align="center" width="500px"></p>


## Installation

See [docs/INSTALL.md](docs/INSTALL.md) for installation instructions.


## Model zoo

All models can be downloaded from HERE

Definitions of the prefix in each model name:

* *pseudoweak*: use weak label loss with pseudo-weak labels, `--use-pseudo` and `--use-weak`
* *cw*: use category-wise alignment, `--use-weak-cw`
* *pa*: use pixel-level alignment, `--use_pixeladapt`
* *weak-image*: use weak label loss with ground truth image-level weak labels, `--use-pseudo` is false and `--use-weak`
* *weak-1point*: use weak label loss with ground truth point-level weak labels, `--use-pointloss` and `--use-weak`


### Source: GTA5

| Filename                               | mIoU  | Paper                                                |
|----------------------------------------|-------|------------------------------------------------------|
| `gta5-cityscapes-pseudoweak-cw-pa.pth` | 48.02 | Table 1 - "Ours (UDA)" -- NOTE: Paper reports 48.2!! |
| `gta5-cityscapes-weak-image-cw-pa.pth` | 53.02 | Table 1 - "Ours (WDA: Image)"                        |
| `gta5-cityscapes-weak-1point.pth`      | 56.42 | Table 1 - "Ours (WDA: Point)"                        |
|----------------------------------------|-------|------------------------------------------------------|
| `gta5-cityscapes-pseudoweak.pth`       | 44.2  | Table 3 - Pseudo-Weak +L_c                           |
| `gta5-cityscapes-pseudoweak-cw.pth`    | 46.55 | Table 3 - Pseudo-Weak +L_c +L^C_adv                  |


### Source: Synthia

| Filename                                  | mIoU* | mIoU  | Paper                                |
|-------------------------------------------|-------|-------|--------------------------------------|
| `synthia-cityscapes-pseudoweak-cw-pa.pth` | 44.27 | 51.9  | Table 2 - "Ours (UDA)"               |
| `synthia-cityscapes-weak-image-cw-pa.pth` | 50.6  | 58.51 | Table 2 - "Ours (WDA: Image)"        |
|-------------------------------------------|-------|-------|--------------------------------------|
| `synthia-cityscapes-pseudoweak.pth`       | 41.74 | 49.08 | Table 4 - Pseudo-Weak +L_c           |
| `synthia-cityscapes-pseudoweak-cw.pth`    | 42.65 | 49.91 | Table 4 - Pseudo-Weak +L_c + L^C_adv |



## Code structure

* Definitions of experimental options, settings, and parameters are in `daweak/util/option.py`
* Dataset loaders and model definitions are in the `daweak` library folder
* Main executing files are in the root folder (see below for testing and training details)
* Main script for training/testing is `run_weak_da.sh`

See [docs/TESTING.md](docs/TESTING.md) and [docs/TRAINING.md](docs/TRAINING.md) for more details on evaluating a (pre-)trained model and training a new model, respectively.


## License

The code is released under the [MIT License](LICENSE).


## Citation

    @inproceedings{paul2020daweak,
      title={Domain Adaptive Semantic Segmentation Using Weak Labels},
      author={Sujoy Paul and Yi-Hsuan Tsai and Samuel Schulter and Amit K. Roy-Chowdhury and Manmohan Chandraker}
      booktitle={ECCV},
      year={2020}
    }
