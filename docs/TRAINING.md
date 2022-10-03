# Main Options in `run_weak_da.sh` for training on your own datasets

* Download the [GTA5 Dataset](https://download.visinf.tu-darmstadt.de/data/from_games/) as the source domain
* Download the [SYNTHIA Dataset](https://synthia-dataset.net) as the source domain
* Download the [Cityscapes Dataset](https://www.cityscapes-dataset.com/) as the target domain

In general, the `run_weak_da.sh` script gives an example of how to train a model.

* During training, models are saved in the `snapshot` folder (default)
* Specify `--val` in `run_weak_da.sh` if there is a validation set with ground truths in the target domain
* The initialization of DeepLab models can be downloaded for [GTA5](https://drive.google.com/file/d/1n0zrw_utoFPoR--KwBy8FOQSWWPcKPwy/view?usp=sharing) and [other source datasets](https://drive.google.com/file/d/1mykAx3BW9B7upnIK6rZFDfpvtctWI11m/view?usp=sharing).


## Dataset
* Define the source dataset by specifying `source`, `source_path` for the dataset path, the resized source image size `source_size`, and the evaluation set `source_split`, e.g., train

* Define the target dataset by specifying `target`, `target_path` for the dataset path, the resized target image size `target_size`, and the evaluation set `target_split`, e.g., train

* To determine the input size of images to the model, i.e., `source_size` and `target_size`, it would be better to use the size as larger as possible.
If the GPU cannot fit the original size, one can specify a smaller size but should still keep the same aspect ratio, e.g., if the original size is (2048, 1024), one can specify (1024, 512).


### How to use pixel-adapted (via CycleGAN) images

Need the images for the source domain (GTA5, SYNTHIA). The code uses these images during training if the option `use-pixeladapt` is turned ON. The code assumes the data to be stored at `data/cycleGAN_output/gta5` and `data/cycleGAN_output/synthia` for GTA5 and SYNTHIA, respectively. You can create symbolic links. The underlying structure for both paths should follow the same as how the GTA5 and SYNTHIA images are structured, with the same file names. For SYNTHIA, for example, it's `RGB/<filename>`.


## Data Loader
* In order to load your dataset, one would need to write your own data loader. Please check examples in the `daweak/dataset` folder, where `gta5.py` is one example for source and `cityscapes.py` is one for target.
Note that, as long as the images are successfully loaded, the format of the data loader can be different (depend on the data structure of the dataset).


## Semantic Category
* In the current setting, 19 semantic categories are used as in Cityscapes. If one needs a different set of categories, the number of categories `num_classes` should be changed.
Therefore, one would also have your own category definition as defined in your source dataset.


## Parameters for Training

* When using GTA5 as the source domain, set the pre-trained model as `pretrain="models/gta5_pretrained.pth"`. If the source domain is the other dataset, set `pretrain="models/MS_DeepLab_resnet_pretrained_COCO_init.pth"`.

* There are two options: 1) purely unsupervised domain adaptation (no any ground truth annotations in the target domain), and 2) weakly-supervised domain adaptation (image-level or point-level ground truths in the target domain)

* Weak label loss: `--use-weak`, Category-wise feature alignment: `--use-weak-cw`

* When there is no weak ground truth annotations available in the target domain, add the `--use-pseudo` option and use `lambda_weak2=0.01`, otherwise use `lambda_weak2=0.2`

* Specify whether to use point-level weak supervision by setting `--use-pointloss`

* Specify whether to use pixel-level adaptation by setting `--use-pixeladapt`, e.g., in the GTA5 data loader, line 44 in `./daweak/dataset/gta5.py`.

* Training Iteration: specify how many itertations `num_steps` to train the model and when to do early stopping `num_steps_stop` for efficiency, as training too long may overfit to the source data, which is not good for the target domain.
One practice is to specify `num_steps` roughly equal to 10 times the number of source images, i.e., 10 epochs.


## Save Model

* Choose where to save the models `snapshot_dir`

* Overwrite the model for every `save_step` iterations and print loss values for every `print_step` iterations

* If there is a validation set `test_split` with ground truths available in the target domain, one can use `--val` to perform online evaluation.
This will be performed for every `save_step` iterations. It will also keep tracing the best accuracy and save the best model in `snapshot_dir`.


## Other Guidelines

* Since the domain gap between source and target could be very large, it would be better to prepare a source dataset (e.g., GTA5) with:
1) Better rendering quality
2) Higher diversity, e.g., type of objects and road scenes
3) Larger quantity, e.g., more then 10K images

* If there is no validation data with ground truths, one should not use `--val`
To know which model is better in the target domain, I would recommend to at least manually annotate a small set for validation.
Otherwise, one might need to test various models at different iterations and visualize the quality.

* After adaptation, one can expect that for objects or stuffs that are more dominant in the source data, the performance is better, e.g., road, building, sky, tree, person, and car.
For small objects or the ones that appear less frequently, it would be more difficult to adapt.
