## Evaluating a (pre-)trained model on Cityscapes

* Download the [Cityscapes Dataset](https://www.cityscapes-dataset.com/) as the target domain and for evaluation of a (pre-)trained model
* Use the model zoo to pick a pre-trained model, e.g., `gta5-cityscapes-pseudoweak-cw.pth`
* Results are saved in the `result` folder (default)
* Specify `--val-only` in `run_weak_da.sh` to enter the testing mode
* Define the target dataset by specifying `target`, `target_path` for the dataset path, the resized targe image size `target_size`, and the evaluation set `test_split`, e.g., val
* Define `pretrain` to load the model for testing, e.g., `model/gta5-cityscapes-pseudoweak-cw.pth`
* Segmentation outputs are saved with the colorized format via a pre-defined color palette in `daweak/engine/trainer_base.py` (for 19 categories as in Cityscapes)
