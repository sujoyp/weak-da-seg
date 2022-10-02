# Installation

Clone this repository, install Anaconda (or Miniconda), then do the following:

```
conda create -n da_seg_py35 python=3.5 -y
conda activate da_seg_py35

conda install pytorch=0.4.1 cuda80 -c pytorch
conda install numpy pillow
conda install torchvision -c pytorch
```

While the code was developed with the above environment, it should also work with a more modern version of Python and PyTorch (only verified for inference code, though):

```
conda create -n da_seg_py38 python=3.8 -y
conda activate da_seg_py38
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia -y
conda install numpy pillow -y
```
