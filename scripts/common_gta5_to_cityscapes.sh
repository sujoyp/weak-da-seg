#!/bin/bash

# model
model="deeplab"

# dataset
source="gta5"
target="cityscapes"

if [ "$source" = "gta5" ]
then
    source_path="datasets/gta5"
fi

if [ "$target" = "cityscapes" ]
then
    target_path="datasets/cityscapes"
fi

if [ "$source" = "gta5" ]
then
    source_size="1280,720"
fi

if [ "$target" = "cityscapes" ]
then
    target_size='1024,512'
fi

if [ "$source" = "gta5" ]
then
    num_classes=19
fi

source_split="train"
target_split="train"
test_split="val"

# parameters
batch_size=1
num_steps=250000
num_steps_stop=120000

lambda_seg=0.0
lambda_adv1=0.0
lambda_adv2=0.001
lambda_weak2=0.01
lambda_weak_cwadv2=0.001

lr=2.5e-4
lr_d=1e-4

save_step=1000
print_step=100
pweak_th=0.2

#
# training models
#
# - use [--val] to run testing during training for selecting better models (require a validation set in the target domain with ground truths)
# - If `--val-only` is added, the code runs the testing mode.


## training models when GTA is the source domain
pretrain="models/"$source"_pretrained.pth"

## training models when other dataset is the source domain ##
# pretrain="models/MS_DeepLab_resnet_pretrained_COCO_init.pth"


#
# testing models
#   NB: Do not forget to add the argument `--val-only` below
#
# pretrain="model/gta5-cityscapes-pseudoweak-cw.pth"

