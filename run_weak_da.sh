#!/bin/bash

# model
model="deeplab"

# dataset
source="gta5"
target="cityscapes"

if [ "$source" = "gta5" ]
then
    source_path="path/to/GTAdata"
fi

if [ "$target" = "cityscapes" ]
then
    target_path="path/to/cityscapes/data"
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

# save models
## training models when GTA is the source domain ##
# pretrain="model/"$source"_pretrained.pth"

## training models when other dataset is the source domain ##
# pretrain="model/MS_DeepLab_resnet_pretrained_COCO_noclasslayer.pth"

## testing models ##
# pretrain="snapshot/gta5-cityscapes-weak-cw/G-gta5-cityscapes.pth"
pretrain="model/gta5-cityscapes-pseudoweak-cw.pth"

snapshot_dir="snapshot/"$source"-"$target"-weak-cw"
result_dir="result/"$source"-"$target"-weak-cw"
save_step=1000
print_step=100
pweak_th=0.2

# use [--val] to run testing during training for selecting better models (require a validation set in the target domain with ground truths)

# Run the code
# IF `--val-only` IS ADDED, THE CODE RUNS THE TESTING MODE.
python train.py \
    --model $model \
    --dataset-source $source \
    --dataset-target $target \
    --data-path-source $source_path \
    --data-path-target $target_path \
    --input-size-source $source_size \
    --input-size-target $target_size \
    --num-classes $num_classes \
    --source-split $source_split \
    --target-split $target_split \
    --test-split $test_split \
    --batch-size $batch_size \
    --num-steps $num_steps \
    --num-steps-stop $num_steps_stop \
    --lambda-seg $lambda_seg \
    --lambda-adv-target1 $lambda_adv1 \
    --lambda-adv-target2 $lambda_adv2 \
    --lambda-weak-cwadv2 $lambda_weak_cwadv2 \
    --lambda-weak-target2 $lambda_weak2 \
    --learning-rate $lr \
    --learning-rate-D $lr_d \
    --restore-from $pretrain \
    --pweak-th $pweak_th \
    --snapshot-dir $snapshot_dir \
    --result-dir $result_dir \
    --save-pred-every $save_step \
    --print-loss-every $print_step \
    --use-weak \
    --use-weak-cw \
    --use-pseudo \
    --val-only