#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

python3 ../offline_test.py \
	--root_path /home/cyjiang \
	--video_path IPN-hand-dataset \
	--annotation_path IPN-hand/annotation_ipnGesture/ipnall_but_None.json \
	--result_path IPN-hand/results_ipn \
	--resume_path IPN-hand-dataset/baseline_models/resnext/ipnClfRs_jes32rb32_resnext-101.pth \
    --store_name ipnClfRs_jes32r_b32 \
	--modality RGB-seg \
	--dataset ipn \
	--sample_duration 32 \
    --model resnext \
	--model_depth 101 \
	--resnet_shortcut B \
	--batch_size 1 \
	--n_classes 13 \
	--n_finetune_classes 13 \
	--n_val_samples 1 \
	--test_subset test \
    --n_epochs 100 \
    --no_train \
    --no_val \
    --test \
