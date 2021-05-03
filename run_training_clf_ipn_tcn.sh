#!/bin/bash
python main.py \
	--root_path /home/cyjiang \
	--video_path IPN-hand-dataset \
	--annotation_path IPN-hand/annotation_ipnGesture/ipnall_but_None.json \
	--result_path IPN-hand/results_ipn_adam_avg_max_fc3_resnext_cropped \
	--dataset ipn \
	--sample_duration 32 \
    --learning_rate 0.001 \
    --model mstcn \
	--embedding_dim 128 \
	--batch_size 96 \
	--n_classes 13 \
	--n_finetune_classes 13 \
	--n_threads 16 \
	--checkpoint 1 \
	--modality RGB-seg \
	--train_crop random \
	--n_val_samples 1 \
	--test_subset test \
    --n_epochs 1000 \
    --store_name ipnClfRs_TCN_b32 \
