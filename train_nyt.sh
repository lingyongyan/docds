#!/bin/bash

count=15
loss="dsloss"
export NYT_DIR=dataset/NYT/$count
export CUDA_VISIBLE_DEVICES="8"
model_name=docre_${loss}${suffix}_${count}

nohup python -u main.py \
  --do_train \
  --retain_entity \
  --loss_func ${loss} \
  --risk_sensitive \
  --model_name_or_path bert-base-uncased \
  --train_file $NYT_DIR/${count}_train${suffix}.json \
  --predict_file $NYT_DIR/${count}_dev${suffix}.json \
  --model_type bert \
  --do_lower_case \
  --is_with_negative \
  --per_gpu_train_batch_size 32 \
  --per_gpu_eval_batch_size 48 \
  --learning_rate 3e-5 \
  --warmup_propotion 0.1 \
  --max_grad_norm 1.0 \
  --gamma 2.0 \
  --lambda_weight 0.5 \
  --logging_steps 100 \
  --num_train_epochs 3 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --overwrite_output_dir \
  --output_dir checkpoints/nyt_bert_${model_name}/ > nyt_bert_${model_name}.log 2>&1 &
