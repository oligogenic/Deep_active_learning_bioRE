#!/bin/bash

accelerate launch --config_file config_accelerate.yaml main.py \
  --do_active \
  --model_name pubmedbert \
  --max_seq_length 256 \
  --num_epochs 3 \
  --train_batch_size 16 \
  --eval_batch_size 32 \
  --predict_batch_size 32 \
  --binary \
  --learning_rate 2e-5 \
  --data_dir data/DDI_hf \
  --output_dir out_DDI_bin_pubmed_2500 \
  --cross_val 5 \
  --num_repeats 3 \
  --size_seed 2500 \
  --batch_size_active 2500 \
  --restart_model
