#!/bin/bash

accelerate launch --config_file config_accelerate.yaml main.py \
  --do_active \
  --model_name pubmedbert \
  --max_seq_length 256 \
  --num_epochs 10 \
  --train_batch_size 16 \
  --eval_batch_size 32 \
  --predict_batch_size 32 \
  --learning_rate 1e-5 \
  --binary \
  --data_dir data/aimed/processed \
  --output_dir out_aimed_bin_256_pubmed_500 \
  --cross_val 5 \
  --num_repeats 3 \
  --size_seed 500 \
  --batch_size_active 500 \
  --restart_model
