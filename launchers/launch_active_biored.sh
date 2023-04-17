#!/bin/bash

accelerate launch --config_file config_accelerate.yaml main.py \
  --do_active \
  --model_name pubmedbert \
  --max_seq_length 512 \
  --num_epochs 3 \
  --train_batch_size 16 \
  --eval_batch_size 32 \
  --predict_batch_size 32 \
  --learning_rate 2e-5 \
  --binary \
  --data_dir data/BioRED/processed/all \
  --output_dir out_biored_all_pubmed_2500 \
  --cross_val 5 \
  --num_repeats 3 \
  --size_seed 2500 \
  --batch_size_active 2500 \
  --restart_model
