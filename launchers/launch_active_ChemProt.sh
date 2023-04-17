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
  --data_dir data/chemprot_hf \
  --output_dir out_ChemProt_bin_pubmed_3600 \
  --cross_val 5 \
  --num_repeats 3 \
  --size_seed 3600 \
  --batch_size_active 3600 \
  --restart_model
