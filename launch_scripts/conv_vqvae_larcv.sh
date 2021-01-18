#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 ../train.py \
--gpu 3 \
--k 256 \
--d 8 \
--beta 1 \
--vqvae_batch_size 250 \
--vqvae_epochs 50 \
--vqvae_lr 3e-4 \
--vqvae_layers 16 32
