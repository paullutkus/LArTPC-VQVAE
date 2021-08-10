#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 ../test.py \
--gpu 0 \
--k 256 \
--d 8 \
--beta 1 \
--vqvae_batch_size 250 \
--vqvae_epochs 5 \
--vqvae_lr 3e-4 \
--vqvae_layers 16 32 \
--pcnn_features 64 \
--pcnn_epochs 10
