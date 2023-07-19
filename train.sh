#!/bin/sh

python train.py \
   --batch_size 64 \
   --rnn_layers 5 \
   --learning_rate 0.001 \
   --epochs 25 \
   --embedding_dim 64 \
   --hidden_size 128 \
   --dropout_prob 0.5 \
   --device mps

python classifier.py --text "All tilapias should be burnt alive"
python classifier.py --text "All tilapias should be removed from extistence from this planet. They are useless "
python classifier.py --text "they are useless"
