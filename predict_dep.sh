#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32  python  predict.py \
 --save_weights_path=weights/tinydarknet5_shortcuts \
 --epoch_number=61 \
 --test_images="data/stanford/images_prepped_test/" \
 --test_depth="data/stanford/depth_prepped_test/" \
 --output_path="data/predictions_stanford/" \
 --n_classes=12 \
 --model_name="tinyyolonet5" \
 --input_height=224 \
 --input_width=224 
