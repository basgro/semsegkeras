#!/bin/bash

#THEANO_FLAGS=device=gpu,floatX=float32  python  predict.py \
 #--save_weights_path=weights/ex1 \
 #--epoch_number=0 \
 #--test_images="data/dataset1/images_prepped_test/" \
 #--output_path="data/predictions/" \
 #--n_classes=10 \
 #--model_name="vgg_segnet" \
 #--input_height=240 \
 #--input_width=320 \
 #--model_name="vgg_segnet"

THEANO_FLAGS=device=gpu,floatX=float32  python  predict.py \
 --save_weights_path=weights/ex1 \
 --epoch_number=19 \
 --test_images="data/dataset1/images_prepped_test/" \
 --output_path="data/predictions/" \
 --n_classes=10 \
 --input_height=320 \
 --input_width=480 \
 --model_name="vgg_segnet" 
