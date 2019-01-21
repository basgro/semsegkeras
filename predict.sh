#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32  python  predict.py \
 --save_weights_path=weights/tinyyolonet7_area6 \
 --epoch_number=51 \
 --test_images="/home/basgro/Documents/GraduationProject/code/keras/image-segmentation-keras/data/stanford/area6/images_prepped_test/" \
 --test_depth="/home/basgro/Documents/GraduationProject/code/keras/image-segmentation-keras/data/stanford/area6/depth_prepped_test/" \
 --output_path="data/stanford/area6/predictions/" \
 --n_classes=14 \
 --model_name="tinyyolonet7" \
 --input_height=224 \
 --input_width=320 \
 #--model_name="vgg_segnet"
