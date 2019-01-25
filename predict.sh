#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32  python  predict.py \
 --save_weights_path="weights/darknet19_rgbd_r2" \
 --epoch_number=90 \
 --test_images="/home/basgro/Documents/GraduationProject/code/keras/image-segmentation-keras/data/stanford/area1/images_prepped_test/" \
 --test_depth="/home/basgro/Documents/GraduationProject/code/keras/image-segmentation-keras/data/stanford/area1/depth_prepped_test/" \
 --output_path="data/stanford/area1/predictions/RGBD_dn19/" \
 --n_classes=14 \
 --model_name="darknet19" \
 --input_height=224 \
 --input_width=320 \
 #--model_name="vgg_segnet"
