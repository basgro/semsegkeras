#!/bin/bash

#THEANO_FLAGS=device=gpu,floatX=float32  python  train.py \
# --save_weights_path=weights/ex1 \
# --train_images="data/dataset1/images_prepped_train/" \
# --train_annotations="data/dataset1/annotations_prepped_train/" \
# --val_images="data/dataset1/images_prepped_test/" \
# --val_annotations="data/dataset1/annotations_prepped_test/" \
# --n_classes=10 \
# --model_name="vgg_segnet" \
 #--input_height=240 \
 #--input_width=320 \
 #--model_name="vgg_segnet"

THEANO_FLAGS=device=gpu,floatX=float32  python  train.py \
 --save_weights_path=weights/ex1 \
 --train_images="data/dataset1/images_prepped_train/" \
 --train_annotations="data/dataset1/annotations_prepped_train/" \
 --val_images="data/dataset1/images_prepped_test/" \
 --val_annotations="data/dataset1/annotations_prepped_test/" \
 --n_classes=10 \
 --epochs=1 \
 --model_name="vgg_segnet" \
 --input_height=360 \
 --input_width=480 

