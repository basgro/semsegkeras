#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32  python  train.py \
 --save_weights_path="weights/darknet19_rgbd_r2" \
 --train_images="data/stanford/images_prepped_train/" \
 --train_depth="data/stanford/depth_prepped_train/" \
 --train_annotations="data/stanford/annotations_prepped_train/" \
 --val_images="data/stanford/images_prepped_test/" \
 --val_depth="data/stanford/depth_prepped_test/" \
 --val_annotations="data/stanford/annotations_prepped_test/" \
 --n_classes=14 \
 --model_name="darknet19" \
 --input_height=224 \
 --input_width=320 \
 --epochs=90 \
 --batch_size=2 \
 --load_weights="weights/darknet19_rgbd_r2.23"

 	
