#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32  python  train.py \
 --save_weights_path=weights/tinyyolonet7_area1 \
 --train_images="data/stanford/area1/images_prepped_train/" \
 --train_depth="data/stanford/area1/depth_prepped_train/" \
 --train_annotations="data/stanford/area1/annotations_prepped_train/" \
 --val_images="data/stanford/area1/images_prepped_test/" \
 --val_depth="data/stanford/area1/depth_prepped_test/" \
 --val_annotations="data/stanford/area1/annotations_prepped_test/" \
 --n_classes=14 \
 --model_name="tinyyolonet7" \
 --input_height=224 \
 --input_width=320 \
 --epochs=16 \
 --batch_size=2 \
 --load_weights="weights/tinyyolonet7_area1.20"

 	
