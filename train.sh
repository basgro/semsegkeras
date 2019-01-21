#!/bin/bash

THEANO_FLAGS=device=gpu,floatX=float32  python  train.py \
 --save_weights_path=weights/tinyyolonet2 \
 --train_images="data/cmu_corridor_dataset/images_prepped_train_resized/" \
 --train_annotations="data/cmu_corridor_dataset/annotations_prepped_train_resized/" \
 --val_images="data/cmu_corridor_dataset/images_prepped_test_resized/" \
 --val_annotations="data/cmu_corridor_dataset/annotations_prepped_test_resized/" \
 --n_classes=2 \
 --model_name="tinyyolonet2" \
 --input_height=224 \
 --input_width=320 \
 --epochs=5 \
 --batch_size=2
 #--model_name="vgg_segnet"
