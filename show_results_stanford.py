#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 16:27:03 2018

@author: basgro
"""

from __future__ import division
import cv2
import numpy as np
import argparse
import glob, os
import random

base_dir="/home/basgro/Documents/GraduationProject/code/keras/image-segmentation-keras/data/stanford/area1/"

images_test_dir_rs="images_prepped_test/"
annotations_test_dir="annotations_prepped_test/"
prediction_dir = "predictions/RGBD_dn19/"
n_classes=14
#colors = [[  255.0000,  255.0000,  255.0000],
#  [255.0000,   63.7500,         0],
#  [255.0000,  127.5000,         0],
#  [255.0000,  191.2500,         0],
#  [255.0000,  255.0000,         0],
#  [191.2500,  255.0000,   63.7500],
#  [127.5000,  255.0000,  127.5000],
#  [ 63.7500,  255.0000,  191.2500],
#  [       0,  255.0000,  255.0000],
#  [       0,  191.2500,  255.0000],
#  [       0,  127.5000,  255.0000],
#  [       0,   63.7500,  255.0000],
#  [       0,         0,  255.0000]]


colors = [[0,0,0],
          [0,         0,  255.0000],
         [0,   63.7500,  255.0000],
         [0,  127.5000,  255.0000],
         [0,  191.2500,  255.0000],
         [0,  255.0000,  255.0000],
   [63.7500,  255.0000,  191.2500],
  [127.5000,  255.0000,  127.5000],
  [191.2500,  255.0000,   63.7500],
  [255.0000,  255.0000,         0],
  [255.0000,  191.2500,         0],
  [255.0000,  127.5000,         0],
  [255.0000,   63.7500,         0],
  [255.0000,         0,         0]]

#colors = [[0,0,0],[0,0,1],[1,0,0],[0,1,0],[0,0,0.1724],[1,0.1034,0.7241],[1,0.8276,0],[0,0.3448,0],[0.5172,0.5172,1],[0.6207,0.3103,0.2759],[0,1,0.7586],[0,0.5172,0.5862],[0,0,0.4828],[0.5862,0.8276,0.3103]]
#if n_classes == 12:
#    colors = [[0,0,0],[0,0,255],[255,0,0],[0,255,0],[0,0,43.962],[255,26.367,184.6455],[255,211.038,0],[0,87.924,0],[131.886,131.886,255],[158.2785,79.1265,70.3545],[0,255,193.443],[0,131.886,149.481]];
#    


classes = ['', 'floor', 'ceiling', 'wall', 'bookcase', 'door', 'window', 'clutter', 'column', 'beam', 'table', 'chair', 'board', 'sofa'];

os.chdir(base_dir+prediction_dir)
for file in glob.glob("*.png"):
    img = cv2.imread(base_dir+images_test_dir_rs+file)
    pred = cv2.imread(base_dir+prediction_dir+file)
    gt = cv2.imread(base_dir+annotations_test_dir+file[:-7]+'semantic.png')
    seg_img = np.zeros( ( 224 , 320 , 3  ), dtype='uint8' )
    gt_img= np.zeros( ( 224 , 320 , 3  ), dtype='uint8' )
    for c in range(n_classes):
        seg_img[:,:,0] += ( (pred[:,:,0 ] == c )*( colors[c][0])).astype('uint8')
        seg_img[:,:,1] += ((pred[:,:,1 ] == c )*( colors[c][1])).astype('uint8')
        seg_img[:,:,2] += ((pred[:,:,2 ] == c )*( colors[c][2])).astype('uint8')
        gt_img[:,:,0] += ( (gt[:,:,0 ] == c )*( colors[c][0])).astype('uint8')
        gt_img[:,:,1] += ((gt[:,:,1 ] == c )*( colors[c][1] )).astype('uint8')
        gt_img[:,:,2] += ((gt[:,:,2 ] == c )*( colors[c][2] )).astype('uint8')
#    gt = cv2.imread(base_dir+annotations_test_dir_rs+file)
#    gt[:,:,0] = 0
#    gt[:,:,1] = 0
    merged = cv2.addWeighted(img,1.0,seg_img,0.5,0.0)
#    merged_gt = cv2.addWeighted(img,1.0,gt,0.4,0.0)
    both = np.hstack((img, gt_img, seg_img))
    font = cv2.FONT_HERSHEY_SIMPLEX
    height, width, channels = both.shape
    res = np.zeros((height+60, width+200, 3), dtype=np.uint8)
    x_offset = 60  
    y_offset = 0  
    
    res[x_offset:both.shape[0]+x_offset,y_offset:both.shape[1]+y_offset,0:3] = both
    cv2.putText(res,file, (10,15) , font, 0.5,(255,255,255),1,cv2.LINE_AA) 
    cv2.putText(res,'Original', (10,32) , font, 0.5,(255,255,255),1,cv2.LINE_AA) 
    cv2.putText(res,'Ground truth', (int(width/2),30) , font, 0.5,(255,255,255),1,cv2.LINE_AA) 
    cv2.putText(res,'Predicted', (int(3*width/4),30) , font, 0.5,(255,255,255),1,cv2.LINE_AA) 
    c = 0;
#    while c<n_classes-1:
    blockheight = np.int(np.round(224/n_classes)-1)
    
    for c in range(n_classes):
        res[61+c*blockheight:60+(c+1)*blockheight,width+1:width+1+blockheight,0]=colors[c][0]
        res[61+c*blockheight:60+(c+1)*blockheight,width+1:width+1+blockheight,1]=colors[c][1]
        res[61+c*blockheight:60+(c+1)*blockheight,width+1:width+1+blockheight,2]=colors[c][2]
        cv2.putText(res,classes[c], (width+2+blockheight,57+(c+1)*blockheight) , font, 0.5,(255,255,255),1,cv2.LINE_AA) 
        c = c + 1
    
#    y_pred = pred[:,:,2]
#    y_true = gt[:,:,2]
#    y_int = y_pred*y_true
#    if (cv2.countNonZero(y_true)>0):
#        dice = 2*cv2.countNonZero(y_int)/(cv2.countNonZero(y_true)+cv2.countNonZero(y_pred))
#        cv2.putText(res,'Dice index: '+str(dice), (10,50) , font, 0.5,(255,255,255),1,cv2.LINE_AA)
#        overlap = y_pred*y_true # Logical AND
#        union = y_pred + y_true # Logical OR
#        IOU = cv2.countNonZero(overlap)/cv2.countNonZero(union)
#        cv2.putText(res,'IOU: '+str(IOU), (int(width/2)+32,50) , font, 0.5,(255,255,255),1,cv2.LINE_AA) 
    cv2.imshow('result',res)
    while True:
        key = cv2.waitKey(30)
        if key == ord('n'):
            break
        if key == ord('q'):
            cv2.destroyAllWindows()
            exit()
		
