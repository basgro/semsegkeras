#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 16:30:34 2019

@author: basgro
"""
from __future__ import division

import argparse
import Models , LoadBatches
from tensorflow.keras.models import load_model
import tensorflow as tf
import glob
import cv2
import numpy as np
import random
import time
import tensorflow.keras.backend as K

import numpy as np
import cv2
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9

def exitt():
    cap.release()
    cv2.destroyAllWindows()
    exit()
    return

def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def  dice2(y_true, y_pred):
    # Symbolically compute the intersection
    y_int = y_true*y_pred
    # Technically this is the negative of the Sorensen-Dice index. This is done for
    # minimization purposes
    return -(2*K.sum(y_int) / (K.sum(y_true) + K.sum(y_pred)))

def mIOU(y_true, y_pred):
    shp = y_pred.shape
    IOUc = np.zeros((14));
    for i in range(14):
        pred = (y_pred==i)
        true = (y_true==i)
        intersection = pred*true
        union = pred+true
        if np.sum(intersection) > 0:
            IOUc[i] = float(np.sum(intersection))/float(np.sum(union))
#    print(np.round(IOUc,2))
    return np.mean(IOUc[np.nonzero(IOUc)])


def getImageArr( img_in , width , height , imgNorm="sub_mean" , odering='channels_first' ):
    try:
        img = img_in
#        img = cv2.imread("/home/basgro/Documents/GraduationProject/code/keras/image-segmentation-keras/data/stanford/annotations_reclassed_test/camera_0a70cd8d4f2b48239aaa5db59719158a_office_12_frame_4_domain_semantic.png", 1)


        if imgNorm == "sub_and_divide":
            img = np.float32(cv2.resize(img, ( width , height ))) / 127.5 - 1
        elif imgNorm == "sub_mean":
            img = cv2.resize(img, ( width , height ))
            img = img.astype(np.float32)
            img[:,:,0] -= 103.939
            img[:,:,1] -= 116.779
            img[:,:,2] -= 123.68
            img = img/255.0

        elif imgNorm == "divide":
            img = cv2.resize(img, ( width , height ))
            img = img.astype(np.float32)
            img = img/255.0
            
        elif imgNorm == "depth":
            img_depth = img[:,:]
            img_depth = np.float32(cv2.resize(img_depth, ( width , height ))) / 32768 - 1
            img_depth = np.expand_dims(img_depth,2)
            if odering == 'channels_first':
                img_depth = np.rollaxis(img_depth, 2, 0)
            return img_depth
            

        if odering == 'channels_first':
            img = np.rollaxis(img, 2, 0)
        return img
    except Exception, e:
        img = np.zeros((  height , width  , 3 ))
        if odering == 'channels_first':
            img = np.rollaxis(img, 2, 0)
        return img


#base_dir="/home/basgro/Documents/GraduationProject/code/keras/image-segmentation-keras/data/stanford/"
#
#images_test_dir_rs="images_prepped_train/"
#annotations_test_dir="annotations_prepped_train/"



parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type = str, default = "weights/tinydarknet7_shortcuts_rgbd"  )
parser.add_argument("--epoch_number", type = int, default = 12 )
parser.add_argument("--test_images", type = str , default = "")
parser.add_argument("--test_depth", type = str , default = "")
parser.add_argument("--output_path", type = str , default = "")
parser.add_argument("--input_height", type=int , default = 224  )
parser.add_argument("--input_width", type=int , default = 320 )
parser.add_argument("--model_name", type = str , default = "tinyyolonet7")
parser.add_argument("--optimizer_name", type = str , default = "adadelta")
parser.add_argument("--n_classes", type=int, default = 7 )

args = parser.parse_args()

n_classes = args.n_classes
model_name = args.model_name
images_path = args.test_images
depth_path = args.test_depth
input_width =  args.input_width
input_height = args.input_height
epoch_number = args.epoch_number
optimizer_name = args.optimizer_name

modelFns = { 'vgg_segnet':Models.VGGSegnet.VGGSegnet , 'vgg_unet':Models.VGGUnet.VGGUnet , 'vgg_unet2':Models.VGGUnet.VGGUnet2 , 'fcn8':Models.FCN8.FCN8 , 'fcn32':Models.FCN32.FCN32 , 'vgg_segnet_own':Models.VGGSegnetOwn.VGGSegnetOwn   , 'yolonet':Models.YOLONet.YOLONet ,  'tinyyolonet':Models.YOLONet.TinyYOLONet , 'tinyyolonet2':Models.YOLONet.TinyYOLONet2 , 'tinyyolonet3':Models.YOLONet.TinyYOLONet3,  'tinyyolonet4':Models.YOLONet.TinyYOLONet4 ,  'tinyyolonet5':Models.YOLONet.TinyYOLONet5, 'tinyyolonet6':Models.YOLONet.TinyYOLONet6 , 'tinyyolonet7':Models.YOLONet.TinyYOLONet7,  'squeezenet':Models.Squeezenet.Squeezenet}
modelFN = modelFns[ model_name ]

m = modelFN( n_classes , input_height=input_height, input_width=input_width   )
print(args.save_weights_path + "." + str(  epoch_number ))
m.load_weights(  args.save_weights_path + "." + str(  epoch_number )  )
#optim = tf.keras.optimizers.SGD(0.001, 0.9, 0.005);
m.compile(loss='categorical_crossentropy',
      optimizer= optimizer_name ,
      metrics=['categorical_accuracy'])

m.summary()

output_height = m.outputHeight
output_width = m.outputWidth

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
if n_classes==12:
    classes = ['','floor', 'ceiling', 'wall', 'bookcase', 'door', 'window', 'clutter', 'beam', 'table', 'chair', 'board']


cap = cv2.VideoCapture(0)

t = glob.glob('/home/basgro/Documents/GraduationProject/code/keras/image-segmentation-keras/data/stanford/images_prepped_test/*.png')
idx = 0;

while(idx<len(t)):
    
    frame = cv2.imread(t[idx],-1)
    depth = cv2.imread(t[idx][:91]+'depth_prepped_test/'+t[idx][111:len(t[idx])-7]+'depth.png',-1)
    semantic = cv2.imread(t[idx][:91]+'annotations_reclassed_test/'+t[idx][111:len(t[idx])-7]+'semantic.png',0)
    X = getImageArr(frame , args.input_width  , args.input_height, odering='channels_last'  )
    Y = getImageArr(depth , args.input_width  , args.input_height, imgNorm="depth", odering='channels_last'  )
    
#    cv2.imshow('ORIGINAL',frame)
    start_time=time.time()
    pr = m.predict( [np.array([X]), np.array([Y])] )[0]
    print("--- %s seconds predicting---" % (time.time() - start_time))
    pr_temp = pr
    pr = pr.reshape(( output_height ,  output_width , n_classes ) ).argmax( axis=2 )
    seg_img = np.zeros( ( output_height , output_width , 3  ) )
    for c in range(n_classes):
        seg_img[:,:,0] += ( (pr[:,: ] == c )*( colors[c][0] )).astype('uint8')
        seg_img[:,:,1] += ((pr[:,: ] == c )*( colors[c][1] )).astype('uint8')
        seg_img[:,:,2] += ((pr[:,: ] == c )*( colors[c][2] )).astype('uint8')
    seg_img = cv2.resize(seg_img  , (input_width , input_height ))
    
    gt = np.zeros( ( output_height , output_width , 3  ) )
    for c in range(n_classes):
        gt[:,:,0] += ( (semantic[:,: ] == c )*( colors[c][0] )).astype('uint8')
        gt[:,:,1] += ((semantic[:,: ] == c )*( colors[c][1] )).astype('uint8')
        gt[:,:,2] += ((semantic[:,: ] == c )*( colors[c][2] )).astype('uint8')
    gt = cv2.resize(gt  , (input_width , input_height ))
    
#    both = cv2.addWeighted( frame, 1,np.uint8(seg_img) , 0.4, 0.0);
    
    both = np.hstack((frame, np.uint8(seg_img), np.uint8(gt) ))
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    height, width, channels = both.shape
    res = np.zeros((height+60, width+200, 3), dtype=np.uint8)
    x_offset = 60  
    y_offset = 0  
    
    res[x_offset:both.shape[0]+x_offset,y_offset:both.shape[1]+y_offset,0:3] = both
    cv2.putText(res,t[idx], (10,16) , font, 0.5,(255,255,255),1,cv2.LINE_AA) 
    cv2.putText(res,'Original/predicted/GT', (10,32) , font, 0.5,(255,255,255),1,cv2.LINE_AA) 
    c = 0;
#    while c<n_classes-1:
    for c in range(n_classes):
        res[61+c*17:60+(c+1)*17,width+1:width+1+17,0]=colors[c][0]
        res[61+c*17:60+(c+1)*17,width+1:width+1+17,1]=colors[c][1]
        res[61+c*17:60+(c+1)*17,width+1:width+1+17,2]=colors[c][2]
        cv2.putText(res,classes[c], (width+1+18,57+(c+1)*17) , font, 0.5,(255,255,255),1,cv2.LINE_AA) 
        c = c + 1
    
    cv2.imshow('frame',res)
    print("--- %s seconds total---" % (time.time() - start_time))
    print("mIOU: " + str(mIOU(semantic.reshape((224*320)),pr_temp.argmax(1))))
    
    
    while True: 
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            exitt()
        if key == ord('n'):
            break
        
    
    idx=idx+1


