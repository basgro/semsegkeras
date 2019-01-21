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

parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type = str, default = 'weights/tinyyolonet7_area6'  )
parser.add_argument("--epoch_number", type = int, default = 62 )
parser.add_argument("--test_images", type = str , default = "/home/basgro/Documents/GraduationProject/DATA/Stanford/2D-3D-Semantics/area_3/data/rgb_rs/")
parser.add_argument("--test_depth", type = str , default = "/home/basgro/Documents/GraduationProject/DATA/Stanford/2D-3D-Semantics/area_3/data/depth_rs/")
parser.add_argument("--output_path", type = str , default = "data/predictions_stanford/")
parser.add_argument("--input_height", type=int , default = 224  )
parser.add_argument("--input_width", type=int , default = 320 )
parser.add_argument("--model_name", type = str , default = "tinyyolonet7")
parser.add_argument("--optimizer_name", type = str , default = "adadelta")
parser.add_argument("--n_classes", type=int, default = 14 )

args = parser.parse_args()

n_classes = args.n_classes
model_name = args.model_name
images_path = args.test_images
depth_path = args.test_depth
input_width =  args.input_width
input_height = args.input_height
epoch_number = args.epoch_number
optimizer_name = args.optimizer_name

modelFns = { 'vgg_segnet':Models.VGGSegnet.VGGSegnet , 'vgg_unet':Models.VGGUnet.VGGUnet , 'vgg_unet2':Models.VGGUnet.VGGUnet2 , 'fcn8':Models.FCN8.FCN8 , 'fcn32':Models.FCN32.FCN32 , 'vgg_segnet_own':Models.VGGSegnetOwn.VGGSegnetOwn   , 'yolonet':Models.YOLONet.YOLONet ,  'tinyyolonet':Models.YOLONet.TinyYOLONet , 'tinyyolonet2':Models.YOLONet.TinyYOLONet2 , 'tinyyolonet3':Models.YOLONet.TinyYOLONet3,  'tinyyolonet4':Models.YOLONet.TinyYOLONet4 ,  'tinyyolonet5':Models.YOLONet.TinyYOLONet5 , 'tinyyolonet7':Models.YOLONet.TinyYOLONet7 ,  'squeezenet':Models.Squeezenet.Squeezenet}
modelFN = modelFns[ model_name ]

m = modelFN( n_classes , input_height=input_height, input_width=input_width   )
print('***************************\n'+ args.save_weights_path + "." + str(  epoch_number )+ '\n***********************************')
m.load_weights(  'weights/tinyyolonet7_area6.51'  )
#optim = tf.keras.optimizers.SGD(0.001, 0.9, 0.005);
optim = tf.keras.optimizers.Adam();
m.compile(loss='categorical_crossentropy',
      optimizer= optimizer_name ,
      metrics=['categorical_accuracy'])



output_height = m.outputHeight
output_width = m.outputWidth

images = glob.glob( images_path + "*.jpg"  ) + glob.glob( images_path + "*.png"  ) +  glob.glob( images_path + "*.jpeg"  )
images.sort()
depth = glob.glob( depth_path + "*.jpg"  ) + glob.glob( depth_path + "*.png"  ) +  glob.glob( depth_path + "*.jpeg"  )
depth.sort()

colors = [  ( random.randint(0,255),random.randint(0,255),random.randint(0,255)   ) for _ in range(n_classes)  ]

#colors = np.array([[0, 0, 0], [0, 0, 255]]) 

for imgName in images:
    outName = imgName.replace( images_path ,  args.output_path )
    depName = depth_path+imgName[len(images_path):len(imgName)-7]+'depth.png'
    X = LoadBatches.getImageArr(imgName , args.input_width  , args.input_height ,odering="channels_last" )
    Y = LoadBatches.getImageArr(depName , args.input_width  , args.input_height, imgNorm = "depth" ,odering="channels_last" )
    start_time=time.time()
    pr = m.predict( [np.array([X]), np.array([Y])] )[0]
   
    pr = pr.reshape(( output_height ,  output_width , n_classes ) ).argmax( axis=2 )
    print("--- %s seconds predicting---" % (time.time() - start_time))
#    seg_img = np.zeros( ( output_height , output_width , 3  ) )
#    for c in range(n_classes):
#        seg_img[:,:,0] += ( (pr[:,: ] == c )*( colors[c][0] )).astype('uint8')
#        seg_img[:,:,1] += ((pr[:,: ] == c )*( colors[c][1] )).astype('uint8')
#        seg_img[:,:,2] += ((pr[:,: ] == c )*( colors[c][2] )).astype('uint8')
#    seg_img = cv2.resize(seg_img  , (input_width , input_height ))
#    print("--- %s seconds total---" % (time.time() - start_time))
#    cv2.imwrite(  outName , seg_img )
    cv2.imwrite(  outName , pr )

