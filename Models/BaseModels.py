#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 14:26:21 2018

@author: basgro
"""






from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.utils import *

def EnetInit(img_input, input_name):
    x = img_input
    y = Conv2D(13, (3, 3), strides=2, padding='same', data_format='channels_first', name = input_name )(x)
    z = MaxPooling2D((3, 3), 2, data_format='channels_first' )(x)
    return concatenate([ y, z],axis=1 )

def EnetBottleneckModule(x, type):
    #left baranch
    l = MaxPooling2D((3, 3), 2, data_format='channels_first' )(x)
    l = ZeroPadding2D((1,1)  , data_format='channels_first' )(o)
    #right branch
    r = Conv2D(13, (1, 1), strides=2, padding='same', data_format='channels_first', name = input_name )(x)
    r = BatchNormalization()(r)
    r = Activation('relu')(o)
    r = Conv2D(13, (3, 3), strides=2, padding='same', data_format='channels_first', name = input_name )(x)
    r = BatchNormalization()(r)
    r = Activation('relu')(o)
    r = Conv2D(13, (1, 1), strides=2, padding='same', data_format='channels_first', name = input_name )(x)
    
    r = Conv2D(13, (3, 3), strides=2, padding='same', data_format='channels_first', name = input_name )(x)
    
    return concatenate([ l, r],axis=1 )

def BaseEncoder (input_img, n_classes, input_name):
    
    return x


def BaseDecoder (o, n_classes):
    
    return o



def basicNet( n_classes ,  input_height=416, input_width=608 , vgg_level=3):
    img_input = Input(shape=(3,input_height,input_width))
    depth_input = Input(shape=(1,input_height,input_width))
    x = BaseEncoder(img_input, 'image_input')
    
    o = x
    o = BaseDecoder(o,n_classes)
    
    o_shape = Model(inputs=[img_input, depth_input] , outputs = o ).output_shape
    outputHeight = o_shape[2]
    outputWidth = o_shape[3]

    o = (Reshape((  -1  , outputHeight*outputWidth   )))(o)
    o = (Permute((2, 1)))(o)
    o = (Activation('softmax'))(o)
    model = Model( inputs=[img_input, depth_input] , outputs = o )
    model.outputWidth = outputWidth
    model.outputHeight = outputHeight
    plot_model(model, to_file='tiny_yolonet4.png')

    return model

if __name__ == '__main__':
    m = YOLONet( 101 )
    from keras.utils import plot_model
    plot_model( m , show_shapes=True , to_file='model.png')
    
    

