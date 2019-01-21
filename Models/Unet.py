

# todo upgrade to keras 2.0

from tensorflow.keras.models import *
from tensorflow.keras.layers import *

#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Reshape
#from tensorflow.keras.models import Model
#from tensorflow.keras.layers import Layer, Dense, Dropout, Activation, Flatten, Reshape, Merge, Permute
#from tensorflow.keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout
#from tensorflow.keras.layers.normalization import BatchNormalization
#from tensorflow.keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D , ZeroPadding3D , UpSampling3D
#from tensorflow.keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
#from tensorflow.keras.layers.convolutional import Convolution1D, MaxPooling1D
#from tensorflow.keras.layers.recurrent import LSTM
#from tensorflow.keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.optimizers import Adam , SGD
#from tensorflow.keras.layers.embeddings import Embedding
#from tensorflow.keras.utils import np_utils
from tensorflow.keras import regularizers
#from tensorflow.keras.regularizers import ActivityRegularizer
from tensorflow.keras import backend as K





def Unet (nClasses , optimizer=None , input_width=360 , input_height=480 , nChannels=3 ): 
    
    inputs = Input((nChannels, input_height, input_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Convolution2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2), data_format='channels_first')(conv1)
    
    conv2 = Convolution2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Convolution2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2), data_format='channels_first')(conv2)
    
    conv3 = Convolution2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Convolution2D(128, (3, 3), activation='relu', padding='same')(conv3)

    #up1 = merge([UpSampling2D(size=(2, 2))(conv3), conv2], mode='concat', concat_axis=1)
    up1s = concatenate([UpSampling2D(size=(2, 2), data_format='channels_first')(conv3), conv2])
    conv4 = Convolution2D(64, (3, 3), activation='relu', padding='same')(up1s)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Convolution2D(64, (3, 3), activation='relu', padding='same')(conv4)
    
    #up2 = merge([UpSampling2D(size=(2, 2))(conv4), conv1], mode='concat', concat_axis=1)
    up2 = concatenate([UpSampling2D(size=(2, 2), data_format='channels_first')(conv4), conv1])
    conv5 = Convolution2D(32, (3, 3), activation='relu', padding='same')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Convolution2D(32, (3, 3), activation='relu', padding='same')(conv5)
    
    conv6 = Convolution2D(nClasses, (1, 1), activation='relu',padding='same')(conv5)
    #conv6 = Reshape((nClasses,input_height*input_width))(conv6)
    #conv6 = Permute((2,1))(conv6)


    conv7 = Activation('softmax')(conv6)

    model = Model(inputs, conv7)
    #model.outputWidth = 320
    #model.outputHeight = 224

    if not optimizer is None:
	    model.compile(loss="categorical_crossentropy", optimizer= optimizer , metrics=['accuracy'] )
    model.summary()
    return model
	
	
	

