from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.utils import *


import os
file_path = os.path.dirname( os.path.abspath(__file__) )
VGG_Weights_path = file_path+"/../data/vgg16_weights_th_dim_ordering_th_kernels.h5"

def def TinyDarknet( n_classes ,  input_height=416, input_width=608 , vgg_level=3):

    img_input = Input(shape=(3,input_height,input_width))
    # block 1
    x = Conv2D(16,   (3, 3), activation='relu', padding='same', name='conv0', data_format='channels_first' )(img_input)
    x = MaxPooling2D(2, strides=(2, 2), name='pool1', data_format='channels_first' )(x)
    x = Conv2D(32,   (3, 3), activation='relu', padding='same', name='conv1', data_format='channels_first' )(x)    
    x = MaxPooling2D(2, strides=(2, 2), name='pool2', data_format='channels_first' )(x)
    # block 2
    x = Conv2D(16,   (1, 1), activation='relu', padding='same', name='conv2',  data_format='channels_first' )(x)	
    x = Conv2D(128,  (3, 3), activation='relu', padding='same', name='conv3',  data_format='channels_first' )(x)
    x = Conv2D(16,   (1, 1), activation='relu', padding='same', name='conv4',  data_format='channels_first' )(x)
    x = Conv2D(128,  (3, 3), activation='relu', padding='same', name='conv5',  data_format='channels_first' )(x)
    x = MaxPooling2D(2, strides=(2, 2), name='pool3', data_format='channels_first' )(x)
    # block 3
    x = Conv2D(32,  (1, 1), activation='relu', padding='same', name='conv6',  data_format='channels_first' )(x)
    x = Conv2D(256,  (3, 3), activation='relu', padding='same', name='conv7',  data_format='channels_first' )(x)
    x = Conv2D(32,  (1, 1), activation='relu', padding='same', name='conv8',  data_format='channels_first' )(x)
    x = Conv2D(256,  (3, 3), activation='relu', padding='same', name='conv9',  data_format='channels_first' )(x)
    x = MaxPooling2D(2, strides=(2, 2), name='pool4', data_format='channels_first' )(x)
    # block 4
    x = Conv2D(64,  (1, 1), activation='relu', padding='same', name='conv10', data_format='channels_first' )(x)
    x = Conv2D(512,  (3, 3), activation='relu', padding='same', name='conv11', data_format='channels_first' )(x)
    x = Conv2D(64,  (1, 1), activation='relu', padding='same', name='conv12', data_format='channels_first' )(x)
    x = Conv2D(512,  (3, 3), activation='relu', padding='same', name='conv13', data_format='channels_first' )(x)
    x = MaxPooling2D(2, strides=(2, 2), name='pool5', data_format='channels_first' )(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv14', data_format='channels_first' )(x)
    x = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv15', data_format='channels_first' )(x)
    return x


def Encoder (x, model_name)
    modelFns = { 'vgg_segnet':Models.VGGSegnet.VGGSegnet , 'vgg_unet':Models.VGGUnet.VGGUnet , 'vgg_unet2':Models.VGGUnet.VGGUnet2 , 'fcn8':Models.FCN8.FCN8 , 'fcn32':Models.FCN32.FCN32 , 'vgg_segnet_own':Models.VGGSegnetOwn.VGGSegnetOwn , 'unet':Models.Unet.Unet , 'yolonet':Models.YOLONet.YOLONet , 'tinyyolonet':Models.YOLONet.TinyYOLONet , 'tinyyolonet2':Models.YOLONet.TinyYOLONet2 , 'squeezenet':Models.Squeezenet.Squeezenet}
    modelFN = modelFns[ model_name ]
    m = modelFN( n_classes , input_height=input_height, input_width=input_width   )
    return m

def Decoder (x)
	o = UpSampling2D( (2,2), data_format='channels_first')(o)
    o = concatenate([ o ,f3],axis=1 )
    o = Conv2D(N, (3,3), padding='same', data_format='channels_first'))(o)
    o = Conv2D(N, (3,3), padding='same', data_format='channels_first'))(o)	    
    o = Activation('ReLu')(o)

    return m



def Dynamic_Unet( n_classes ,  input_height=416, input_width=608 , vgg_level=3):

    img_input = Input(shape=(3,input_height,input_width))

    m = Encoder(1, 'tinydarknet')
	m = Decoder(m)

    model = Model( img_input , o )
    model.outputWidth = outputWidth
    model.outputHeight = outputHeight
    model.summary()
    plot_model(model, to_file='DUnet-darknet.png')

    return model


if __name__ == '__main__':
    m = Squeezenet( 101 )
    from keras.utils import plot_model
    plot_model( m , show_shapes=True , to_file='model.png')

