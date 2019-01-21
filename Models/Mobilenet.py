from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.utils import *


import os
file_path = os.path.dirname( os.path.abspath(__file__) )
VGG_Weights_path = file_path+"/../data/vgg16_weights_th_dim_ordering_th_kernels.h5"

def Conv_dw (x, n_filters):
    x = Conv2D(n_filters, (3, 3), padding='same', data_format='channels_first' )(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = Conv2D(n_filters, (1, 1), padding='same', data_format='channels_first' )(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    o=x
    return o

def Conv (x, n_filters):
    x = Conv2D(n_filters, (3, 3), padding='same', data_format='channels_first' )(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    o=x
    return o

def Mobilenet( n_classes ,  input_height=416, input_width=608 , vgg_level=3):

    img_input = Input(shape=(3,input_height,input_width))

    x = Conv(img_input, 32)
    x = Conv_dw (x, 32)
    x = Conv(x, 64)
    x = Conv_dw (x, 64)
    x = Conv(x, 128)
    x = Conv_dw (x, 128)
    
    
        
    
    

    
    o = fire9
	
    o = ( ZeroPadding2D( (1,1) , data_format='channels_first' ))(o)
    o = ( Conv2D(512, (3, 3), padding='valid', data_format='channels_first'))(o)
    o = ( BatchNormalization())(o)

    o = ( UpSampling2D( (2,2), data_format='channels_first'))(o)
    #o = ( concatenate([ o ,f3],axis=1 )  )	
    o = ( ZeroPadding2D( (1,1), data_format='channels_first'))(o)
    o = ( Conv2D( 256, (3, 3), padding='valid', data_format='channels_first'))(o)
    o = ( BatchNormalization())(o)

    o = ( UpSampling2D((2,2)  , data_format='channels_first' ) )(o)
    #o = ( concatenate([ o ,f2],axis=1 )  )
    o = ( ZeroPadding2D((1,1) , data_format='channels_first' ))(o)
    o = ( Conv2D( 128 , (3, 3), padding='valid' , data_format='channels_first' ))(o)
    o = ( BatchNormalization())(o)

    o = ( UpSampling2D((2,2)  , data_format='channels_first' ))(o)
    #o = ( concatenate([ o ,f1],axis=1 )  )
    o = ( ZeroPadding2D((1,1)  , data_format='channels_first' ))(o)
    o = ( Conv2D( 64 , (3, 3), padding='valid'  , data_format='channels_first' ))(o)
    o = ( BatchNormalization())(o)


    o =  Conv2D( n_classes , (3, 3) , padding='same', data_format='channels_first' )( o )
    o_shape = Model(img_input , o ).output_shape
    outputHeight = o_shape[2]
    outputWidth = o_shape[3]

    o = (Reshape((  -1  , outputHeight*outputWidth   )))(o)
    o = (Permute((2, 1)))(o)
    o = (Activation('softmax'))(o)
    model = Model( img_input , o )
    model.outputWidth = outputWidth
    model.outputHeight = outputHeight
    model.summary()
    plot_model(model, to_file='squeezenet.png')

    return model


if __name__ == '__main__':
    m = Mobilenet( 101 )
    from keras.utils import plot_model
    plot_model( m , show_shapes=True , to_file='model.png')

