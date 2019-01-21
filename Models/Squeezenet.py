from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.utils import *


import os
file_path = os.path.dirname( os.path.abspath(__file__) )
VGG_Weights_path = file_path+"/../data/vgg16_weights_th_dim_ordering_th_kernels.h5"

def FireModule (x, s1x1, e1x1, e3x3, idx,branch):
    # squeeze
    x = Conv2D(s1x1,   (1, 1), name="fire"+str(idx)+"_squeeze"+branch, padding='same', data_format='channels_last' )(x)
    # expand 
    y = Conv2D(e1x1,   (1, 1), name="fire"+str(idx)+"_expand_1x1"+branch, padding='same', data_format='channels_last' )(x)
    z = Conv2D(e3x3,   (3, 3), name="fire"+str(idx)+"_expand_3x3"+branch, padding='same', data_format='channels_last' )(x)
    o = ( concatenate([ y ,z],axis=3, name="fire"+str(idx)+"_concatenate"+branch )  )
    return o

def Squeezenet( n_classes ,  input_height=416, input_width=608 , vgg_level=3):

    img_input = Input(shape=(3,input_height,input_width))

    x = Conv2D(96, (7, 7), strides=2, padding='same', data_format='channels_last' )(img_input)
    x = MaxPooling2D((3, 3), 2, data_format='channels_last' )(x)
    fire2 = FireModule(x, 16, 64, 64, 2)
    fire3 = FireModule(fire2, 16, 64, 64, 3)
    fire4 = FireModule(add([fire2, fire3]), 32, 128, 128, 4)     
    pool2 = MaxPooling2D(pool_size=(3, 3), strides=2, data_format='channels_last' )(fire4)  
    fire5 = FireModule(pool2, 32, 128, 128, 5)
    fire6 = FireModule(add([pool2, fire5]), 48, 192, 192, 6)
    fire7 = FireModule(fire6, 48, 192, 192, 7)
    fire8 = FireModule(add([fire6, fire7]), 64, 256, 256, 8)
    pool3 = MaxPooling2D(pool_size=(3, 3), strides=2, data_format='channels_last' )(fire8)  
    fire9 = FireModule(pool3, 64, 256, 256, 9)
    fire9 = add([pool3, fire9])
    #x = Conv2D(96, (13, 13), activation='relu', padding='same', data_format='channels_first' )(x)
	
    o = fire9
	
    o = ( ZeroPadding2D( (1,1) , data_format='channels_last' ))(o)
    o = ( Conv2D(512, (3, 3), padding='valid', data_format='channels_last'))(o)
    o = ( BatchNormalization())(o)

    o = ( UpSampling2D( (2,2), data_format='channels_last'))(o)
    #o = ( concatenate([ o ,f3],axis=1 )  )	
    o = ( ZeroPadding2D( (1,1), data_format='channels_last'))(o)
    o = ( Conv2D( 256, (3, 3), padding='valid', data_format='channels_last'))(o)
    o = ( BatchNormalization())(o)

    o = ( UpSampling2D((2,2)  , data_format='channels_last' ) )(o)
    #o = ( concatenate([ o ,f2],axis=1 )  )
    o = ( ZeroPadding2D((1,1) , data_format='channels_last' ))(o)
    o = ( Conv2D( 128 , (3, 3), padding='valid' , data_format='channels_last' ))(o)
    o = ( BatchNormalization())(o)

    o = ( UpSampling2D((2,2)  , data_format='channels_last' ))(o)
    #o = ( concatenate([ o ,f1],axis=1 )  )
    o = ( ZeroPadding2D((1,1)  , data_format='channels_last' ))(o)
    o = ( Conv2D( 64 , (3, 3), padding='valid'  , data_format='channels_last' ))(o)
    o = ( BatchNormalization())(o)


    o =  Conv2D( n_classes , (3, 3) , padding='same', data_format='channels_last' )( o )
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

def SqueezenetEncoder( input_img , input_name, branch):
    # block 1
    x = Conv2D(96, (7, 7), strides=2, padding='same', data_format='channels_last', name = input_name )(input_img)
    x = MaxPooling2D((3, 3), 2, data_format='channels_last' )(x)
    fire2 = FireModule(x, 16, 64, 64, 2, branch)
    fire3 = FireModule(fire2, 16, 64, 64, 3, branch)
    fire4 = FireModule(add([fire2, fire3]), 32, 128, 128, 4, branch)     
    pool2 = MaxPooling2D(pool_size=(3, 3), strides=2, data_format='channels_last' )(fire4)  
    fire5 = FireModule(pool2, 32, 128, 128, 5, branch)
    fire6 = FireModule(add([pool2, fire5]), 48, 192, 192, 6, branch)
    fire7 = FireModule(fire6, 48, 192, 192, 7, branch)
    fire8 = FireModule(add([fire6, fire7]), 64, 256, 256, 8, branch)
    pool3 = MaxPooling2D(pool_size=(3, 3), strides=2, data_format='channels_last' )(fire8)  
    fire9 = FireModule(pool3, 64, 256, 256, 9, branch)
    fire9 = add([pool3, fire9])

    return (fire9)      

def SegnetDecoder (o, n_classes):
    o = ( ZeroPadding2D( (1,1) , data_format='channels_last' ))(o)
    o = ( Conv2D(512, (3, 3), padding='valid', data_format='channels_last'))(o)
    o = ( BatchNormalization())(o)

    o = ( UpSampling2D( (2,2), data_format='channels_last'))(o)
    #o = ( concatenate([ o ,f3],axis=1 )  )	
    o = ( ZeroPadding2D( (1,1), data_format='channels_last'))(o)
    o = ( Conv2D( 256, (3, 3), padding='valid', data_format='channels_last'))(o)
    o = ( BatchNormalization())(o)

    o = ( UpSampling2D((2,2)  , data_format='channels_last' ) )(o)
    #o = ( concatenate([ o ,f2],axis=1 )  )
    o = ( ZeroPadding2D((1,1) , data_format='channels_last' ))(o)
    o = ( Conv2D( 128 , (3, 3), padding='valid' , data_format='channels_last' ))(o)
    o = ( BatchNormalization())(o)

    o = ( UpSampling2D((2,2)  , data_format='channels_last' ))(o)
    #o = ( concatenate([ o ,f1],axis=1 )  )
    o = ( ZeroPadding2D((1,1)  , data_format='channels_last' ))(o)
    o = ( Conv2D( 64 , (3, 3), padding='valid'  , data_format='channels_last' ))(o)
    o = ( BatchNormalization())(o)


    o =  Conv2D( n_classes , (3, 3) , padding='same', data_format='channels_last' )( o )
    return (o)


def Squeezenet2( n_classes ,  input_height=416, input_width=608 , vgg_level=3):
    depth = False
    img_input = Input(shape=(3,input_height,input_width))
    depth_input = Input(shape=(1,input_height,input_width))
    
    x = SqueezenetEncoder(img_input, 'image_input')
    
#    if depth:
#        (xd, fd1, fd2, fd3) = TinyDarknetEncoder(depth_input, 'depth_input')
#        x = ( concatenate([ x ,xd],axis=1 )  )	
    
    o = x
    o = SegnetDecoder(o,n_classes)
    
    o_shape = Model(inputs=[img_input, depth_input] , outputs = o ).output_shape
    outputHeight = o_shape[2]
    outputWidth = o_shape[3]

    o = (Reshape((  -1  , outputHeight*outputWidth   )))(o)
    o = (Permute((2, 1)))(o)
    o = (Activation('softmax'))(o)
    model = Model( inputs=[img_input, depth_input] , outputs = o )
    model.outputWidth = outputWidth
    model.outputHeight = outputHeight
    plot_model(model, to_file='squeezenet2.png')

    return model

def Squeezenet3( n_classes ,  input_height=416, input_width=608 , vgg_level=3):
    depth = 5
    img_input = Input(shape=(input_height,input_width,3))
    depth_input = Input(shape=(input_height,input_width,1))
    
    x = SqueezenetEncoder(img_input, 'image_input', 'rgb')
    xd = SqueezenetEncoder(depth_input, 'depth_input', 'depth')
    
    o = x
    od = xd    
    o = ( Add()([ o ,od] ))
    o = SegnetDecoder(o,n_classes) 
    o = ( BatchNormalization())(o)
    o = ( Conv2D( n_classes , (1, 1), padding='valid'  , data_format='channels_last' ))(o)
            
    o_shape = Model(inputs=[img_input, depth_input] , outputs = o ).output_shape
    outputHeight = o_shape[1]
    outputWidth = o_shape[2]

    o = (Reshape(( outputHeight*outputWidth ,  n_classes    )))(o)
    o = (Activation('softmax'))(o)
    model = Model( inputs=[img_input, depth_input] , outputs = o )
 #   model = Model( inputs=img_input , outputs = o )
    model.outputWidth = outputWidth
    model.outputHeight = outputHeight
#    plot_model(model, to_file='tiny_yolonet7.png')

    return model


if __name__ == '__main__':
    m = Squeezenet( 101 )
    from keras.utils import plot_model
    plot_model( m , show_shapes=True , to_file='model.png')

