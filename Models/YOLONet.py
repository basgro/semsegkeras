




from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.utils import *


import os
file_path = os.path.dirname( os.path.abspath(__file__) )
VGG_Weights_path = file_path+"/../data/vgg16_weights_th_dim_ordering_th_kernels.h5"


def YOLONet( n_classes ,  input_height=416, input_width=608 , vgg_level=3):

    img_input = Input(shape=(3,input_height,input_width))
    # block 1
    x = Conv2D(32,   (3, 3), activation='relu', padding='same', name='conv0', data_format='channels_first' )(img_input)
    x = MaxPooling2D(2, strides=(2, 2), name='pool1', data_format='channels_first' )(x)
    x = Conv2D(64,   (3, 3), activation='relu', padding='same', name='conv1', data_format='channels_first' )(x)    
    x = MaxPooling2D(2, strides=(2, 2), name='pool2', data_format='channels_first' )(x)
    f1 = x
    # block 2
    x = Conv2D(128,   (3, 3), activation='relu', padding='same', name='conv2',  data_format='channels_first' )(x)	
    x = Conv2D(64,  (1, 1), activation='relu', padding='same', name='conv3',  data_format='channels_first' )(x)
    x = Conv2D(128,   (3, 3), activation='relu', padding='same', name='conv4',  data_format='channels_first' )(x)
    x = MaxPooling2D(2, strides=(2, 2), name='pool3', data_format='channels_first' )(x)
    f2 = x
    # block 3
    x = Conv2D(256,  (3, 3), activation='relu', padding='same', name='conv5',  data_format='channels_first' )(x)
    x = Conv2D(128,  (1, 1), activation='relu', padding='same', name='conv6',  data_format='channels_first' )(x)
    x = Conv2D(256,  (3, 3), activation='relu', padding='same', name='conv7',  data_format='channels_first' )(x)
    x = MaxPooling2D(2, strides=(2, 2), name='pool4', data_format='channels_first' )(x)
    f3 = x
    # block4
    x = Conv2D(512,  (3, 3), activation='relu', padding='same', name='conv8',  data_format='channels_first' )(x)
    x = Conv2D(256,  (1, 1), activation='relu', padding='same', name='conv9',  data_format='channels_first' )(x)
    x = Conv2D(512,  (3, 3), activation='relu', padding='same', name='conv10', data_format='channels_first' )(x)
    x = Conv2D(256,  (1, 1), activation='relu', padding='same', name='conv11', data_format='channels_first' )(x)
    x = Conv2D(512,  (3, 3), activation='relu', padding='same', name='conv12', data_format='channels_first' )(x)
    x = MaxPooling2D(2, strides=(2, 2), name='pool5', data_format='channels_first' )(x)
    f4 = x
    # block 5
    x = Conv2D(1024,  (3, 3), activation='relu', padding='same', name='conv13',  data_format='channels_first' )(x)
    x = Conv2D(512,  (1, 1), activation='relu', padding='same', name='conv14',  data_format='channels_first' )(x)
    x = Conv2D(1024,  (3, 3), activation='relu', padding='same', name='conv15', data_format='channels_first' )(x)
    x = Conv2D(512,  (1, 1), activation='relu', padding='same', name='conv16', data_format='channels_first' )(x)
    x = Conv2D(1024,  (3, 3), activation='relu', padding='same', name='conv17', data_format='channels_first' )(x)
        
    o = x
	
    o = ( ZeroPadding2D( (1,1) , data_format='channels_first' ))(o)
    o = ( Conv2D(512, (3, 3), padding='valid', data_format='channels_first'))(o)
    o = ( BatchNormalization())(o)

    #o = ( UpSampling2D( (2,2), data_format='channels_first'))(o)
    o = ( concatenate([ o ,f4],axis=1 )  )	
    o = ( ZeroPadding2D( (1,1), data_format='channels_first'))(o)
    o = ( Conv2D( 256, (3, 3), padding='valid', data_format='channels_first'))(o)
    o = ( BatchNormalization())(o)

    o = ( UpSampling2D((2,2)  , data_format='channels_first' ) )(o)
    o = ( concatenate([ o ,f3],axis=1 )  )
    o = ( ZeroPadding2D((1,1) , data_format='channels_first' ))(o)
    o = ( Conv2D( 128 , (3, 3), padding='valid' , data_format='channels_first' ))(o)
    o = ( BatchNormalization())(o)

    o = ( UpSampling2D((2,2)  , data_format='channels_first' ))(o)
    o = ( concatenate([ o ,f2],axis=1 )  )
    o = ( ZeroPadding2D((1,1)  , data_format='channels_first' ))(o)
    o = ( Conv2D( 64 , (3, 3), padding='valid'  , data_format='channels_first' ))(o)
    o = ( BatchNormalization())(o)

    o = ( UpSampling2D((2,2)  , data_format='channels_first' ))(o)
    o = ( concatenate([ o ,f1],axis=1 )  )
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
    plot_model(model, to_file='yolonet.png')
    return model

def TinyYOLONet( n_classes ,  input_height=416, input_width=608 , vgg_level=3):

    img_input = Input(shape=(3,input_height,input_width))
    # block 1
    x = Conv2D(16,   (3, 3), activation='relu', padding='same', name='conv0', data_format='channels_first' )(img_input)
    x = MaxPooling2D(2, strides=(2, 2), name='pool1', data_format='channels_first' )(x)
    x = Conv2D(32,   (3, 3), activation='relu', padding='same', name='conv1', data_format='channels_first' )(x)    
    x = MaxPooling2D(2, strides=(2, 2), name='pool2', data_format='channels_first' )(x)
    f1 = x
    # block 2
    x = Conv2D(16,   (1, 1), activation='relu', padding='same', name='conv2',  data_format='channels_first' )(x)	
    x = Conv2D(128,  (3, 3), activation='relu', padding='same', name='conv3',  data_format='channels_first' )(x)
    x = Conv2D(16,   (1, 1), activation='relu', padding='same', name='conv4',  data_format='channels_first' )(x)
    x = Conv2D(128,  (3, 3), activation='relu', padding='same', name='conv5',  data_format='channels_first' )(x)
    x = ( concatenate([ x ,f1],axis=1 )  )
    x = MaxPooling2D(2, strides=(2, 2), name='pool3', data_format='channels_first' )(x)
    f2 = x
    # block 3
    x = Conv2D(32,  (1, 1), activation='relu', padding='same', name='conv6',  data_format='channels_first' )(x)
    x = Conv2D(256,  (3, 3), activation='relu', padding='same', name='conv7',  data_format='channels_first' )(x)
    x = Conv2D(32,  (1, 1), activation='relu', padding='same', name='conv8',  data_format='channels_first' )(x)
    x = Conv2D(256,  (3, 3), activation='relu', padding='same', name='conv9',  data_format='channels_first' )(x)
    x = ( concatenate([ x ,f2],axis=1 )  )
    x = MaxPooling2D(2, strides=(2, 2), name='pool4', data_format='channels_first' )(x)
    f3 = x
    # block 4
    x = Conv2D(64,  (1, 1), activation='relu', padding='same', name='conv10', data_format='channels_first' )(x)
    x = Conv2D(512,  (3, 3), activation='relu', padding='same', name='conv11', data_format='channels_first' )(x)
    x = Conv2D(64,  (1, 1), activation='relu', padding='same', name='conv12', data_format='channels_first' )(x)
    x = Conv2D(512,  (3, 3), activation='relu', padding='same', name='conv13', data_format='channels_first' )(x)
    x = ( concatenate([ x ,f3],axis=1 )  )
    x = MaxPooling2D(2, strides=(2, 2), name='pool5', data_format='channels_first' )(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv14', data_format='channels_first' )(x)
    x = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv15', data_format='channels_first' )(x)
        
    o = x
	
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
    plot_model(model, to_file='tiny_yolonet.png')

    return model

def TinyYOLONet2( n_classes ,  input_height=416, input_width=608 , vgg_level=3):

    img_input = Input(shape=(3,input_height,input_width))

    # block 1
    x = Conv2D(16,   (3, 3), activation='relu', padding='same', name='conv0', data_format='channels_first' )(img_input)
    x = MaxPooling2D(2, strides=(2, 2), name='pool1', data_format='channels_first' )(x)
    x = Conv2D(32,   (3, 3), activation='relu', padding='same', name='conv1', data_format='channels_first' )(x)    
    x = MaxPooling2D(2, strides=(2, 2), name='pool2', data_format='channels_first' )(x)
    f1 = x
    # block 2
    x = Conv2D(16,   (1, 1), activation='relu', padding='same', name='conv2',  data_format='channels_first' )(x)	
    x = Conv2D(128,  (3, 3), activation='relu', padding='same', name='conv3',  data_format='channels_first' )(x)
    x = Conv2D(16,   (1, 1), activation='relu', padding='same', name='conv4',  data_format='channels_first' )(x)
    x = Conv2D(128,  (3, 3), activation='relu', padding='same', name='conv5',  data_format='channels_first' )(x)
    #x = ( concatenate([ x ,f1],axis=1 )  )
    x = MaxPooling2D(2, strides=(2, 2), name='pool3', data_format='channels_first' )(x)
    f2 = x
    # block 3
    x = Conv2D(32,  (1, 1), activation='relu', padding='same', name='conv6',  data_format='channels_first' )(x)
    x = Conv2D(256,  (3, 3), activation='relu', padding='same', name='conv7',  data_format='channels_first' )(x)
    x = Conv2D(32,  (1, 1), activation='relu', padding='same', name='conv8',  data_format='channels_first' )(x)
    x = Conv2D(256,  (3, 3), activation='relu', padding='same', name='conv9',  data_format='channels_first' )(x)
    #x = ( concatenate([ x ,f2],axis=1 )  )
    x = MaxPooling2D(2, strides=(2, 2), name='pool4', data_format='channels_first' )(x)
    f3 = x
    # block 4
    x = Conv2D(64,  (1, 1), activation='relu', padding='same', name='conv10', data_format='channels_first' )(x)
    x = Conv2D(512,  (3, 3), activation='relu', padding='same', name='conv11', data_format='channels_first' )(x)
    x = Conv2D(64,  (1, 1), activation='relu', padding='same', name='conv12', data_format='channels_first' )(x)
    x = Conv2D(512,  (3, 3), activation='relu', padding='same', name='conv13', data_format='channels_first' )(x)
    #x = ( concatenate([ x ,f3],axis=1 )  )
    x = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv14', data_format='channels_first' )(x)        
    o = x

    #o = Conv2DTranspose(128, (1,1), padding='same', name='deconv0', data_format='channels_first' )(o)        
    o = Conv2DTranspose(512, (3,3), padding='same', name='deconv1', data_format='channels_first' )(o)   
    #o = Conv2DTranspose(64, (1,1), padding='same', name='deconv2', data_format='channels_first' )(o)   	
    #o = Conv2DTranspose(512, (3,3), padding='same', name='deconv3', data_format='channels_first' )(o)   
    #o = Conv2DTranspose(64, (1,1), padding='same', name='deconv4', data_format='channels_first' )(o)   
    o = ( concatenate([ o ,f3],axis=1 )  )	
    o = ( UpSampling2D( (2,2), data_format='channels_first'))(o)
    # Block
    o = Conv2DTranspose(256, (3,3), padding='same', name='deconv5', data_format='channels_first' )(o)        
    #o = Conv2DTranspose(32, (1,1), padding='same', name='deconv6', data_format='channels_first' )(o)   
    #o = Conv2DTranspose(256, (3,3), padding='same', name='deconv7', data_format='channels_first' )(o)   	
    #o = Conv2DTranspose(32, (1,1), padding='same', name='deconv8', data_format='channels_first' )(o)   
    o = ( concatenate([ o ,f2],axis=1 )  )
    o = ( UpSampling2D((2,2)  , data_format='channels_first' ) )(o)
    # Block
    o = Conv2DTranspose(128, (3,3), padding='same', name='deconv9', data_format='channels_first' )(o)        
    #o = Conv2DTranspose(16, (1,1), padding='same', name='deconv10', data_format='channels_first' )(o)   
    #o = Conv2DTranspose(128, (3,3), padding='same', name='deconv11', data_format='channels_first' )(o)   	
    #o = Conv2DTranspose(16, (1,1), padding='same', name='deconv12', data_format='channels_first' )(o)   
    o = ( concatenate([ o ,f1],axis=1 )  )    
    o = ( UpSampling2D((2,2)  , data_format='channels_first' ))(o)
    
    # Block
    o = Conv2DTranspose(32, (1,1), padding='same', name='deconv13', data_format='channels_first' )(o)   
    o = ( UpSampling2D((2,2)  , data_format='channels_first' ))(o)
    o =  Conv2DTranspose(n_classes, (1,1), padding='same', name='deconv14', data_format='channels_first') ( o )
    #o =  Conv2D( n_classes , (3, 3) , padding='same', data_format='channels_first' )( o )
    o_shape = Model(img_input , o ).output_shape
    outputHeight = o_shape[2]
    outputWidth = o_shape[3]

    o = (Reshape((  -1  , outputHeight*outputWidth   )))(o)
    o = (Permute((2, 1)))(o)
    o = (Activation('softmax'))(o)
    model = Model( img_input , o )
    model.outputWidth = outputWidth
    model.outputHeight = outputHeight
    plot_model(model, to_file='tiny_yolonet2.png')

    return model

def Darknet19Encoder(  input_img , input_name, depth, resnet):
    # block 1
    x = Conv2D(32,   (3, 3), activation='relu', padding='same', data_format='channels_last', name=input_name )(input_img)
    x = MaxPooling2D(2, strides=(2, 2), data_format='channels_last' )(x)
    
    x = Conv2D(64,   (3, 3), activation='relu', padding='same', data_format='channels_last' )(x)    
    x = MaxPooling2D(2, strides=(2, 2), data_format='channels_last' )(x)
    f1 = x

    x = Conv2D(128,  (3, 3), activation='relu', padding='same', data_format='channels_last' )(x)
    x = Conv2D(64,   (1, 1), activation='relu', padding='same', data_format='channels_last' )(x)
    x = Conv2D(128,  (3, 3), activation='relu', padding='same', data_format='channels_last' )(x)
    x = MaxPooling2D(2, strides=(2, 2), data_format='channels_last' )(x)
    f2 = x
    
    x = Conv2D(256,  (3, 3), activation='relu', padding='same', data_format='channels_last' )(x)
    x = Conv2D(128,  (1, 1), activation='relu', padding='same', data_format='channels_last' )(x)
    x = Conv2D(256,  (3, 3), activation='relu', padding='same', data_format='channels_last' )(x)
    x = MaxPooling2D(2, strides=(2, 2), data_format='channels_last' )(x)
    f3 = x
    
    x = Conv2D(512,  (3, 3), activation='relu', padding='same', data_format='channels_last' )(x)
    x = Conv2D(256,  (1, 1), activation='relu', padding='same', data_format='channels_last' )(x)
    x = Conv2D(512,  (3, 3), activation='relu', padding='same', data_format='channels_last' )(x)
    x = Conv2D(256,  (1, 1), activation='relu', padding='same', data_format='channels_last' )(x)
    x = Conv2D(512,  (3, 3), activation='relu', padding='same', data_format='channels_last' )(x)
    x = MaxPooling2D(2, strides=(2, 2), data_format='channels_last' )(x)
    f4 = x
    
    x = Conv2D(1024,  (3, 3), activation='relu', padding='same', data_format='channels_last' )(x)
    x = Conv2D(512,  (1, 1), activation='relu', padding='same', data_format='channels_last' )(x)
    x = Conv2D(1024,  (3, 3), activation='relu', padding='same', data_format='channels_last' )(x)
    x = Conv2D(512,  (1, 1), activation='relu', padding='same', data_format='channels_last' )(x)
    x = Conv2D(1024,  (3, 3), activation='relu', padding='same', data_format='channels_last' )(x)

    f5 = x
    
    f = (f1, f2, f3, f4, f5)
    return (x, f)     



def TinyDarknetEncoder( input_img , input_name, depth, resnet):
    
    # block 1
    x = Conv2D(16,   (3, 3), activation='relu', padding='same', data_format='channels_first', name=input_name )(input_img)
    #x = ( concatenate([ x ,y],axis=1 )  )    
    x = MaxPooling2D(2, strides=(2, 2), data_format='channels_first' )(x)
    f1 = x
    x = Conv2D(32,   (3, 3), activation='relu', padding='same', data_format='channels_first' )(x)    
    x = MaxPooling2D(2, strides=(2, 2), data_format='channels_first' )(x)
    f2 = x
    # block 2
    x = Conv2D(16,   (1, 1), activation='relu', padding='same', data_format='channels_first' )(x)	
    y = Conv2D(128,  (3, 3), activation='relu', padding='same', data_format='channels_first' )(x)
    if resnet == True:
        y = concatenate([y, f2], axis=1)
    x = Conv2D(16,   (1, 1), activation='relu', padding='same', data_format='channels_first' )(y)
    x = Conv2D(128,  (3, 3), activation='relu', padding='same', data_format='channels_first' )(x)
    if resnet == True:
        x = concatenate([x, y], axis=1)
    
    x = MaxPooling2D(2, strides=(2, 2), data_format='channels_first' )(x)
    f3 = x
    # block 3
    x = Conv2D(32,  (1, 1), activation='relu', padding='same', data_format='channels_first' )(x)
    y = Conv2D(256,  (3, 3), activation='relu', padding='same', data_format='channels_first' )(x)
    if resnet == True:
        y = concatenate([y, f3], axis=1)
    x = Conv2D(32,  (1, 1), activation='relu', padding='same', data_format='channels_first' )(y)
    x = Conv2D(256,  (3, 3), activation='relu', padding='same', data_format='channels_first' )(x)
    if resnet == True:
        x = concatenate([x, y], axis=1)
    x = MaxPooling2D(2, strides=(2, 2), data_format='channels_first' )(x)
    f4 = x
    # block 4
    x = Conv2D(64,  (1, 1), activation='relu', padding='same', data_format='channels_first' )(x)
    y = Conv2D(512,  (3, 3), activation='relu', padding='same', data_format='channels_first' )(x)
    if resnet == True:
        y = concatenate([y, f4], axis=1)
    x = Conv2D(64,  (1, 1), activation='relu', padding='same', data_format='channels_first' )(y)
    x = Conv2D(512,  (3, 3), activation='relu', padding='same', data_format='channels_first' )(x)
    if resnet == True:
        x = concatenate([x, y], axis=1)
#    #x = ( concatenate([ x ,f3],axis=1 )  )
    x = Conv2D(128, (1, 1), activation='relu', padding='same', data_format='channels_first' )(x)  
    x = MaxPooling2D(2, strides=(2, 2), data_format='channels_first' )(x)
    f5 = x;
    
    if depth==1:
        x = f1
    elif depth==2:
        x = f2
    elif depth==3:
        x = f3
    elif depth==4:
        x = f4
    elif depth==5:
        x = f5
        
    f = (f1, f2, f3, f4, f5)
    
    return (x, f)      

def TinyDarknetEncoderLast( input_img , input_name, depth, resnet):
    
    # block 1
    x = Conv2D(16,   (3, 3), activation='relu', padding='same', data_format='channels_last', name=input_name )(input_img)
    #x = ( concatenate([ x ,y],axis=1 )  )    
    x = MaxPooling2D(2, strides=(2, 2), data_format='channels_last' )(x)
    f1 = x
    x = Conv2D(32,   (3, 3), activation='relu', padding='same', data_format='channels_last' )(x)    
    x = MaxPooling2D(2, strides=(2, 2), data_format='channels_last' )(x)
    f2 = x
    # block 2
    x = Conv2D(16,   (1, 1), activation='relu', padding='same', data_format='channels_last' )(x)	
    y = Conv2D(128,  (3, 3), activation='relu', padding='same', data_format='channels_last' )(x)
    if resnet == True:
        y = concatenate([y, f2], axis=3)
    x = Conv2D(16,   (1, 1), activation='relu', padding='same', data_format='channels_last' )(y)
    x = Conv2D(128,  (3, 3), activation='relu', padding='same', data_format='channels_last' )(x)
    if resnet == True:
        x = concatenate([x, y], axis=3)
    
    x = MaxPooling2D(2, strides=(2, 2), data_format='channels_last' )(x)
    x = Dropout(0.5)(x)
    f3 = x
    # block 3
    x = Conv2D(32,  (1, 1), activation='relu', padding='same', data_format='channels_last' )(x)
    y = Conv2D(256,  (3, 3), activation='relu', padding='same', data_format='channels_last' )(x)
    if resnet == True:
        y = concatenate([y, f3], axis=3)
    x = Conv2D(32,  (1, 1), activation='relu', padding='same', data_format='channels_last' )(y)
    x = Conv2D(256,  (3, 3), activation='relu', padding='same', data_format='channels_last' )(x)
    if resnet == True:
        x = concatenate([x, y], axis=3)
    x = MaxPooling2D(2, strides=(2, 2), data_format='channels_last' )(x)
    x = Dropout(0.5)(x)
    f4 = x
    # block 4
    x = Conv2D(64,  (1, 1), activation='relu', padding='same', data_format='channels_last' )(x)
    y = Conv2D(512,  (3, 3), activation='relu', padding='same', data_format='channels_last' )(x)
    if resnet == True:
        y = concatenate([y, f4], axis=3)
    x = Conv2D(64,  (1, 1), activation='relu', padding='same', data_format='channels_last' )(y)
    x = Conv2D(512,  (3, 3), activation='relu', padding='same', data_format='channels_last' )(x)
    if resnet == True:
        x = concatenate([x, y], axis=3)
#    #x = ( concatenate([ x ,f3],axis=1 )  )
    x = Conv2D(128, (1, 1), activation='relu', padding='same', data_format='channels_last' )(x)  
    x = MaxPooling2D(2, strides=(2, 2), data_format='channels_last' )(x)
    x = Dropout(0.5)(x)
    f5 = x;
    
    if depth==1:
        x = f1
    elif depth==2:
        x = f2
    elif depth==3:
        x = f3
    elif depth==4:
        x = f4
    elif depth==5:
        x = f5
        
    f = (f1, f2, f3, f4, f5)
    
    return (x, f)   

def TinyDarknetEncoderLastV2( input_img , input_name, depth, resnet):
    
    # block 1
    x = Conv2D(16,   (3, 3), padding='same', data_format='channels_last', name=input_name )(input_img)
    #x = ( concatenate([ x ,y],axis=1 )  
    x = Conv2D(16, (2,2),strides=(2, 2),padding='same', data_format='channels_last')(x)
#    x = MaxPooling2D(2, strides=(2, 2), data_format='channels_last' )(x)
    f1 = x
    x = Conv2D(32,   (3, 3), padding='same', data_format='channels_last' )(x)   
    x = Conv2D(32, (2,2),strides=(2, 2),padding='same', data_format='channels_last')(x)
#    x = MaxPooling2D(2, strides=(2, 2), data_format='channels_last' )(x)
    f2 = x
    # block 2
    x = Conv2D(16,   (1, 1), padding='same', data_format='channels_last' )(x)	
    y = Conv2D(128,  (3, 3), padding='same', data_format='channels_last' )(x)
    if resnet == True:
        y = concatenate([y, f2], axis=3)
    x = Conv2D(16,   (1, 1), padding='same', data_format='channels_last' )(y)
    x = Conv2D(128,  (3, 3), padding='same', data_format='channels_last' )(x)
    if resnet == True:
        x = concatenate([x, y], axis=3)
    
    x = Conv2D(128, (2,2),strides=(2, 2),padding='same', data_format='channels_last' )(x)
#    x = MaxPooling2D(2, strides=(2, 2), data_format='channels_last' )(x)
    f3 = x
    # block 3
    x = Conv2D(32,  (1, 1), padding='same', data_format='channels_last' )(x)
    y = Conv2D(256,  (3, 3), padding='same', data_format='channels_last' )(x)
    if resnet == True:
        y = concatenate([y, f3], axis=3)
    x = Conv2D(32,  (1, 1), padding='same', data_format='channels_last' )(y)
    x = Conv2D(256,  (3, 3), padding='same', data_format='channels_last' )(x)
    if resnet == True:
        x = concatenate([x, y], axis=3)
    x = Conv2D(256, (2,2),strides=(2, 2),padding='same', data_format='channels_last')(x)
#    x = MaxPooling2D(2, strides=(2, 2), data_format='channels_last' )(x)

    f4 = x
    # block 4
    x = Conv2D(64,  (1, 1), padding='same', data_format='channels_last' )(x)
    y = Conv2D(512,  (3, 3), padding='same', data_format='channels_last' )(x)
    if resnet == True:
        y = concatenate([y, f4], axis=3)
    x = Conv2D(64,  (1, 1), padding='same', data_format='channels_last' )(y)
    x = Conv2D(512,  (3, 3), padding='same', data_format='channels_last' )(x)
    if resnet == True:
        x = concatenate([x, y], axis=3)
#    #x = ( concatenate([ x ,f3],axis=1 )  )
    x = Conv2D(128, (1, 1), padding='same', data_format='channels_last' )(x)  
    x = Conv2D(128, (2,2),strides=(2, 2),padding='same', data_format='channels_last')(x)
#    x = MaxPooling2D(2, strides=(2, 2), data_format='channels_last' )(x)
    x = Dropout(0.5)(x)
    f5 = x;
    
    if depth==1:
        x = f1
    elif depth==2:
        x = f2
    elif depth==3:
        x = f3
    elif depth==4:
        x = f4
    elif depth==5:
        x = f5
        
    f = (f1, f2, f3, f4, f5)
    
    return (x, f)      

def Darknet19Decoder(  oin , n_classes, depth, resnet, unet, f):

    o = oin    
    if depth==5:
        o = oin
    o = Conv2D(1024,(3,3), activation='relu',padding='same', data_format='channels_last')(o)
    o = Conv2D(512,(1,1), activation='relu',padding='same', data_format='channels_last')(o)
    o = Conv2D(1024,(3,3), activation='relu',padding='same', data_format='channels_last')(o)
    o = Conv2D(512,(1,1), activation='relu',padding='same', data_format='channels_last')(o)
    o = Conv2D(1024,(3,3), activation='relu',padding='same', data_format='channels_last')(o)  
    if depth==4:
        o = f[3]
    o = ( UpSampling2D( (2,2), data_format='channels_last'))(o)
    o = Conv2D(512,(3,3), activation='relu',padding='same', data_format='channels_last')(o)
    o = Conv2D(256,(1,1), activation='relu',padding='same', data_format='channels_last')(o)
    o = Conv2D(512,(3,3), activation='relu',padding='same', data_format='channels_last')(o)
    o = Conv2D(256,(1,1), activation='relu',padding='same', data_format='channels_last')(o)
    o = Conv2D(512,(3,3), activation='relu',padding='same', data_format='channels_last')(o)  
    if depth==3:
        o = f[2]
    o = ( UpSampling2D( (2,2), data_format='channels_last'))(o)
    o = Conv2D(256,(3,3), activation='relu',padding='same', data_format='channels_last')(o)
    o = Conv2D(128,(1,1), activation='relu',padding='same', data_format='channels_last')(o)
    o = Conv2D(256,(3,3), activation='relu',padding='same', data_format='channels_last')(o)  
    if depth==2:
        o = f[1]
    o = ( UpSampling2D( (2,2), data_format='channels_last'))(o)
    o = Conv2D(128,(3,3), activation='relu',padding='same', data_format='channels_last')(o)
    o = Conv2D(64,(1,1), activation='relu',padding='same', data_format='channels_last')(o)
    o = Conv2D(128,(3,3), activation='relu',padding='same', data_format='channels_last')(o)  
    
    if depth==1:
        o = f[0]
    o = ( UpSampling2D( (2,2), data_format='channels_last'))(o)
    o = Conv2D(64,(3,3), activation='relu',padding='same', data_format='channels_last')(o)  
    
    o = ( UpSampling2D( (2,2), data_format='channels_last'))(o)
    o = Conv2D(32,(3,3), activation='relu',padding='same', data_format='channels_last')(o)  
    
    return (o)    

def TinyDarknetDecoder( oin , n_classes, depth, resnet, unet, f):
    o = f[4]
    if depth==5:
        o = oin
    if unet==True:
        o = concatenate([o, f[4]], axis=1)
    o = ( UpSampling2D( (2,2), data_format='channels_first'))(o)
    p = Conv2D(128, (1, 1), activation='relu', padding='same', data_format='channels_first' )(o) 

    o = Conv2D(512,  (3, 3), activation='relu', padding='same', data_format='channels_first' )(p)
    o = Conv2D(64,  (1, 1), activation='relu', padding='same', data_format='channels_first' )(o)
    if resnet == True:
        o = concatenate([o, p], axis=1)
    p = Conv2D(512,  (3, 3), activation='relu', padding='same', data_format='channels_first' )(o)
    o = Conv2D(64,  (1, 1), activation='relu', padding='same', data_format='channels_first' )(p)
    if resnet == True:
        o = concatenate([o, p], axis=1)
    o = ( BatchNormalization())(o)
    
     # block 4
    if depth==4:
        o = oin
    if unet==True:
        o = concatenate([o, f[3]], axis=1)
    o = ( UpSampling2D( (2,2), data_format='channels_first'))(o)
    o = Conv2D(256,  (3, 3), activation='relu', padding='same', data_format='channels_first' )(o)
    o = Conv2D(32,  (1, 1), activation='relu', padding='same', data_format='channels_first' )(o)
    o = Conv2D(256,  (3, 3), activation='relu', padding='same', data_format='channels_first' )(o)
    o = Conv2D(32,  (1, 1), activation='relu', padding='same', data_format='channels_first' )(o)
    o = ( BatchNormalization())(o)
    # block 3
    if depth==3:
        o = oin
    if unet==True:
        o = concatenate([o, f[2]], axis=1)
    p = ( UpSampling2D( (2,2), data_format='channels_first'))(o)
    o = Conv2D(128,  (3, 3), activation='relu', padding='same', data_format='channels_first' )(p)
    q = Conv2D(16,   (1, 1), activation='relu', padding='same', data_format='channels_first' )(o)
    if resnet == True:
        q = concatenate([q, p], axis=1)	
    o = Conv2D(128,  (3, 3), activation='relu', padding='same', data_format='channels_first' )(q)
    o = Conv2D(16,   (1, 1), activation='relu', padding='same', data_format='channels_first' )(o)
    if resnet == True:
        o = concatenate([o, q], axis=1)
    o = ( BatchNormalization())(o)
    # block 2
    if depth==2:
        o = oin
    if unet==True:
        o = concatenate([o, f[1]], axis=1)
    o = ( UpSampling2D( (2,2), data_format='channels_first'))(o)
    o = ( BatchNormalization())(o)
    o = Conv2D(32,   (3, 3), activation='relu', padding='same', data_format='channels_first' )(o)      
    if depth==1:
        o = oin
    if unet==True:
        o = concatenate([o, f[0]], axis=1)	
    o = ( UpSampling2D( (2,2), data_format='channels_first'))(o)
    o = ( BatchNormalization())(o)
    o = ( ZeroPadding2D((1,1)  , data_format='channels_first' ))(o)
    o = ( Conv2D( 32 , (3, 3), padding='valid'  , data_format='channels_first' ))(o)
    
    o = Conv2D(n_classes,   (3, 3), activation='relu', padding='same', data_format='channels_first')(o)
    # block 1


    return (o)      

def TinyDarknetDecoderLast( oin , n_classes, depth, resnet, unet, f):
    o = f[4]
    if depth==5:
        o = oin
    if unet==True:
        o = concatenate([o, f[4]], axis=3)
    o = ( UpSampling2D( (2,2), data_format='channels_last'))(o)
    p = Conv2D(128, (1, 1), activation='relu', padding='same', data_format='channels_last' )(o) 

    o = Conv2D(512,  (3, 3), activation='relu', padding='same', data_format='channels_last' )(p)
    o = Conv2D(64,  (1, 1), activation='relu', padding='same', data_format='channels_last' )(o)
    if resnet == True:
        o = concatenate([o, p], axis=3)
    p = Conv2D(512,  (3, 3), activation='relu', padding='same', data_format='channels_last' )(o)
    o = Conv2D(64,  (1, 1), activation='relu', padding='same', data_format='channels_last' )(p)
    if resnet == True:
        o = concatenate([o, p], axis=3)
    o = ( BatchNormalization())(o)
    
     # block 4
    if depth==4:
        o = oin
    if unet==True:
        o = concatenate([o, f[3]], axis=3)
    o = ( UpSampling2D( (2,2), data_format='channels_last'))(o)
    o = Conv2D(256,  (3, 3), activation='relu', padding='same', data_format='channels_last' )(o)
    o = Conv2D(32,  (1, 1), activation='relu', padding='same', data_format='channels_last' )(o)
    o = Conv2D(256,  (3, 3), activation='relu', padding='same', data_format='channels_last' )(o)
    o = Conv2D(32,  (1, 1), activation='relu', padding='same', data_format='channels_last' )(o)
    o = ( BatchNormalization())(o)
    o = Dropout(0.5)(o)
    # block 3
    if depth==3:
        o = oin
    if unet==True:
        o = concatenate([o, f[2]], axis=3)
    p = ( UpSampling2D( (2,2), data_format='channels_last'))(o)
    o = Conv2D(128,  (3, 3), activation='relu', padding='same', data_format='channels_last' )(p)
    q = Conv2D(16,   (1, 1), activation='relu', padding='same', data_format='channels_last' )(o)
    if resnet == True:
        q = concatenate([q, p], axis=3)	
    o = Conv2D(128,  (3, 3), activation='relu', padding='same', data_format='channels_last' )(q)
    o = Conv2D(16,   (1, 1), activation='relu', padding='same', data_format='channels_last' )(o)
    if resnet == True:
        o = concatenate([o, q], axis=3)
    o = ( BatchNormalization())(o)
    o = Dropout(0.5)(o)
    # block 2
    if depth==2:
        o = oin
    if unet==True:
        o = concatenate([o, f[1]], axis=3)
    o = ( UpSampling2D( (2,2), data_format='channels_last'))(o)
    o = Dropout(0.5)(o)
    o = Conv2D(32,   (3, 3), activation='relu', padding='same', data_format='channels_last' )(o)      
    if depth==1:
        o = oin
    if unet==True:
        o = concatenate([o, f[0]], axis=3)	
    o = ( UpSampling2D( (2,2), data_format='channels_last'))(o)
    o = ( BatchNormalization())(o)
    o = Dropout(0.5)(o)
    o = ( ZeroPadding2D((1,1)  , data_format='channels_last' ))(o)
    o = ( Conv2D( 16 , (3, 3), padding='valid'  , data_format='channels_last' ))(o)
    
    o = Conv2D(n_classes,   (3, 3), activation='relu', padding='same', data_format='channels_last')(o)
    # block 1


    return (o)     

def SegnetDecoder (oin, n_classes, depth):
    if depth==5:
        o = oin
    o = ( UpSampling2D( (2,2), data_format='channels_first'))(o)
    o = ( ZeroPadding2D( (1,1) , data_format='channels_first' ))(o)
    o = ( Conv2D(512, (3, 3), padding='valid', data_format='channels_first'))(o)
    o = ( BatchNormalization())(o)

    if depth==4:
        o = oin

    o = ( UpSampling2D( (2,2), data_format='channels_first'))(o)
    #o = ( concatenate([ o ,f4],axis=1 )  )	
    o = ( ZeroPadding2D( (1,1), data_format='channels_first'))(o)
    o = ( Conv2D( 256, (3, 3), padding='valid', data_format='channels_first'))(o)
    o = ( BatchNormalization())(o)

    if depth==3:
        o = oin

    o = ( UpSampling2D((2,2)  , data_format='channels_first' ) )(o)
    #o = ( concatenate([ o ,f3],axis=1 )  )
    o = ( ZeroPadding2D((1,1) , data_format='channels_first' ))(o)
    o = ( Conv2D( 128 , (3, 3), padding='valid' , data_format='channels_first' ))(o)
    o = ( BatchNormalization())(o)

    if depth==2:
        o = oin

    o = ( UpSampling2D((2,2)  , data_format='channels_first' ))(o)
    #o = ( concatenate([ o ,f2],axis=1 )  )
    o = ( ZeroPadding2D((1,1)  , data_format='channels_first' ))(o)
    o = ( Conv2D( 64 , (3, 3), padding='valid'  , data_format='channels_first' ))(o)
    o = ( BatchNormalization())(o)

    if depth==1:
        o = oin

    o = ( UpSampling2D((2,2)  , data_format='channels_first' ))(o)
    #o = ( concatenate([ o ,f1],axis=1 )  )
    o = ( ZeroPadding2D((1,1)  , data_format='channels_first' ))(o)
    o = ( Conv2D( 64 , (3, 3), padding='valid'  , data_format='channels_first' ))(o)
    o = ( BatchNormalization())(o)


    oout =  Conv2D( n_classes , (3, 3) , padding='same', data_format='channels_first' )( o )
    return (oout)


def TinyYOLONet3( n_classes ,  input_height=416, input_width=608 , vgg_level=3):
    depth = True
    img_input = Input(shape=(3,input_height,input_width))
    depth_input = Input(shape=(1,input_height,input_width))
    
    (x, f1, f2, f3) = TinyDarknetEncoder(img_input, 'image_input')
    
    if depth:
        (xd, fd1, fd2, fd3) = TinyDarknetEncoder(depth_input, 'depth_input')
        x = ( concatenate([ x ,xd],axis=1 )  )	
    
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
    plot_model(model, to_file='tiny_yolonet3.png')

    return model

def TinyYOLONet4( n_classes ,  input_height=416, input_width=608 , vgg_level=3):
    depth = True
    img_input = Input(shape=(3,input_height,input_width))
    depth_input = Input(shape=(1,input_height,input_width))
    
    (x, f1, f2, f3, f4, f5) = TinyDarknetEncoder(img_input, 'image_input', 5)
    
    if depth:
        (xd, fd1, fd2, fd3, fd4, fd5) = TinyDarknetEncoder(depth_input, 'depth_input', 5)
        x = ( concatenate([ x ,xd],axis=1 )  )	
    
    o = x
    o = TinyDarknetDecoder(o,n_classes, 5)
    
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

def TinyYOLONet5( n_classes ,  input_height=416, input_width=608 , vgg_level=3):
    depth = 4
    img_input = Input(shape=(input_height,input_width,3))
    depth_input = Input(shape=(input_height,input_width,1))
    
    (x, f) = TinyDarknetEncoderLast(img_input, 'image_input', depth, True)
    (xd, fd) = TinyDarknetEncoderLast(depth_input, 'depth_input', depth, True)
    o = x
    od = xd    
    o = TinyDarknetDecoderLast(o,n_classes, depth, True, False, f)
    od = TinyDarknetDecoderLast(od,n_classes, depth, True, False, fd)
    o = ( concatenate([ o ,od],axis=3 )  )	
    o = ( BatchNormalization())(o)
    o = ( Conv2D( n_classes , (1, 1), padding='valid'  , data_format='channels_last' ))(o)
        
    
    o_shape = Model(inputs=[img_input, depth_input] , outputs = o ).output_shape
    outputHeight = o_shape[1]
    outputWidth = o_shape[2]

#    o = (Reshape((  n_classes  , outputHeight*outputWidth   )))(o)
    o = (Reshape(( outputHeight*outputWidth ,  n_classes    )))(o)
#    o = (Reshape((  n_classes  , outputHeight*outputWidth   )))(o)
#    o = (Permute((2, 1)))(o)
    o = (Activation('softmax'))(o)
    model = Model( inputs=[img_input, depth_input] , outputs = o )
 #   model = Model( inputs=img_input , outputs = o )
    model.outputWidth = outputWidth
    model.outputHeight = outputHeight
#    plot_model(model, to_file='tiny_yolonet5.png')

    return model

def TinyYOLONet6( n_classes ,  input_height=416, input_width=608 , vgg_level=3):
    depth = 4
    img_input = Input(shape=(input_height,input_width,3))
    depth_input = Input(shape=(input_height,input_width,1))
    
    (x, f) = TinyDarknetEncoderLast(img_input, 'image_input', depth, True)
    (xd, fd) = TinyDarknetEncoderLast(depth_input, 'depth_input', depth, True)
    o = x
    od = xd    
    o = TinyDarknetDecoderLast(o,n_classes, depth, True, False, f)
    od = TinyDarknetDecoderLast(od,n_classes, depth, True, False, fd)
#    o = ( concatenate([ o ,od],axis=1 )  )	
    o = ( BatchNormalization())(o)
    o = ( Conv2D( n_classes , (1, 1), padding='valid'  , data_format='channels_last' ))(o)
        
    
    o_shape = Model(inputs=[img_input, depth_input] , outputs = o ).output_shape
    outputHeight = o_shape[1]
    outputWidth = o_shape[2]

#    o = (Reshape((  n_classes  , outputHeight*outputWidth   )))(o)
    o = (Reshape(( outputHeight*outputWidth ,  n_classes    )))(o)
    
    
#    o = (Reshape((  n_classes  , outputHeight*outputWidth   )))(o)
#    o = (Permute((2, 1)))(o)
    o = (Activation('softmax'))(o)
#    model = Model( inputs=[img_input, depth_input] , outputs = o )
    model = Model( inputs=img_input , outputs = o )
    model.outputWidth = outputWidth
    model.outputHeight = outputHeight
#    plot_model(model, to_file='tiny_yolonet5.png')

    return model

def TinyYOLONet7( n_classes ,  input_height=416, input_width=608 , vgg_level=3):
    depth = 5
    img_input = Input(shape=(input_height,input_width,3))
    depth_input = Input(shape=(input_height,input_width,1))
    
    (x, f) = TinyDarknetEncoderLastV2(img_input, 'image_input', depth, True)
    (xd, fd) = TinyDarknetEncoderLastV2(depth_input, 'depth_input', depth, True)
#    (x, f) = Darknet19Encoder(img_input, 'image_input', depth, True)
#    (xd, fd) = Darknet19Encoder(depth_input, 'depth_input', depth, True)
    o = x
    od = xd    
    o = ( Add()([ o ,od] ))
    o = TinyDarknetDecoderLast(o,n_classes, depth, True, False, f)    
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

def TinyYOLONet8( n_classes ,  input_height=416, input_width=608 , vgg_level=3):
    depth = 5
    img_input = Input(shape=(input_height,input_width,3))
    depth_input = Input(shape=(input_height,input_width,1))
    
    (x, f) = TinyDarknetEncoderLastV2(img_input, 'image_input', depth, True)

    o = x
    o = TinyDarknetDecoderLast(o,n_classes, depth, True, False, f)    
    o = ( BatchNormalization())(o)
    o = ( Conv2D( n_classes , (1, 1), padding='valid'  , data_format='channels_last' ))(o)
            
    o_shape = Model(inputs=img_input , outputs = o ).output_shape
    outputHeight = o_shape[1]
    outputWidth = o_shape[2]

    o = (Reshape(( outputHeight*outputWidth ,  n_classes    )))(o)
    o = (Activation('softmax'))(o)
    model = Model( inputs=img_input , outputs = o )
 #   model = Model( inputs=img_input , outputs = o )
    model.outputWidth = outputWidth
    model.outputHeight = outputHeight
#    plot_model(model, to_file='tiny_yolonet7.png')

    return model

def DarkNet19( n_classes ,  input_height=416, input_width=608 , vgg_level=3):
    depth = 4
    img_input = Input(shape=(input_height,input_width,3))
    depth_input = Input(shape=(input_height,input_width,1))
    
    (x, f) = Darknet19Encoder(img_input, 'image_input', depth, True)
    (xd, fd) = Darknet19Encoder(depth_input, 'depth_input', depth, True)
    o = x
    od = xd
    
    g = (( Add()([ f[0], fd[0]])), ( Add()([ f[1], fd[1]])), ( Add()([ f[2], fd[2]])), ( Add()([ f[3], fd[3]])), ( Add()([ f[4], fd[4]])))
    
    p = ( Add()([ o ,od] ))
    o = Darknet19Decoder(p,n_classes, depth, True, False, g)    
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
    m = YOLONet( 101 )
    from keras.utils import plot_model
    plot_model( m , show_shapes=True , to_file='model.png')
    
    

