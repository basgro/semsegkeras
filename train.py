from __future__ import division

import argparse
import Models , LoadBatches
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import matplotlib.pyplot as plt  
from livelossplot import PlotLossesKeras
import cPickle as pickle
import json

def iou_coef(y_true, y_pred, smooth=1):
    """
    IoU = (|X &amp; Y|)/ (|X or Y|)
    """
    intersection = K.sum(K.abs(y_true * y_pred))
    union = K.sum((y_true) + K.sum(y_pred)) - intersection
    return (intersection + smooth) / ( union + smooth)

def iou_coef_loss(y_true, y_pred):
    return -iou_coef(y_true, y_pred)

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

#def mIOU(y_true, y_pred):
#    shp = y_pred.shape
#    IOUc = np.zeros((14));
#    for i in range(14):
#        pred = (y_pred[:,i]==1)
#        true = (y_true[:,i]==1)
#        intersection = pred*true
#        union = pred+true
#        if np.sum(intersection) > 0:
#            IOUc[i] = float(np.sum(intersection))/float(np.sum(union))
#    print(np.round(IOUc,2))
#    return np.round(np.mean(IOUc[np.nonzero(IOUc)]),2)

def mean_iou(y_true, y_pred):
   score, up_opt = tf.metrics.mean_iou(y_true, y_pred, 13)
   K.get_session().run(tf.local_variables_initializer())
   with tf.control_dependencies([up_opt]):
       score = tf.identity(score)
   return score

def class_acc(y_true, y_pred):
   score, up_opt = tf.metrics.mean_per_class_accuracy(y_true, y_pred, 13)
   K.get_session().run(tf.local_variables_initializer())
   with tf.control_dependencies([up_opt]):
       score = tf.identity(score)
   return score

#def class_acc(y_true, y_pred):
#   score, up_opt = tf.metrics.average_precision_at_k(y_true, y_pred, 13)
#   K.get_session().run(tf.local_variables_initializer())
#   with tf.control_dependencies([up_opt]):
#       score = tf.identity(score)
#   return score





parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type = str  )
parser.add_argument("--train_images", type = str  )
parser.add_argument("--train_depth", type = str)
parser.add_argument("--train_annotations", type = str  )
parser.add_argument("--n_classes", type=int )
parser.add_argument("--input_height", type=int , default = 224  )
parser.add_argument("--input_width", type=int , default = 224 )

parser.add_argument('--validate',action='store_false')
parser.add_argument("--val_images", type = str , default = "")
parser.add_argument("--val_depth", type = str)
parser.add_argument("--val_annotations", type = str , default = "")

parser.add_argument("--epochs", type = int, default = 5 )
parser.add_argument("--batch_size", type = int, default = 2 )
parser.add_argument("--val_batch_size", type = int, default = 2 )
parser.add_argument("--load_weights", type = str , default = "")

parser.add_argument("--model_name", type = str , default = "")
parser.add_argument("--optimizer_name", type = str , default = "adadelta")


args = parser.parse_args()

train_images_path = args.train_images
train_depth_path = args.train_depth
train_segs_path = args.train_annotations
train_batch_size = args.batch_size
n_classes = args.n_classes
input_height = args.input_height
input_width = args.input_width
validate = args.validate
save_weights_path = args.save_weights_path
epochs = args.epochs
load_weights = args.load_weights

optimizer_name = args.optimizer_name
model_name = args.model_name

cmu=False
if cmu:
    class_weight = {0:1., 1:5.}

if validate:
	val_images_path = args.val_images
	val_segs_path = args.val_annotations
	val_batch_size = args.val_batch_size
        val_depth_path = args.val_depth

modelFns = { 'vgg_segnet':Models.VGGSegnet.VGGSegnet , 'vgg_unet':Models.VGGUnet.VGGUnet , 'vgg_unet2':Models.VGGUnet.VGGUnet2 , 'fcn8':Models.FCN8.FCN8 , 'fcn32':Models.FCN32.FCN32 , 'vgg_segnet_own':Models.VGGSegnetOwn.VGGSegnetOwn , 'unet':Models.Unet.Unet , 'yolonet':Models.YOLONet.YOLONet , 'tinyyolonet':Models.YOLONet.TinyYOLONet , 'tinyyolonet2':Models.YOLONet.TinyYOLONet2 ,'tinyyolonet3':Models.YOLONet.TinyYOLONet3, 'tinyyolonet4':Models.YOLONet.TinyYOLONet4,  'tinyyolonet5':Models.YOLONet.TinyYOLONet5 , 'tinyyolonet6':Models.YOLONet.TinyYOLONet6  , 'tinyyolonet7':Models.YOLONet.TinyYOLONet7 ,  'squeezenet':Models.Squeezenet.Squeezenet, 'squeezenet2':Models.Squeezenet.Squeezenet2,  'squeezenet3':Models.Squeezenet.Squeezenet3}
modelFN = modelFns[ model_name ]

m = modelFN( n_classes , input_height=input_height, input_width=input_width   )
optim = tf.keras.optimizers.SGD(0.001, 0.9, 0.005);
optim = tf.keras.optimizers.Adam();
m.compile(loss='categorical_crossentropy',
      optimizer= optimizer_name ,
      metrics=['accuracy'])
m.summary()

#class_weights = { 2.8921,0.40153,0.34536,0.10226,0.42239,0.34653,1.3387,0.31832,1.998,13.8256,1.401,0.79807,1.5706,11.7135}
#class_weights = {0,0.95061,0.81762,0.23031,1,0.8204,3.1694,0.75361,32.7316,3.3168,1.7689,3.7184} #12 classes fold 3
#class_weights = {0, 0.5332,    0.4241,    0.1395,    0.8103,    0.3987,    1.8436,    0.3977, 1.3018,    2.1714,    1.7475,    1.0000,    1.7159,   11.7875 }
#class_weights = { 8.349,0.90366,1,0.29392,1.0034,3.8764,0.36264}          

class_weights = {4.4449,0.82222,0.65221,0.1738,0.93241,0.45262,1.3791,0.45549,0.94235,1.0652,1.6342,2.5373,1.313,16.5788}  # area1 50/50
#class_weights = {3.3386,0.53808,0.46286,0.14516,1.2195,0.32162,1.3126,0.46347,0.65074,0.95109,1.0542,1.4303,1.3436,11.5529}; #area6 50/50
       
#shapes = [(w.shape) for w in m.get_weights()]
#weights = [np.random.randn(*s) for s in shapes]
#m.set_weights(weights)
#m.compile(loss='categorical_crossentropy',  
#      optimizer= optimizer_name ,
#      metrics=['accuracy'])


if len( load_weights ) > 0:
    print load_weights
    m.load_weights(load_weights)


print "Model output shape" ,  m.output_shape

output_height = m.outputHeight
output_width = m.outputWidth

G  = LoadBatches.imageSegmentationGenerator( train_images_path , train_segs_path ,  train_batch_size,  n_classes , input_height , input_width , output_height , output_width   )

G22  = LoadBatches.imageSegmentationGenerator2( train_images_path , train_depth_path, train_segs_path ,  train_batch_size,  n_classes , input_height , input_width , output_height , output_width   )


if validate:
	G2  = LoadBatches.imageSegmentationGenerator2( val_images_path ,val_depth_path,  val_segs_path ,  val_batch_size,  n_classes , input_height , input_width , output_height , output_width   )
if validate:
	G3  = LoadBatches.imageSegmentationGenerator( val_images_path,  val_segs_path ,  val_batch_size,  n_classes , input_height , input_width , output_height , output_width   )



if not validate:
	for ep in range( epochs ):
		m.fit_generator( G , 512  , epochs=1 )
		m.save_weights( save_weights_path + "." + str( ep ) )
		m.save( save_weights_path + ".model." + str( ep ) )
else:
	for ep in range( epochs ):
            G22  = LoadBatches.imageSegmentationGenerator2( train_images_path , train_depth_path, train_segs_path ,  train_batch_size,  n_classes , input_height , input_width , output_height , output_width   )
            G2  = LoadBatches.imageSegmentationGenerator2( val_images_path ,val_depth_path,  val_segs_path ,  val_batch_size,  n_classes , input_height , input_width , output_height , output_width   )
            print("****************\n*** Epoch: "+str(ep)+"/"+str(epochs-1)+"\n****************")
            history = m.fit_generator( G22 , 2582  , validation_data=G2, class_weight=class_weights, validation_steps=2582 ,  epochs=1)
	m.save_weights( save_weights_path + "." + str( epochs )  )
	m.save( save_weights_path + "." + str( epochs ) )
        #26451
plt.figure(1)  
        # summarize history for accuracy
plt.subplot(211)  
plt.plot(history.history['acc'])  
plt.plot(history.history['val_acc'])  
plt.title('model accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'test'], loc='upper left')    
         # summarize history for loss  
         
plt.subplot(212)  
plt.plot(history.history['loss'])  
plt.plot(history.history['val_loss'])  
plt.title('model loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train', 'test'], loc='upper left')  

#plt.subplot(223)  
#plt.plot(history.history['class_acc'])  
#plt.plot(history.history['val_class_acc'])  
#plt.title('model loss')  
#plt.ylabel('mean per class accuracy')  
#plt.xlabel('epoch')  
#plt.legend(['train', 'test'], loc='upper left')  
#
#plt.subplot(224)  
#plt.plot(history.history['mean_iou'])  
#plt.plot(history.history['val_mean_iou'])  
#plt.title('model loss')  
#plt.ylabel('mean iou')  
#plt.xlabel('epoch')  
#plt.legend(['train', 'test'], loc='upper left')  
#plt.show() 

#with open('file.json', 'w') as f:
#    json.dump(history.history, f)


m.save('keras_model.h5',include_optimizer=False)