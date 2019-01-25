########################################################################
#
# Copyright (c) 2017, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

"""
    Live camera sample showing the camera information and video in real time and allows to control the different
    settings.
"""
from __future__ import division
import cv2
import pyzed.sl as sl
import argparse
import Models
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

camera_settings = sl.CAMERA_SETTINGS.CAMERA_SETTINGS_BRIGHTNESS
str_camera_settings = "BRIGHTNESS"
step_camera_settings = 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_weights_path", type = str, default = "weights/tinydarknet7_shortcuts_rgbd"  )
    parser.add_argument("--epoch_number", type = int, default = 14 )
    parser.add_argument("--test_images", type = str , default = "")
    parser.add_argument("--test_depth", type = str , default = "")
    parser.add_argument("--output_path", type = str , default = "")
    parser.add_argument("--input_height", type=int , default = 224  )
    parser.add_argument("--input_width", type=int , default = 320 )
    parser.add_argument("--model_name", type = str , default = "tinyyolonet7")
    parser.add_argument("--optimizer_name", type = str , default = "adadelta")
    parser.add_argument("--n_classes", type=int, default = 12 )
    args = parser.parse_args()

    n_classes = args.n_classes
    model_name = args.model_name
    images_path = args.test_images
    depth_path = args.test_depth
    input_width =  args.input_width
    input_height = args.input_height
    epoch_number = args.epoch_number
    optimizer_name = args.optimizer_name

    modelFN = Models.YOLONet.TinyYOLONet7

    print("Running...")
    filepath = '/home/basgro/Documents/ZED/HD720_gang_thuis.svo'
    cam = sl.Camera()
    if not cam.is_opened():
        print("Opening ZED Camera...")
    init = sl.InitParameters(svo_input_filename=filepath)
    #    init = sl.InitParameters()
    init.depth_mode = sl.DEPTH_MODE.DEPTH_MODE_ULTRA # Set the depth mode to ULTRA
    init.coordinate_units = UNIT_METER;
    init_parameters.depth_minimum_distance = 0.15
    init_parameters.depth_maximum_distance = 40

    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    runtime = sl.RuntimeParameters()
    mat = sl.Mat()
    depth = sl.Mat()

    print_camera_information(cam)
    print_help()

    m = modelFN( n_classes , input_height=input_height, input_width=input_width   )
    
    print( args.save_weights_path + "." + str(  epoch_number ) )
    m.load_weights(  args.save_weights_path + "." + str(  epoch_number ) )
    optim = tf.keras.optimizers.SGD(0.001, 0.9, 0.005);
    m.compile(loss='categorical_crossentropy',
      optimizer= optimizer_name ,
      metrics=['categorical_accuracy'])
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
    classes = ['', 'floor', 'ceiling', 'wall', 'bookcase', 'door', 'window', 'clutter', 'column', 'beam', 'table', 'chair', 'board', 'sofa'];


    key = ''
    
    height = 224
    width = 640
    res = np.zeros((height+60, width+200, 3), dtype=np.uint8)
    x_offset = 60  
    y_offset = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(res,'Original/predicted', (10,16) , font, 0.5,(255,255,255),1,cv2.LINE_AA) 
    c = 0;
#    while c<n_classes-1:
    
    blockheight = 15
    for c in range(n_classes):
        res[61+c*blockheight:60+(c+1)*blockheight,width+1:width+1+blockheight,0]=colors[c][0]
        res[61+c*blockheight:60+(c+1)*blockheight,width+1:width+1+blockheight,1]=colors[c][1]
        res[61+c*blockheight:60+(c+1)*blockheight,width+1:width+1+blockheight,2]=colors[c][2]
        cv2.putText(res,classes[c], (width+1+18,57+(c+1)*blockheight) , font, 0.4,(255,255,255),1,cv2.LINE_AA) 
        c = c + 1
        
    
    
    while key != 113:  # for 'q' key
        start_time=time.time()
        err = cam.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            cam.retrieve_image(mat, sl.VIEW.VIEW_LEFT)
            cam.retrieve_image(depth, sl.VIEW.VIEW_DEPTH)
            frame = cv2.resize(mat.get_data(), ( 320 , 224 ))
            cv_depth = cv2.resize(depth.get_data(), ( 320 , 224 ))
            
            
            X = getImageArr(frame[:,:,:3] , args.input_width  , args.input_height, odering='channels_last'  )
            Y = getImageArr(cv_depth[:,:,:3] , args.input_width  , args.input_height, imgNorm='depth', odering='channels_last'  )
            mark1=time.time()
            pr = m.predict( [np.array([X]), np.array([Y])] )[0]
            mark2 = time.time()
            pr = pr.reshape(( output_height ,  output_width , n_classes ) ).argmax( axis=2 )
            seg_img = np.zeros( ( output_height , output_width , 3  ) )
            for c in range(n_classes):
                seg_img[:,:,0] += ( (pr[:,: ] == c )*( colors[c][0] )).astype('uint8')
                seg_img[:,:,1] += ((pr[:,: ] == c )*( colors[c][1] )).astype('uint8')
                seg_img[:,:,2] += ((pr[:,: ] == c )*( colors[c][2] )).astype('uint8')
            seg_img = cv2.resize(seg_img  , (input_width , input_height ))            
            
            both = np.hstack((frame[:,:,:3], np.uint8(seg_img) ))
            

            res[x_offset:both.shape[0]+x_offset,y_offset:both.shape[1]+y_offset,0:3] = both
            res[20:60,5:300,:]=0
            total_time = time.time() - start_time
            inference_time = mark2-mark1
            cv2.putText(res, 'Framerate = '+str(np.round(1/(time.time() - start_time),1))+' fps',(10,32), font, 0.5,(255,255,255),1,cv2.LINE_AA) 
            cv2.putText(res, 'Inference time = ' + str(round(inference_time*1000,2))+' ms',(10,48), font, 0.5,(255,255,255),1,cv2.LINE_AA) 
            cv2.imshow('frame',res)
            
            key = cv2.waitKey(5)
            settings(key, cam, runtime, mat)
        else:
            key = cv2.waitKey(5)
    cv2.destroyAllWindows()

    cam.close()
    print("\nFINISH")


def print_camera_information(cam):
    print("Resolution: {0}, {1}.".format(round(cam.get_resolution().width, 2), cam.get_resolution().height))
    print("Camera FPS: {0}.".format(cam.get_camera_fps()))
    print("Firmware: {0}.".format(cam.get_camera_information().firmware_version))
    print("Serial number: {0}.\n".format(cam.get_camera_information().serial_number))


def print_help():
    print("Help for camera setting controls")
    print("  Increase camera settings value:     +")
    print("  Decrease camera settings value:     -")
    print("  Switch camera settings:             s")
    print("  Reset all parameters:               r")
    print("  Record a video:                     z")
    print("  Quit:                               q\n")


def settings(key, cam, runtime, mat):
    if key == 115:  # for 's' key
        switch_camera_settings()
    elif key == 43:  # for '+' key
        current_value = cam.get_camera_settings(camera_settings)
        cam.set_camera_settings(camera_settings, current_value + step_camera_settings)
        print(str_camera_settings + ": " + str(current_value + step_camera_settings))
    elif key == 45:  # for '-' key
        current_value = cam.get_camera_settings(camera_settings)
        if current_value >= 1:
            cam.set_camera_settings(camera_settings, current_value - step_camera_settings)
            print(str_camera_settings + ": " + str(current_value - step_camera_settings))
    elif key == 114:  # for 'r' key
        cam.set_camera_settings(sl.CAMERA_SETTINGS.CAMERA_SETTINGS_BRIGHTNESS, -1, True)
        cam.set_camera_settings(sl.CAMERA_SETTINGS.CAMERA_SETTINGS_CONTRAST, -1, True)
        cam.set_camera_settings(sl.CAMERA_SETTINGS.CAMERA_SETTINGS_HUE, -1, True)
        cam.set_camera_settings(sl.CAMERA_SETTINGS.CAMERA_SETTINGS_SATURATION, -1, True)
        cam.set_camera_settings(sl.CAMERA_SETTINGS.CAMERA_SETTINGS_GAIN, -1, True)
        cam.set_camera_settings(sl.CAMERA_SETTINGS.CAMERA_SETTINGS_EXPOSURE, -1, True)
        cam.set_camera_settings(sl.CAMERA_SETTINGS.CAMERA_SETTINGS_WHITEBALANCE, -1, True)
        print("Camera settings: reset")
    elif key == 122:  # for 'z' key
        record(cam, runtime, mat)


def switch_camera_settings():
    global camera_settings
    global str_camera_settings
    if camera_settings == sl.CAMERA_SETTINGS.CAMERA_SETTINGS_BRIGHTNESS:
        camera_settings = sl.CAMERA_SETTINGS.CAMERA_SETTINGS_CONTRAST
        str_camera_settings = "Contrast"
        print("Camera settings: CONTRAST")
    elif camera_settings == sl.CAMERA_SETTINGS.CAMERA_SETTINGS_CONTRAST:
        camera_settings = sl.CAMERA_SETTINGS.CAMERA_SETTINGS_HUE
        str_camera_settings = "Hue"
        print("Camera settings: HUE")
    elif camera_settings == sl.CAMERA_SETTINGS.CAMERA_SETTINGS_HUE:
        camera_settings = sl.CAMERA_SETTINGS.CAMERA_SETTINGS_SATURATION
        str_camera_settings = "Saturation"
        print("Camera settings: SATURATION")
    elif camera_settings == sl.CAMERA_SETTINGS.CAMERA_SETTINGS_SATURATION:
        camera_settings = sl.CAMERA_SETTINGS.CAMERA_SETTINGS_GAIN
        str_camera_settings = "Gain"
        print("Camera settings: GAIN")
    elif camera_settings == sl.CAMERA_SETTINGS.CAMERA_SETTINGS_GAIN:
        camera_settings = sl.CAMERA_SETTINGS.CAMERA_SETTINGS_EXPOSURE
        str_camera_settings = "Exposure"
        print("Camera settings: EXPOSURE")
    elif camera_settings == sl.CAMERA_SETTINGS.CAMERA_SETTINGS_EXPOSURE:
        camera_settings = sl.CAMERA_SETTINGS.CAMERA_SETTINGS_WHITEBALANCE
        str_camera_settings = "White Balance"
        print("Camera settings: WHITEBALANCE")
    elif camera_settings == sl.CAMERA_SETTINGS.CAMERA_SETTINGS_WHITEBALANCE:
        camera_settings = sl.CAMERA_SETTINGS.CAMERA_SETTINGS_BRIGHTNESS
        str_camera_settings = "Brightness"
        print("Camera settings: BRIGHTNESS")


def record(cam, runtime, mat):
    vid = sl.ERROR_CODE.ERROR_CODE_FAILURE
    out = False
    while vid != sl.ERROR_CODE.SUCCESS and not out:
        filepath = input("Enter filepath name: ")
        vid = cam.enable_recording(filepath)
        print(repr(vid))
        if vid == sl.ERROR_CODE.SUCCESS:
            print("Recording started...")
            out = True
            print("Hit spacebar to stop recording: ")
            key = False
            while key != 32:  # for spacebar
                err = cam.grab(runtime)
                if err == sl.ERROR_CODE.SUCCESS:
                    cam.retrieve_image(mat)
                    cv2.imshow("ZED", mat.get_data())
                    key = cv2.waitKey(5)
                    cam.record()
        else:
            print("Help: you must enter the filepath + filename + SVO extension.")
            print("Recording not started.")
    cam.disable_recording()
    print("Recording finished.")
    cv2.destroyAllWindows()

def getImageArr( img_in , width , height , imgNorm="sub_mean" , odering='channels_first' ):
        img = img_in
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
            img_depth = img[:,:,0]
            img_depth = np.float32(cv2.resize(img_depth, ( width , height ))) / 32768 - 1
            img_depth = np.expand_dims(img_depth,2)
            if odering == 'channels_first':
                img_depth = np.rollaxis(img_depth, 2, 0)
            return img_depth
        elif imgNorm == "zed":
            img_depth = img
            

        if odering == 'channels_first':
            img = np.rollaxis(img, 2, 0)
        return img

if __name__ == "__main__":
    main()
