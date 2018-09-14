#!/usr/bin/env python3
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Hand gesture training dataset collector.

Purpose: In case you want to train the model on your own data, training_data_collector helps to collect the training images. 

How it works: The session has two stages.

Stage #1. Raw image collection  
  During this stage you would display in front of a camera the hand gesture for a specific 
  label (which name you specify in the command-line argument --label.) 
  The code runs continuous face detection on the VisionBonnet and, once the face is detected, 
  it selects the largest face (if several were detected),
  determines the size and location of the hand box, makes a snapshot, 
  saves the image on the disk in the folder /Raw and saves the location of
  the hand box on each image in the list hand_boxes_locations. This cycle is then repeated by --num_images times.
  Raw image collection will start when the led (in the button on the top of AIY Google Vision box) turns RED.

Stage #2: Processing raw images and storing training images
  Once all raw images are collected (and camera preview switches off), each raw image is cropped, 
  resized to 160x160 pixels and saved in the subfolder of the training_data folder with the name of the specified label. 
  (Example: if you selected "--label no_hands" in the command line, the images will be stored in folder /training_data/no_hands) 
  If this folder does not exist, it will be created. The folder with raw images (/Raw) will be deleted at the end of this stage. 
  The processing and storing of training images will start when the led (in the button on the top of AIY Google Vision box) turns BLUE.

Practical suggestions during the collection of training data:
(1) Make sure that both your head box and your chest box fit in the image. Otherwise the training image will not be saved. 
(2) Select a reasonable number of images (100-200) to capture within the session so you don't get tired. 
    I recorded 200 images which took about 2-3 minutes to collect.
(3) Make sure that the hand gesture you are recording match the label you specified in –label argument.
(4) Vary position of your body and hands slightly during a session (moving closer or further away from the camera, 
    slightly rotating your body and hands, etc.) to make your training data set more diverse.
(5) Record the images in 2-5 different environments. For example, you may record 200 images for each label wearing a red T-short 
    in a bright room (environment #1), then record another 200 images for each label wearing a blue sweater 
    in a darker (environment #2), etc.
(6) Make sure that in each environment you record the training images for all labels so there is no correlation 
    between the particular environment and specific hand command. 
(7) Capture images in the room bright enough so you hands are visible. For the same reason 
    use T-shorts/sweaters with plain and darker colors.
(8) Review the images you collected and remove the bad ones if you see any.

Parameters:
--label: a string specifying the name of the class (label) of the collected images
--num_images: number of images to record during the session    

Example:
training_data_collector.py --label no_hands --num_images 100
"""

import argparse
import time
import os
import random
import string
import numpy as np

from PIL import Image
from aiy.vision.inference import CameraInference
from aiy.vision.models import face_detection
from picamera import PiCamera, array
from aiy.vision.leds import Leds

RED = (0xFF, 0x00, 0x00)
BLUE = (0x00, 0x00, 0xFF)

# Led (button on the top of the AIY Google Vision box)
leds = Leds()

# Global variables
path_to_training_folder = "training_data/"
input_img_width = 1640
input_img_height = 1232
output_img_size = 160

# Parameters of hand box (determined with hand_box_locator.py)
x_shift_coef=0.0
y_shift_coef=1.3
scale=2.0

# Create training_data folder if it does not exist
if not os.path.exists(path_to_training_folder):
    os.makedirs(path_to_training_folder)

# Check if box boundaries (in pixels) are within the limits
def image_boundary_check(box):
    left, upper, right, lower = box
    return (int(left)>0 and int(upper)>0 and 
              int(right)< input_img_width and int(lower) < input_img_height and 
              int(right)>int(left) and int(lower)>int(upper)) 

# Crop raw images, resize the results and store final training images in 
# subfolder (with name of a class) of training_data folder 
def crop_and_store_images(label,hand_box,image):
    output_folder = path_to_training_folder + label.lower() + '/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    left, upper, right, lower = hand_box
    box = (int(left), int(upper), int(right), int(lower))
    if image_boundary_check(box):
        time.sleep(0.1)
        random_string = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(10))
        cropped_img_name = output_folder + label.lower() + '_' + random_string + ".jpg"
        cropped_image = image.crop(box)
        cropped_image = cropped_image.resize((output_img_size, output_img_size), Image.ANTIALIAS)
        cropped_image.save(cropped_img_name)

# Transform incoming boxes from (x, y, width, height) to format (x1, y1, x2, y2) 
# where (x1,y1) are the coordinates of the upper left box corner (i.e. (x,y))
#       (x2,y2) are the coordinates of the lower right box corner (i.e. (x+width, y+height))
def transform(bounding_box):
    x, y, width, height = bounding_box
    return (x, y, x + width,y + height)

# Determines location/size of hand box given the location/size of detected head box
# Use the values of x_shift_coef, y_shift_coef, scale you found earlier with hand_box_locator.py
def hand_box(face_box, x_shift_coef=x_shift_coef, y_shift_coef=y_shift_coef, scale=scale):
    x1, y1, x2, y2 = face_box
    x_center = int(0.5 * (x2 + x1))
    y_center = int(0.5 * (y2 + y1))
    face_box_width = x2 - x1
    face_box_height = y2 - y1
    x_shift = int(x_shift_coef * face_box_width)
    y_shift = int(y_shift_coef * face_box_height)
    x1_hand_box = x_center + x_shift - int(0.5 * scale * face_box_width)
    x2_hand_box = x_center + x_shift + int(0.5 * scale * face_box_width)
    y1_hand_box = y_center + y_shift - int(0.5 * scale * face_box_height)
    y2_hand_box = y_center + y_shift + int(0.5 * scale * face_box_height)
    return (x1_hand_box, y1_hand_box, x2_hand_box, y2_hand_box) 

# Select the face with the largest width (others will be ignored)
def select_face(faces):
    if len(faces) > 0:
        max_width = 0
        for face in faces:
            width = face.bounding_box[2]
            if width > max_width:
                max_width = width
                face_selected = face
        return face_selected
    else:
        return None

def main():
    """Face detection camera inference example."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--label',
        '-lbl',
        type=str,
        dest='label',
        required=True,
        help='Specifies the class (label) of training images (e.g. no_hangs).')

    parser.add_argument(
        '--num_images',
        '-nimg',
        type=int,
        dest='num_images',
        default=10,
        help='Sets the number of training images to make.')

    args = parser.parse_args()

    with PiCamera() as camera:
        # Forced sensor mode, 1640x1232, full FoV. See:
        # https://picamera.readthedocs.io/en/release-1.13/fov.html#sensor-modes
        # This is the resolution inference run on.
        camera.sensor_mode = 4

        # Scaled and cropped resolution. If different from sensor mode implied
        # resolution, inference results must be adjusted accordingly. This is
        # true in particular when camera.start_recording is used to record an
        # encoded h264 video stream as the Pi encoder can't encode all native
        # sensor resolutions, or a standard one like 1080p may be desired.
        camera.resolution = (1640, 1232)

        # Start the camera stream.
        camera.framerate = 30
        camera.start_preview()

        # Stage #1: Capture and store raw images
        # Create foler to store raw images
        path_to_raw_img_folder = path_to_training_folder + 'raw/' 
        if not os.path.exists(path_to_raw_img_folder):
            os.makedirs(path_to_raw_img_folder)
        time.sleep(2)

        # Create list to store hand boxes location for each image
        hand_boxes_locations = []
 
        with CameraInference(face_detection.model()) as inference:
            leds.update(Leds.rgb_on(RED))
            time.sleep(3)
            counter = 1
            start = time.time()

            for result in inference.run():
                faces = face_detection.get_faces(result)
                face = select_face(faces)
                if face:
                    if counter > args.num_images:
                        break
                    face_box = transform(face.bounding_box)
                    hands = hand_box(face_box)

                    # Capture raw image 
                    img_name = path_to_raw_img_folder + 'img' + str(counter) + '.jpg'
                    camera.capture(img_name)
                    time.sleep(0.2)

                    # Record position of hands
                    hand_boxes_locations.append([counter,hands])

                    
                    print('Captured ',str(counter)," out of ",str(args.num_images))
                    counter += 1
            print('Stage #1: It took',str(round(time.time()-start,1)), 'sec to record',str(args.num_images),'raw images')    
        camera.stop_preview()

        # Stage #2: Crop training images from the raw ones and store them in class (label) subfolder
        leds.update(Leds.rgb_on(BLUE))
        start = time.time()
        for i,entry in enumerate(hand_boxes_locations):
            img_number = entry[0]
            hands = entry[1]
            raw_img_name = path_to_raw_img_folder + 'img' + str(img_number) + '.jpg'
            if os.path.isfile(raw_img_name):
                raw_image = Image.open(raw_img_name)
                crop_and_store_images(args.label,hands,raw_image)
                raw_image.close()  
                time.sleep(0.5)
                os.remove(raw_img_name)
            print('Processed ',str(i+1)," out of ",str(args.num_images))
        print('Stage #2: It took ',str(round(time.time()-start,1)), 'sec to process',str(args.num_images),'images')    
        time.sleep(3)
        # Delete empty folder for raw images 
        if os.listdir(path_to_raw_img_folder) == []:
            os.rmdir(path_to_raw_img_folder)
        leds.update(Leds.rgb_off())

if __name__ == '__main__':
    main()
