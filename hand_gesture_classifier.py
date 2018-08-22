#!/usr/bin/env python3
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
"""
Script to runs MobileNet-based classification model modified so it recognize the hand gestures.

index  label           function        pin_A pin_B pin_C
0      moutzas_in      de/activation   1     1     0 
1      moutzas_out     de/activation   1     1     0
2      namaste         forward         0     1     1
3      no_hands        no action       0     0     0
4      right           right           0     1     0
5      t               left            0     0     1 
6      thumbs_dn       backward        1     0     0
7      x               stop            1     0     1
8      none            no action       0     0     0

"""

import argparse
import time
import os
import numpy as np

from picamera import PiCamera
from PIL import Image

#from aiy.vision import inference
from aiy.vision.models import utils
from aiy.vision.models import face_detection
from aiy.vision.inference import CameraInference
from aiy.vision.inference import ImageInference
from aiy.vision.inference import ModelDescriptor
from gpiozero import Button
from aiy.vision.pins import BUTTON_GPIO_PIN
from aiy.vision.leds import Leds
from gpiozero import LED
from aiy.vision.pins import PIN_A
from aiy.vision.pins import PIN_B
from aiy.vision.pins import PIN_C
import aiy.toneplayer

# Initialize the GPIO pins A,B,C
pin_A = LED(PIN_A)
pin_B = LED(PIN_B)
pin_C = LED(PIN_C)
#gpio_logic = 'INVERSE'

# Initialize the buzzer
ready = [
        'C6q',
        'G5q',
        'E5q',
        'C5q',
    ]

activated = [
        'C5q',
        'C5q',
        'C5q',
        'G5q',
    ]

deactivated = [
        'G5q',
        'G5q',
        'G5q',
        'C5q',
    ]

player = aiy.toneplayer.TonePlayer(22)
player.play(*ready)

# Initialize the button (on the top of AIY Google Vision box)
button = Button(BUTTON_GPIO_PIN)

# Initialize LED (in the button on the top of AIY Google Vision box)
leds = Leds()
leds.update(Leds.rgb_off())

# Global variables
input_img_width = 1640
input_img_height = 1232
output_img_size = 160
faces_buffer_size = 40
hand_gesture_buffer_size = 5
threshold = 0.6

long_buffer_length = 10
short_buffer_length = 3
max_no_activity_period = 45

long_buffer = []
short_buffer = []
activation_index = 0
deactivation_index = 1


# Parameters used to collect training images in training_data_collector.py
x_shift_coef=0.0
y_shift_coef=1.3
scale = 2.0

RED = (0xFF, 0x00, 0x00)
GREEN = (0x00, 0xFF, 0x00)
BLUE = (0x00, 0x00, 0xFF)
PURPLE = (0xFF, 0x00, 0xFF)

# Blink LED
def blink_led(color=RED,period=1,num_blinks=5):
   for _ in range(num_blinks):
       leds.update(Leds.rgb_on(color))
       time.sleep(period/2)
       leds.update(Leds.rgb_off())
       time.sleep(period/2)

# Set status of GPIO pin
def pinStatus(pin,status,gpio_logic):
    if gpio_logic=='INVERSE':
        if status=='HIGH':
            pin.off()
        if status=='LOW':
            pin.on()
    else:
        if status=='HIGH':
            pin.on()
        if status=='LOW':
            pin.off()


# Send signal to pins
"""
index  label           function        pin_A pin_B pin_C
0      moutzas_in      de/activation   1     1     0 
1      moutzas_out     de/activation   1     1     0
2      namaste         forward         0     1     1
3      no_hands        no action       0     0     0
4      right           right           0     1     0
5      t               left            0     0     1 
6      thumbs_dn       backward        1     0     0
7      x               stop            1     0     1
8      none            no action       0     0     0
"""
def send_signal_to_pins(signal,gpio_logic):
    if signal == 0 or signal == 1:
        pinStatus(pin_A,'HIGH',gpio_logic)
        pinStatus(pin_B,'HIGH',gpio_logic)
        pinStatus(pin_C,'LOW',gpio_logic)
    elif signal == 2:
        pinStatus(pin_A,'LOW',gpio_logic)
        pinStatus(pin_B,'HIGH',gpio_logic)
        pinStatus(pin_C,'HIGH',gpio_logic)
    elif signal == 4:
        pinStatus(pin_A,'LOW',gpio_logic)
        pinStatus(pin_B,'HIGH',gpio_logic)
        pinStatus(pin_C,'LOW',gpio_logic)
    elif signal == 5:
        pinStatus(pin_A,'LOW',gpio_logic)
        pinStatus(pin_B,'LOW',gpio_logic)
        pinStatus(pin_C,'HIGH',gpio_logic)
    elif signal == 6:
        pinStatus(pin_A,'HIGH',gpio_logic)
        pinStatus(pin_B,'LOW',gpio_logic)
        pinStatus(pin_C,'LOW',gpio_logic)
    elif signal == 7:
        pinStatus(pin_A,'HIGH',gpio_logic)
        pinStatus(pin_B,'LOW',gpio_logic)
        pinStatus(pin_C,'HIGH',gpio_logic)
    else:
        pinStatus(pin_A,'LOW',gpio_logic)
        pinStatus(pin_B,'LOW',gpio_logic)
        pinStatus(pin_C,'LOW',gpio_logic)
    time.sleep(0.1)

# Buffer update and best guess estimation
def buffer_update(new_observation,buffer, buffer_length):
    buffer.append(new_observation)
    if len(buffer) > buffer_length:
        buffer.pop(0)
    best_guess = max(set(buffer), key=buffer.count)
    num_best_guess = buffer.count(best_guess)
    return best_guess,num_best_guess

# Print detected hand gestures
def print_hand_command(signal):
    if signal == 2:
        print('Forward')
    elif signal == 6:
        print('Backward')
    elif signal == 5:
        print('Left')
    elif signal == 4:
        print('Right')
    elif signal == 7:
        print('Stop')
    elif signal == 0:
        print('De/Activating')
    elif signal == 1:
        print('De/Activating')
    else:
        print('No hand signal detected')

# Read model labels
def read_labels(label_path):
    with open(label_path) as label_file:
        return [label.strip() for label in label_file.readlines()]

# Processes inference result and returns either
# (1) the index of most likely label with probability > threshold
# (2) index = len(labels) otherwise (i.e. none of hand gestures had a probablity > threshold
def process(result, labels, out_tensor_name, threshold):
    # MobileNet based classification model returns one result vector.
    assert len(result.tensors) == 1
    tensor = result.tensors[out_tensor_name]
    probs, shape = tensor.data, tensor.shape
    assert shape.depth == len(labels)
    pairs = [pair for pair in enumerate(probs) if pair[1] > threshold]
    pairs = sorted(pairs, key=lambda pair: pair[1], reverse=True)
    pair = pairs[0:1]
    if pair==[]:
        return len(labels)
    else:
        index, prob = pair[0]
        return index

# Error update when determining is face detection is stable
def error_update(obs_history,new_observation):
    obs_history.append(new_observation)
    if len(obs_history) > faces_buffer_size:
        obs_history.pop(0)
        split_point = int(faces_buffer_size*0.5)
        s = obs_history[0:split_point]
        e = obs_history[split_point:]
        s_mean = sum(s)/len(s)
        e_mean = sum(e)/len(e)
        error = 2 * abs(e_mean-s_mean) / (e_mean+s_mean)          
        return error
    else:
        return 10e6

# Check is face detection is stable
def face_detection_is_stable(x_err,y_err,w_err,h_err,cutoff=0.02):
    return (x_err < cutoff and y_err < cutoff and w_err < cutoff and h_err < cutoff)

# Check if box boundaries are within the limits
def image_boundary_check(box):
    left, upper, right, lower = box
    return (int(left)>0 and int(upper)>0 and 
              int(right)< input_img_width and int(lower) < input_img_height and 
              int(right)>int(left) and int(lower)>int(upper)) 

# Crop images (numpy array)  
def crop_np(box,image_np):
    left, upper, right, lower = box
    box = (int(left), int(upper), int(right), int(lower))
    if image_boundary_check(box):
        cropped_image_np = image_np[int(upper):(int(lower)+1), int(left):(int(right)+1):,:]
        cropped_image = Image.fromarray(cropped_image_np, 'RGB')
        cropped_image = cropped_image.resize((output_img_size, output_img_size), Image.ANTIALIAS)
        return cropped_image
    else:
        return None

# Incoming boxes are of the form (x, y, width, height). Scale and
# transform to the form (x1, y1, x2, y2).
def transform(bounding_box):
    x, y, width, height = bounding_box
    return (x, y, x + width,y + height)

# Determine location/size of hand box given the location/size of detected face box
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

# Detect (stable) face
def detect_face():
    with CameraInference(face_detection.model()) as camera_inference:
        counter = 1
        x_history, y_history, w_history, h_history = [], [], [], []
        for result in camera_inference.run():
            check_termination_trigger()
            faces = face_detection.get_faces(result)
            face = select_face(faces)
            if face:
                x,y,w,h = face.bounding_box
                x_err = error_update(x_history,x)
                y_err = error_update(y_history,y)
                w_err = error_update(w_history,w)
                h_err = error_update(h_history,h)

                if face_detection_is_stable(x_err,y_err,w_err,h_err,cutoff=0.03):
                    face_box =(int(sum(x_history)/len(x_history)),int(sum(y_history)/len(y_history)),
                          int(sum(w_history)/len(w_history)),int(sum(h_history)/len(h_history)))
                    break
                counter += 1
        return face_box  

# Shutdown Google Vision  AIY kit if termination trigger is activated (button pressed) 
def check_termination_trigger():
    if button.is_pressed:
        print('Terinating session...')
        leds.update(Leds.rgb_off())
        time.sleep(5)
        os.system("sudo shutdown -h now") 

# Determine locations and size of hand box
def determine_hand_box_params(face_box):
    face_box = transform(face_box)
    hand_box_params = hand_box(face_box)
    return hand_box_params

# Capture raw images into numpy array and crop hands images
def capture_hands_image(camera,hand_box_params):
    hands_image = []
    image = np.empty((1664 * 1232 * 3,), dtype=np.uint8)
    camera.capture(image, 'rgb')
    image = image.reshape((1232, 1664, 3))
    image = image[:1232, :1640, :]
    hand_cropped = crop_np(hand_box_params,image)
    if hand_cropped:
        hands_image.append(hand_cropped)
    return hands_image

# Classify hand gestures
def classify_hand_gestures(img_inference,hands_images,model,labels,output_layer,threshold):
    for hand in hands_images:
        result = img_inference.run(hand)
        model_output = process(result, labels, output_layer, threshold)
        return model_output



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        required=True,
        help='Path to converted model file that can run on VisionKit.')
    parser.add_argument(
        '--label_path',
        required=True,
        help='Path to label file that corresponds to the model.')
    parser.add_argument(
        '--input_height', type=int, required=True, help='Input height.')
    parser.add_argument(
        '--input_width', type=int, required=True, help='Input width.')
    parser.add_argument(
        '--input_layer', required=True, help='Name of input layer.')
    parser.add_argument(
        '--output_layer', required=True, help='Name of output layer.')
    parser.add_argument(
        '--num_frames',
        type=int,
        default=-1,
        help='Sets the number of frames to run for, otherwise runs forever.')
    parser.add_argument(
        '--input_mean', type=float, default=128.0, help='Input mean.')
    parser.add_argument(
        '--input_std', type=float, default=128.0, help='Input std.')
    parser.add_argument(
        '--input_depth', type=int, default=3, help='Input depth.')
    parser.add_argument(
        '--threshold', type=float, default=0.6,
        help='Threshold for classification score (from output tensor).')
    parser.add_argument(
        '--preview',
        action='store_true',
        default=False,
        help='Enables camera preview in addition to printing result to terminal.')
    parser.add_argument(
        '--gpio_logic',
        default='NORMAL',
        help='Indicates if NORMAL or INVERSE logic is used in GPIO pins.')
    parser.add_argument(
        '--show_fps',
        action='store_true',
        default=False,
        help='Shows end to end FPS.')
    args = parser.parse_args()


    # Model & labels
    model = ModelDescriptor(
        name='mobilenet_based_classifier',
        input_shape=(1, args.input_height, args.input_width, args.input_depth),
        input_normalizer=(args.input_mean, args.input_std),
        compute_graph=utils.load_compute_graph(args.model_path))
    labels = read_labels(args.label_path)

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

        while True:
            while True:
                long_buffer = []
                short_buffer = []
                pinStatus(pin_A,'LOW',args.gpio_logic)
                pinStatus(pin_B,'LOW',args.gpio_logic)
                pinStatus(pin_C,'LOW',args.gpio_logic)
                leds.update(Leds.rgb_on(GREEN))
                face_box = detect_face()
                hand_box_params = determine_hand_box_params(face_box)
                if image_boundary_check(hand_box_params):
                    break

            # Start hand classifier
            is_active = False
            leds.update(Leds.rgb_on(PURPLE))
            start_timer = time.time()
            with ImageInference(model) as img_inference:
                while True:
                    check_termination_trigger()
                    if is_active:
                        leds.update(Leds.rgb_on(RED))
                    hands_image = capture_hands_image(camera,hand_box_params)
                    output = classify_hand_gestures(img_inference,hands_image,model=model,labels=labels,output_layer=args.output_layer,threshold = args.threshold)

                    short_guess, num_short_guess = buffer_update(output,short_buffer,short_buffer_length)
                    long_guess, num_long_guess = buffer_update(output,long_buffer,long_buffer_length)

                    # Activation of classifier                  
                    if (long_guess == activation_index or long_guess == deactivation_index) and not is_active and num_long_guess >= (long_buffer_length - 3):
                        is_active = True
                        leds.update(Leds.rgb_on(RED))
                        player.play(*activated)
                        send_signal_to_pins(activation_index,args.gpio_logic)
                        long_buffer = []                      
                        num_long_guess = 0                     
                        time.sleep(1)

                    # Deactivation of classifier (go back to stable face detection)                  
                    if (long_guess == activation_index or long_guess == deactivation_index) and is_active and num_long_guess >= (long_buffer_length - 3):
                        is_active = False
                        leds.update(Leds.rgb_off())
                        player.play(*deactivated)
                        long_buffer = []
                        num_long_guess = 0                     
                        send_signal_to_pins(deactivation_index,args.gpio_logic)                      
                        time.sleep(1)
                        break

                    # If not activated within max_no_activity_period seconds, go back to stable face detection
                    if not is_active:
                        timer = time.time()-start_timer
                        if timer >= max_no_activity_period:
                            leds.update(Leds.rgb_off())
                            send_signal_to_pins(deactivation_index,args.gpio_logic)                      
                            time.sleep(1)
                            break
                    else:
                        start_timer = time.time()  

                        # Displaying classified hand gesture commands
                        if num_short_guess >= (short_buffer_length-1) and is_active:
                            print_hand_command(short_guess)
                            send_signal_to_pins(short_guess,args.gpio_logic)
 
        camera.stop_preview()

if __name__ == '__main__':
    main()
