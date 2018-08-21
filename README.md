# Hand Command Recognizer on Google Vision AIY kit

## Objective 

This project demonstrates how, by creating a training set of only 1,500 images from scratch, carefully selecting a search region and applying Transfer Learning technique, one can build and deploy on Edge AI device - Google Vision API kit – the model which reliably recognizes simple hand gestures. A fairly accurate model with a latency of 1-2 seconds runs on the Google Vision box and does not require any access to Internet or Cloud. It can be used to control your mobile robot, replace your TV remote control or for many other applications. The described approach of carefully selecting the search region, collecting a relatively small number of customized training images and re-training open-sourced Deep Learning models to create a model for a specific task (e.g. the model which controls access to the facilities by recognizing faces of the company's employees) can be applied to create numerous and diverse applications.

## Installation

1. Buy Google Vision AIY kit and assemble it following [these instructions](https://aiyprojects.withgoogle.com/vision)
2. Power the assembled Google Vision AIY kit
3. Stop and disable joy_detector_demo application which is set to start automatically after the booting

 ```
 sudo systemctl stop joy_detection_demo.service
 sudo systemctl disable joy_detection_demo.service
 ```
4. Update OS

 ```
 sudo apt-get update
 sudo apt-get upgrade
 ```

5. Clone the github repository with hand gesture classifier and navigate to the project folder

 ```
 cd src/examples/vision 
 git clone https://github.com/dvillevald/hand_gesture_classifier.git
 cd hand_gesture_classifier
 ```

6. Start hand gesture classifier

 ```
 ./hand_gesture_classifier.py  \
  --model_path ~/AIY-projects-python/src/examples/vision/hand_gesture_classifier/hand_gesture_classifier.binaryproto \
  --label_path ~/AIY-projects-python/src/examples/vision/hand_gesture_classifier/hand_gesture_labels.txt  \
  --input_height 160   \
  --input_width 160    \
  --input_layer input  \
  --output_layer final_result
 ```

  **Important:** It seems that on some Google Vision AIY kits the logic of GPIO pins is inversed - pin.off() changes pin status to HIGH and pin.on() - to LOW. If you observe that your hand command classifier works but shows incorrect commands (e.g. displays *right* instead of *left*) then add the following line to the command above:

  ```
   --gpio_logic INVERSE
 ```

![ScreenShot](images/launch_application.png "Launching App")

## How hand gesture classifier works

### Step 1. Face detection

LED = ![#008000](https://placehold.it/15/008000/000000?text=+)

Once you start the application, it launches *face/joy detector* pre-installed on the Google Vision AIY kit which tries to detect the human face and determine the size and location of the bounding box around it. During this step the LED on the top of the Google Vision box is **GREEN**. 

Once the face is reliably detected on several dozens of frames, application uses the size and the location of the face bounding box to determine the size and location of the the chest area (called hand box hereinafter) where the hand gestures are expected to be displayed:

<img width="245" height="326" src="images/hand_box.png">

There are several advantages of this approach:

 1. The search space is greatly reduced to only include the chest area which significantly improves the latency of the detector. 
  
 2. Displaying hand gestures in the chest area improves the quality of the hand detector as a user has a high degree of control of the image background (one's t-shirt) and because the number and diversity of possible backgrounds is greatly reduced (to the number of t-shorts and sweaters in user's wardrobe.) Because of that one does not need a large data set to build a model which makes a fairly accurate predictions so it takes less time to collect the training data and to train your model.

A couple of practical suggestions if face detection takes longer than 10-15 seconds:
 - It is possible that the face detector cannot detect your face if, for example, your wear a particular glasses. Before starting this application, make sure Google's face/joy detector can detect your face (e.g. reacts to your smile.)
 - Make sure you stand still during this step – to better estimate the parameters of the face bounding box the face detector compares face box parameters on several frames and will only proceed to the next step when they are stable. 
 - Move further away from the camera – it is possible that you are too close and while your face is detected, your chest area does not fit into the image. The optimal distance from the Google Vision kit is about 10 feet (3 meters.)  

Once you face is reliably detected the LED on the top of Google Vision box turns **PURPLE**, face detection stops and the hand gesture recognizer is loaded ready to be activated. 

### Step 2. Activating hand gesture recognizer

LED = ![#800080](https://placehold.it/15/800080/000000?text=+)

To make sure the application does not react to the noise, any of two hand commands (Palms_in and Palms_out) are used to either activate and deactivate the application. To activate hand gesture recognizer, display one of these two commands in your chest area for 2-5 seconds.

Activate: 
| 
