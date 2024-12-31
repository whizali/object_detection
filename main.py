import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox

# Open webcam
# the cv2.VideoCapture intializes the webcam for reading video frames. 0 is the default camera index
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    #    ret, frame = cap.read() 
    #reads frame from the webcam, stroing that frame in the frame variable
    # everytime the frame is successfuly captured the ret is true. ret is a boolean value
    # and false when no frames are captured causing the loop to break.

    if not ret:
        break

    # Perform object detection
    # cv.detect_common_objects(frame) is the main functiion that detects
    # objects in given frame. it uses a pre-trained deep learning model  YOLO to detect obejcts and people, 
    # bbox is a list of bounding boxes around detected objects. each bounding box is represented
    # by four values x1, y1, x2, y2 which represent the top-left and bottom-right corners of the rectangle.
    # labels: A list of labels corresponding to the detected objects (e.g., "person", "car", "dog").
    # confidences: A list of confidence scores (probabilities) for each detected object, indicating how confident the model is in its classification.
    bbox, labels, confidences = cv.detect_common_objects(frame)


    # Draw bounding boxes
    output_frame = draw_bbox(frame, bbox, labels, confidences)
# this function returns the frame with boudnig boxes for each object detected.

    # Display the output frame
    cv2.imshow("Real-Time Object Detection", output_frame)

    # Break the loop with the 'q' key after 1 millisecond of key event
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  # stop video capture
cv2.destroyAllWindows() #closes all opencv windwos opened


# code for realtime object detection system using the webcam
# uses cvlib library for object detection
# opencv cv2 for video processing
# cv2: OpenCV library, widely used for computer vision tasks like 
# video processing, image manipulation, and object detection.
# cvlib: a library for object detecion built on top of tensorflow and keras
# A simple and efficient library for object detection built on top of TensorFlow and Keras. It offers an easy-to-use API for common
# computer vision tasks like object detection, face detection, etc.
# draw_bbox: A function from cvlib that helps in drawing bounding boxes on an image or frame.


# cvlib uses pre-trained models. specifically cv.detect_common_objects() uses YOLO
# YOLO -- YOLO is a realtime object detection system for it's high processing speed and accuracy
# cvlib uses a YOLO model trained on COCO dataset. 
# YOLOv3 is trained on a large dataset called COCO (Common Objects in Context). 
# The common objects detected include people, cars, animals, and other everyday items.
# YOLO is a CNN that can detect multiple objetcs in a single frame, and cvlib integrates it for realtime detection

#  OpenCV is a comprehensive computer vision library that offers low level tools such as image and video processing
# while cvlib is a wrapper libray which is absracted from the diverse OpenCV, designed to simplify common computer vision tasks like 
# object detection and face detection using pre-trained models. It abstracts away much of the complexity of OpenCV.

#the coco.names contains the set of classes we can perform object detection on

# the yolov3.cfg is a file that defines the architecture of the neural network
# the yolov3.cfg defines certain layers of yolov3 neural network.
#[net] -- the training parameters of the network
#[convolutional] -- defines convolutional layers and related paramters
#[maxpool] -- pooling layers for downsampling

# it specifies different types of layer like convolutional layer, filters and pooling layers.
# and how they are stacked together. 
# each layer has set of parameters that control its behaviour. 
# such as number of filters, -- kernel size and strides,


# the config file is paired with .weights file
# .weights file contains the learned parameters by the model for the 80 classes in 
# coco dataset. .weight file is pre trained on dataset like COCO or imagenet.


