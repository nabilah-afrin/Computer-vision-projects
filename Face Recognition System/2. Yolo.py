import cv2
import numpy as np
import os
import utils
from utils import *

#####################################################################
# Set the paths and configurations
model_cfg = '/kaggle/working/yolov3.cfg'
model_weights = "/kaggle/working/yolov3.weights"
output_dir = '/kaggle/working/'

# check outputs directory
if not os.path.exists(output_dir):
    print('==> Creating the {} directory...'.format(output_dir))
    os.makedirs(output_dir)
else:
    print('==> Skipping creating the {} directory...'.format(output_dir))

# Load the YOLO model
net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Process each image in X_train
for i, image in enumerate(X_train):
    frame = image.reshape(lfw_people.images.shape[1:])  # Reshape the image to the appropriate size

    # Create a 4D blob from the frame
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT), [0, 0, 0], 1, crop=False)

    # Set the input to the network
    net.setInput(blob)

    # Run the forward pass to get the output of the output layers
    outs = net.forward(get_outputs_names(net))

    # Perform post-processing to remove low confidence bounding boxes
    faces = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)

    print('[i] ==> # detected faces: {}'.format(len(faces)))
    print('#' * 60)

    # Draw bounding boxes and information on the frame
    for face in faces:
        (x, y, w, h) = face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Save the output image to file
    output_file = 'output_{}.jpg'.format(i)
    cv2.imwrite(os.path.join(output_dir, output_file), frame)

    # Display the frame with bounding boxes
    cv2.imshow('Face Detection using YOLOv3', frame)
    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        print('[i] ==> Interrupted by user!')
        break

cv2.destroyAllWindows()
print('==> All done!')
print('***********************************************************')
