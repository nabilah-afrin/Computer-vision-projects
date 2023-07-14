import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#Visiulazation
import matplotlib.pyplot as plt
#image processing
import cv2
#extracting zippped file
import tarfile
#systems
import os
from time import time
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import loguniform
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler


!pip install utils
!wget https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
!wget https://pjreddie.com/media/files/yolov3.weights

lfw_people = fetch_lfw_people(min_faces_per_person=70,  color = True, resize=0.4)
print("Total dataset size:")
print("n_samples: %d" % lfw_people.data.shape[0])
print("n_features: %d" % lfw_people.data.shape[1]) #resized image's pixel value
print("n_classes: %d" % len(lfw_people.target_names))

X = lfw_people.data
# the label to predict is the id of the person
y = lfw_people.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

# Reshape the images to the appropriate size expected by the YOLO model
IMG_WIDTH = 416
IMG_HEIGHT = 416

# Function to resize and preprocess each image
def preprocess_image(image):
    # Resize the image
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))

    # Normalize pixel values to range 0-1
    image = image / 255.0

    return image

# Preprocess and reshape the images in the training set
X_train = np.array([preprocess_image(image.reshape(lfw_people.images.shape[1:])) for image in X_train])

# Preprocess and reshape the images in the test set
X_test = np.array([preprocess_image(image.reshape(lfw_people.images.shape[1:])) for image in X_test])

# Print the shape of the preprocessed training and test sets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
# Reshape the images to the appropriate size expected by the YOLO model
IMG_WIDTH = 416
IMG_HEIGHT = 416

# Function to resize and preprocess each image
def preprocess_image(image):
    # Resize the image
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))

    # Normalize pixel values to range 0-1
    image = image / 255.0

    return image

# Preprocess and reshape the images in the training set
X_train = np.array([preprocess_image(image.reshape(lfw_people.images.shape[1:])) for image in X_train])

# Preprocess and reshape the images in the test set
X_test = np.array([preprocess_image(image.reshape(lfw_people.images.shape[1:])) for image in X_test])

# Print the shape of the preprocessed training and test sets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

import numpy as np
import matplotlib.pyplot as plt

# Plot 15 random images from the X_train dataset
n_row = 3
n_col = 5
plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
for i in range(n_row * n_col):
    # Get a random index
    index = np.random.randint(0, X_train.shape[0])
    # Get the image at that index
    image = X_train[index].reshape(lfw_people.images.shape[1:])
    # Get the label at that index
    label = lfw_people.target_names[y_train[index]]
    # Plot the image and label
    plt.subplot(n_row, n_col, i + 1)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.title(label, size=12)
    plt.axis('off')
plt.show()
