import os
import numpy as np
from numpy import mean
from numpy import std

import cv2
import imgaug as aug
import imgaug.augmenters as iaa

import tensorflow as tf
from keras import backend as K
from tensorflow.keras.utils import to_categorical
from keras.models import Model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Input


# ==== #

# Auxiliary Functions
def adjust_gamma(image, gamma=1.0):
    #lookup table with the adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

# ==== #

# Data Augmentation - Data Generation Function
def enrich_normal(normal_train_df, outdir, extra):
    os.chdir(outdir)

    # Augmentation sequence 
    seq = iaa.OneOf([
        iaa.Fliplr(), # horizontal flips
        iaa.Affine(rotate=20), # rotation
        iaa.Affine(rotate=-15), # rotation
        iaa.Multiply((1.1, 1.5)), #random brightness
        iaa.Sharpen(alpha=0.5)]) # sharpening

    # Get total number of samples in the data
    n = len(normal_train_df)
    
    count = 1
    for i in range(n):
      img_name = normal_train_df.iloc[i]['image']
      
      # read the image and resize
      img = cv2.imread(str(img_name))
      img = cv2.resize(img, (224,224))
      orig_img = img.copy()
      name = "NORMAL_EXTENDED_" + str(count) + ".jpeg"
      cv2.imwrite(name, orig_img)
      count +=1

      # Generating additional normal cases
      for j in range(extra):

        aug_img = seq.augment_image(img)
        name = "NORMAL_EXTENDED_" + str(count) + ".jpeg"
        cv2.imwrite(name, aug_img)
        count +=1


# ==== #

# Pneumonia Dataset Pre-processing Functions

# Basic Pre-processing
def base_pre(data):
  new_data = []
  new_labels = []

  normal_images = data.loc[data['label'] == 0,'image']
  pneumonia_images = data.loc[data['label'] == 1,'image']

  for img_name in normal_images:
    img = cv2.imread(str(img_name))
    # Resizing
    img = cv2.resize(img, (112,112))
    # Channel correction
    if img.shape[2] ==1:
        img = np.dstack([img, img, img])
    # Grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Normalization
    img = img.astype(np.float32)/255.
    label = to_categorical(0, num_classes=2)
    new_data.append(img)
    new_labels.append(label)


  for img_name in pneumonia_images:
    img = cv2.imread(str(img_name))
    # Resizing
    img = cv2.resize(img, (112,112))
    # Channel correction
    if img.shape[2] == 1:
      img = np.dstack([img, img, img])
    # Grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Normalization
    img = img.astype(np.float32)/255.
    label = to_categorical(1, num_classes=2)
    new_data.append(img)
    new_labels.append(label)

  # Convert the list into numpy arrays
  new_data = np.array(new_data)
  new_labels = np.array(new_labels)

  return new_data, new_labels


# Simple 2-step pre-processing
def pre_simple(data):
  new_data = []
  new_labels = []

  normal_images = data.loc[data['label'] == 0,'image']
  pneumonia_images = data.loc[data['label'] == 1,'image']

  for img in normal_images:
    img = cv2.imread(str(img))

    # Resizing
    img = cv2.resize(img, (112,112))
    # Channel correction
    if img.shape[2] ==1:
        img = np.dstack([img, img, img])
    # Grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # == Extra Preprocessing == #
    img = (255-img)
    img = cv2.add(img, -30)
    # == #

    # Normalization
    img = img.astype(np.float32)/255.
    label = to_categorical(0, num_classes=2)
    new_data.append(img)
    new_labels.append(label)


  for img in pneumonia_images:
    img = cv2.imread(str(img))

    # Resizing
    img = cv2.resize(img, (112,112))
    # Channel correction
    if img.shape[2] ==1:
        img = np.dstack([img, img, img])
    # Grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # == Extra Preprocessing == #
    img = (255-img)
    img = cv2.add(img, -30)
    # == #

    # Normalization
    img = img.astype(np.float32)/255.
    label = to_categorical(1, num_classes=2)
    new_data.append(img)
    new_labels.append(label)

  # Convert the list into numpy arrays
  new_data = np.array(new_data)
  new_labels = np.array(new_labels)

  return new_data, new_labels


# Pre-processing with Edge Detector
def pre_canny(data):
  new_data = []
  new_labels = []

  normal_images = data.loc[data['label'] == 0,'image']
  pneumonia_images = data.loc[data['label'] == 1,'image']

  for img in normal_images:
    img = cv2.imread(str(img))

    # Resizing
    img = cv2.resize(img, (112,112))
    # Channel correction
    if img.shape[2] == 1:
      img = np.dstack([img, img, img])
    # Grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # == Extra Preprocessing == #
    img = adjust_gamma(img, gamma=1.2)
    img = cv2.Canny(img, 40, 40)
    img = (255-img)
    # == #

    # Normalization
    img = img.astype(np.float32)/255.
    label = to_categorical(0, num_classes=2)
    new_data.append(img)
    new_labels.append(label)


  for img in pneumonia_images:
    img = cv2.imread(str(img))

    # Resizing
    img = cv2.resize(img, (112,112))
    # Channel correction
    if img.shape[2] == 1:
      img = np.dstack([img, img, img])
    # Grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # == Extra Preprocessing == #
    img = adjust_gamma(img, gamma=1.2)
    img = cv2.Canny(img, 40, 40)
    img = (255-img)
    # == #
    
    # Normalization
    img = img.astype(np.float32)/255.
    label = to_categorical(1, num_classes=2)
    new_data.append(img)
    new_labels.append(label)

  # Convert the list into numpy arrays
  new_data = np.array(new_data)
  new_labels = np.array(new_labels)

  return new_data, new_labels


# Pre-processing with Histogram Normalization
def pre_histo(data):
  new_data = []
  new_labels = []

  normal_images = data.loc[data['label'] == 0,'image']
  pneumonia_images = data.loc[data['label'] == 1,'image']

  for img in normal_images:
    img = cv2.imread(str(img))
    
    # == Extra Preprocessing == #
    hist, bins = np.histogram(img.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()
    cdf_m = np.ma.masked_equal(cdf,0) 
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf2 = np.ma.filled(cdf_m,0).astype('uint8')

    img = cdf2[img]
    img = (255-img)
    img = cv2.add(img, -70)
    # == #

    # Resizing
    img = cv2.resize(img, (112,112))
    # Channel correction
    if img.shape[2] ==1:
        img = np.dstack([img, img, img])
    # Grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Normalization
    img = img.astype(np.float32)/255.
    label = to_categorical(0, num_classes=2)
    new_data.append(img)
    new_labels.append(label)


  for img in pneumonia_images:
    img = cv2.imread(str(img))
    # Resizing
    img = cv2.resize(img, (112,112))
    # Channel correction
    if img.shape[2] == 1:
        img = np.dstack([img, img, img])
    # Grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # == Extra Preprocessing == #
    hist, bins = np.histogram(img.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()
    cdf_m = np.ma.masked_equal(cdf,0) 
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf2 = np.ma.filled(cdf_m,0).astype('uint8')

    img = cdf2[img]
    img = (255-img)
    img = cv2.add(img, -70)
    # == #

    # Normalization
    img = img.astype(np.float32)/255.
    label = to_categorical(1, num_classes=2)
    new_data.append(img)
    new_labels.append(label)

  # Convert the list into numpy arrays
  new_data = np.array(new_data)
  new_labels = np.array(new_labels)

  return new_data, new_labels

# ==== #

# Model Construction

def build_model():

    inp = Input(shape=(112,112,1), name='ImageInput')

    hid = Conv2D(32, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform', input_shape=(112,112,1), name='Conv1_1')(inp)
    hid = Conv2D(32, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform', name='Conv1_2')(inp)
    hid = MaxPooling2D((2,2), name='MaxPool1')(hid)
    hid = Dropout(0.2)(hid)

    hid = Conv2D(64, (3,3), activation = "relu", padding = 'same', kernel_initializer = 'he_uniform', name='Conv2_1')(hid)
    hid = Conv2D(64, (3,3), activation = "relu", padding = 'same', kernel_initializer = 'he_uniform', name='Conv2_2')(hid)
    hid = MaxPooling2D((2,2), name='MaxPool2')(hid)
    hid = Dropout(0.2)(hid)

    # hid = Conv2D(128, (3,3), activation = "relu", padding = 'same', kernel_initializer = 'he_uniform', name='Conv3_1')(hid)
    # hid = Conv2D(128, (3,3), activation = "relu", padding = 'same', kernel_initializer = 'he_uniform', name='Conv3_2')(hid)
    # hid = MaxPooling2D((2,2), name='MaxPool3')(hid)
    # hid = Dropout(0.2)(hid)

    hid = Flatten()(hid)
    hid = Dense(64, activation = 'relu', kernel_initializer = 'he_uniform')(hid)
    hid = Dropout(0.2)(hid)

    out = Dense(2, activation = 'softmax')(hid)

    model = Model(inputs = inp, outputs=out)

    return model

