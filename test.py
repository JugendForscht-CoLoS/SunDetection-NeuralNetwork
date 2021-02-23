from losses.losses import TverskyLoss
import sun_dataset.sun_dataset

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
import tensorflow_datasets as tfds
import sun_dataset.sun_dataset

from tensorflow.keras.layers import MaxPooling2D, UpSampling2D, Conv2D, Conv2DTranspose, Input, Add
from tensorflow.keras.models import Model

OUTPUT_CHANNELS = 1
ACTIVATION='relu'
PICTURE_SIZE=224

inputs = Input(shape=(PICTURE_SIZE, PICTURE_SIZE, 3))
def downsample(input, filters=8, third_conv=False, padding='valid'):
  model = Conv2D(filters=filters, kernel_size=[3,3], activation=ACTIVATION, padding=padding)(input)
  model = Conv2D(filters=(filters+4), kernel_size=[3,3], activation=ACTIVATION, padding=padding)(model)
  if third_conv:
    model = Conv2D(filters=(filters+8), kernel_size=[3,3], activation=ACTIVATION, padding=padding)(model)

  #model = BatchNormalization(axis=1)(model)
  model = MaxPooling2D(pool_size=[2,2])(model)
  return model

def upsample(input, filters=20, third_conv=False, padding='same'):
  model = UpSampling2D(size=[2,2])(input)
  model = Conv2DTranspose(filters=filters, kernel_size=[3,3], activation=ACTIVATION, padding=padding)(model)
  if third_conv:
    model = Conv2DTranspose(filters=filters-4, kernel_size=[3,3], activation=ACTIVATIO, padding=padding)(model)
    filters-=4
  model = Conv2DTranspose(filters=filters-4, kernel_size=[3,3], activation=ACTIVATION, padding=padding)(model)
  return model

def upsample2(input, filters):
  model = UpSampling2D(size=[2,2])(input)
  #model = BatchNormalization(axis=1)(model)
  model = Conv2DTranspose(filters=filters, kernel_size=[3,3], activation=ACTIVATION, padding='same')(model)  
  return model

output0 = downsample(inputs, padding='same')
output1 = downsample(output0, filters=16, padding='same')
output2 = downsample(output1, filters=32, third_conv=True, padding='same')
output3 = downsample(output2, filters=64, padding='same')
model = downsample(output3, filters=128, third_conv=True, padding='same')

model = upsample(model, filters=72) 
model = Add()([model, output3])
model = upsample(model, filters=44)
model = Add()([model, output2])
model = upsample(model, filters=24)
model = Add()([model, output1])
model = upsample(model, filters=16) 
model = Add()([model, output0])
model = upsample(model, filters= 12)

output = Conv2DTranspose(filters=OUTPUT_CHANNELS, kernel_size=[1,1], activation='sigmoid', padding='same')(model)

model = Model(inputs, output)

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005),
              loss=TverskyLoss(alpha=0.9835, smooth=2e4),
              metrics=['acc'])
model.load_weights('saved_weights')

suns= tfds.load('sun_dataset')
PICTURE_SIZE = 224

@tf.autograph.experimental.do_not_convert
def load_images(datapoint):
  input_image = tf.image.resize(datapoint['image'], (PICTURE_SIZE, PICTURE_SIZE))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (PICTURE_SIZE, PICTURE_SIZE), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  #normalize
  input_image = tf.cast(input_image, tf.float32) / 255.0
  input_mask = tf.math.round(tf.cast(input_mask, tf.float32) / 255.0)
  inverted_mask = tf.math.subtract(tf.ones((PICTURE_SIZE, PICTURE_SIZE, 1)), input_mask)

  return input_image, inverted_mask

def display(display_list):

  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()

test = suns['test'].map(load_images).batch(8)

def show_predictions(dataset=None, num=2):
  for image, mask in dataset.take(num):
    pred = model.predict(image)
    display([image[0], mask[0], pred[0]])

show_predictions(test)