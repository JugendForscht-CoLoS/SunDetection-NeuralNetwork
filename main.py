import sun_dataset.sun_dataset
from losses.losses import TverskyLoss
from model.model import getModel

import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.metrics import FalseNegatives, FalsePositives, TruePositives, TrueNegatives

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

suns, info = tfds.load('sun_dataset',with_info=True)
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

TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 8
BUFFER_SIZE = 50
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

print(STEPS_PER_EPOCH*BATCH_SIZE)
print(TRAIN_LENGTH)
print(info.splits['test'].num_examples)


def display(display_list, with_colorbar=False):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    img = plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    if with_colorbar == True and i != 0: plt.colorbar(img)
    plt.axis('off')
  plt.show()

train = suns['train'].map(load_images)
test = suns['test'].map(load_images)

for image, mask in train.take(1):
  sample_image, sample_mask = image, mask

train = train.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat(10)
test = test.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat(10)

model = getModel()
print(model.summary())

model.compile(optimizer='adam',
              loss=TverskyLoss(alpha=0.2, smooth=1e3),
              metrics=['acc', FalseNegatives(), FalsePositives(), TruePositives(), TrueNegatives()])


def show_predictions(dataset=None, num=1):
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image)
      display([image[0], mask[0], pred_mask[0]])
  else:
    display([sample_image, sample_mask, model.predict(sample_image[tf.newaxis, ...])[0]], with_colorbar=True)


class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=True)
    show_predictions()
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))


EPOCHS = 3
VAL_SUBSPLITS = 1
VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS

model_history = model.fit(train, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test,
                          callbacks=[DisplayCallback()])


loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

epochs = range(EPOCHS)

plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()

show_predictions(test, 6)
model.save_weights('neuralnet/saved_weights')
