import sun_dataset.sun_dataset

import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D, Conv2D, Conv2DTranspose, Input, Add, Concatenate
from tensorflow.keras.models import Model
from tensorflow_examples.models.pix2pix import pix2pix

import numpy as np
import matplotlib.pyplot as plt
#from IPython.display import clear_output

suns, info = tfds.load('sun_dataset',with_info=True)
PICTURE_SIZE = 128

def load_images(datapoint):
  input_image = tf.image.resize(datapoint['image'], (PICTURE_SIZE, PICTURE_SIZE))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (PICTURE_SIZE, PICTURE_SIZE), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  #normalize
  input_image = tf.cast(input_image, tf.float32) / 255.0
  input_mask = tf.cast(input_mask, tf.float32) / 255.0

  return input_image, input_mask

TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 8
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

print(TRAIN_LENGTH)
print(info.splits['test'].num_examples)

def _to_ndarray(data):
    img = np.array([])
    mask = np.array([])

    preprocessed = data.map(load_images).batch(BATCH_SIZE)
    for element in preprocessed:
        img = np.append(img, element[0].numpy())
        mask = np.append(mask, element[1].numpy())

    img = img.reshape(len(img) // (PICTURE_SIZE * PICTURE_SIZE * 3), PICTURE_SIZE, PICTURE_SIZE, 3)
    mask = mask.reshape(len(mask) // (PICTURE_SIZE * PICTURE_SIZE * 1), PICTURE_SIZE, PICTURE_SIZE, 1)

    return img, mask

#train_images, train_masks = _to_ndarray(suns['train'])
#test_images, test_masks = _to_ndarray(suns['test'])
#test = (test_images, test_masks)

train = suns['train'].map(load_images).batch(BATCH_SIZE)
test = suns['test'].map(load_images).batch(BATCH_SIZE)

data_gen_args = dict(horizontal_flip=True,
                     rotation_range=15,
                     zoom_range=0.3,
                     vertical_flip=True)
#train_images_datagen = ImageDataGenerator(**data_gen_args)
#train_masks_datagen = ImageDataGenerator(**data_gen_args)

SEED = 1
#train_images_datagen = train_images_datagen.flow(train_images, batch_size=BATCH_SIZE, shuffle=True, seed=SEED)
#train_masks_datagen = train_masks_datagen.flow(train_masks, batch_size=BATCH_SIZE, shuffle=True, seed=SEED)

def image_mask_generator(image_data_generator, mask_data_generator):
    train_generator = zip(image_data_generator, mask_data_generator)
    for (img, mask) in train_generator:
        yield (img, mask)

#train_generator = image_mask_generator(train_images_datagen, train_masks_datagen)

#sample_image, sample_mask = next(train_generator)

OUTPUT_CHANNELS = 2
ACTIVATION = 'elu'

inputs = Input(shape=(PICTURE_SIZE, PICTURE_SIZE, 3))
def downsample(input, filters=8, third_conv=False, padding='valid'):
  model = Conv2D(filters=filters, kernel_size=[3,3], activation=ACTIVATION, padding=padding)(input)
  model = Conv2D(filters=(filters+4), kernel_size=[3,3], activation=ACTIVATION, padding=padding)(model)
  if third_conv:
    model = Conv2D(filters=(filters+8), kernel_size=[3,3], activation=ACTIVATION, padding=padding)(model)

  model = MaxPooling2D(pool_size=[2,2])(model)
  return model

def upsample(input, filters=20, third_conv=False):
  model = UpSampling2D(size=[2,2])(input)
  model = Conv2DTranspose(filters=filters, kernel_size=[3,3], activation=ACTIVATION)(model)
  if third_conv:
    model = Conv2DTranspose(filters=filters-4, kernel_size=[3,3], activation=ACTIVATION)(model)
    filters-=4
  model = Conv2DTranspose(filters=filters-4, kernel_size=[3,3], activation=ACTIVATION)(model)
  return model

def upsample2(input, filters):
  model = UpSampling2D(size=[2,2])(input)
  model = Conv2DTranspose(filters=filters, kernel_size=[3,3], activation=ACTIVATION, padding='same')(model)
  return model

PADDING='valid'

model = downsample(inputs, padding=PADDING)
model = downsample(model, filters=16, third_conv=True, padding=PADDING)
output2 = downsample(model, filters=32, padding=PADDING) #thirconv
output3 = downsample(output2, filters=64, padding=PADDING)
model = downsample(output3, filters=128, third_conv=True, padding=PADDING)


model = upsample2(model, 68)
model = Concatenate()([model, output3])
model = upsample2(model, 40)
model = Concatenate()([model, output2])
model = upsample2(model, 32)
model = upsample2(model, 16)
model = upsample2(model, 8)

model = upsample(model, filters=76, third_conv=True) #thirdconv
model = Add()([model, output3])
model = upsample(model, filters=40)
model = Add()([model, output2])
model = upsample(model, filters=32)
model = upsample(model, filters=16, third_conv=True)
model = upsample(model)

output = Conv2DTranspose(filters=OUTPUT_CHANNELS, kernel_size=[3,3], activation='softmax', padding='same')(model)
model = Model(inputs, output)

print(model.summary())

from tensorflow.keras import backend as K
def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    
    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.
    
    Ref: https://en.wikipedia.org/wiki/Jaccard_index
    
    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

model.compile(optimizer='adam',
              loss=dice_coef_loss,
              metrics=['accuracy'])

#tf.keras.utils.plot_model(model, show_shapes=True)

def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()

def show_predictions(dataset=None, num=1):
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image)
      display([image[0], mask[0], create_mask(pred_mask)])
  else:
    display([sample_image, sample_mask,
             create_mask(model.predict(sample_image[tf.newaxis, ...]))])


'''
class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=True)
    show_predictions()
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))
'''
EPOCHS = 5
VAL_SUBSPLITS = 1
VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS

model_history = model.fit(train, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test)

''',
                          callbacks=[DisplayCallback()])'''


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
model.save('neuronalnet/sun_detection.h5')
