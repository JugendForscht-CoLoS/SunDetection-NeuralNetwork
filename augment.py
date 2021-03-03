import tensorflow_datasets as tfds
#import os
import numpy as np
import sun_dataset.sun_dataset
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

#Datensatz laden
suns, info = tfds.load('sun_dataset',with_info=True)
PICTURE_SIZE = 512

#Datensatz preparieren
def load_images(datapoint):
  input_image = tf.image.resize(datapoint['image'], (PICTURE_SIZE, PICTURE_SIZE))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (PICTURE_SIZE, PICTURE_SIZE), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  
  return input_image, input_mask

TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 4

print(info.splits['test'].num_examples)

#Bilder zu numpy-Arrays konvertieren
def _to_ndarray(data):
    img = np.array([])
    mask = np.array([])

    preprocessed = data.map(load_images)
    for element in preprocessed:
        img = np.append(img, element[0].numpy())
        mask = np.append(mask, element[1].numpy())

    img = img.reshape(len(img) // (PICTURE_SIZE * PICTURE_SIZE * 3), PICTURE_SIZE, PICTURE_SIZE, 3)
    mask = mask.reshape(len(mask) // (PICTURE_SIZE * PICTURE_SIZE * 1), PICTURE_SIZE, PICTURE_SIZE, 1)

    return img, mask


train_images, train_masks = _to_ndarray(suns['train'])
test_images, test_masks = _to_ndarray(suns['test'])

#DataGenerator vorbereiten
data_gen_args = dict(horizontal_flip=True,    #die Bilder sollen automatisch horizontal gespiegelt werden
                     rotation_range=30)       #und zufällig gedreht  
test_images_datagen = ImageDataGenerator(**data_gen_args)
test_masks_datagen = ImageDataGenerator(**data_gen_args)

# Die Daten werden nun in den DataGenerator eingegeben. Zusätzlich wird das Verzeichnis eingegeben, wo die Bilder und Masken gespeichert werden sollen 
SEED = 1  #Der seed sorgt dafür, dass die Maske und das Sonnenbild gleich verändert werden
test_images_datagen = test_images_datagen.flow(test_images, batch_size=BATCH_SIZE, shuffle=True, seed=SEED,save_to_dir='C:/Users/Jonas/Documents/JugendForscht/test/image', save_format='jpeg', save_prefix='aug')
test_masks_datagen = test_masks_datagen.flow(test_masks, batch_size=BATCH_SIZE, shuffle=True, seed=SEED,save_to_dir='C:/Users/Jonas/Documents/JugendForscht/test/mask', save_format='jpeg', save_prefix='aug')

#Gleiches Vorgehen 
data_gen_args = dict(horizontal_flip=True,
                     rotation_range=30)
train_images_datagen = ImageDataGenerator(**data_gen_args)
train_masks_datagen = ImageDataGenerator(**data_gen_args)

SEED = 2
train_images_datagen = train_images_datagen.flow(train_images, batch_size=BATCH_SIZE, shuffle=True, seed=SEED, save_to_dir='C:/Users/Jonas/Documents/JugendForscht/test/image', save_format='jpeg', save_prefix='augm')
train_masks_datagen = train_masks_datagen.flow(train_masks, batch_size=BATCH_SIZE, shuffle=True, seed=SEED, save_to_dir='C:/Users/Jonas/Documents/JugendForscht/test/mask', save_format='jpeg', save_prefix='augm')

#die beiden Datagenerator (welche entweder Masken oder Sonnenbilder beinhalten) werden vereint
def image_mask_generator(image_data_generator, mask_data_generator):
    train_generator = zip(image_data_generator, mask_data_generator)
    for (img, mask) in train_generator:
        yield (img, mask)

#Beim Iterieren durch die train- und test-Generator werden die Bilder gespeichert
test_datagen = image_mask_generator(test_images_datagen, test_masks_datagen)
for i in range(0, info.splits['test'].num_examples):
    batch = next(test_datagen)

train_datagen = image_mask_generator(train_images_datagen, train_masks_datagen)
for i in range(0, info.splits['train'].num_examples):
    batch = next(train_datagen)
