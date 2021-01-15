import tensorflow_datasets as tfds
#import os
import numpy as np
import sun_dataset.sun_dataset
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

suns, info = tfds.load('sun_dataset',with_info=True)
PICTURE_SIZE = 512

def load_images(datapoint):
  input_image = tf.image.resize(datapoint['image'], (PICTURE_SIZE, PICTURE_SIZE))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (PICTURE_SIZE, PICTURE_SIZE), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  #normalize
  #input_image = tf.cast(input_image, tf.float32) / 255.0
  #input_mask = tf.cast(input_mask, tf.float32) / 255.0

  return input_image, input_mask

TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 4

print(info.splits['test'].num_examples)

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

data_gen_args = dict(horizontal_flip=True,
                     rotation_range=30)
test_images_datagen = ImageDataGenerator(**data_gen_args)
test_masks_datagen = ImageDataGenerator(**data_gen_args)

SEED = 1
test_images_datagen = test_images_datagen.flow(test_images, batch_size=BATCH_SIZE, shuffle=True, seed=SEED,save_to_dir='C:/Users/Jonas/Documents/JugendForscht/test/image', save_format='jpeg', save_prefix='aug')
test_masks_datagen = test_masks_datagen.flow(test_masks, batch_size=BATCH_SIZE, shuffle=True, seed=SEED,save_to_dir='C:/Users/Jonas/Documents/JugendForscht/test/mask', save_format='jpeg', save_prefix='aug')

data_gen_args = dict(horizontal_flip=True,
                     rotation_range=30)
train_images_datagen = ImageDataGenerator(**data_gen_args)
train_masks_datagen = ImageDataGenerator(**data_gen_args)

SEED = 2
train_images_datagen = train_images_datagen.flow(train_images, batch_size=BATCH_SIZE, shuffle=True, seed=SEED, save_to_dir='C:/Users/Jonas/Documents/JugendForscht/test/image', save_format='jpeg', save_prefix='augm')
train_masks_datagen = train_masks_datagen.flow(train_masks, batch_size=BATCH_SIZE, shuffle=True, seed=SEED, save_to_dir='C:/Users/Jonas/Documents/JugendForscht/test/mask', save_format='jpeg', save_prefix='augm')


def image_mask_generator(image_data_generator, mask_data_generator):
    train_generator = zip(image_data_generator, mask_data_generator)
    for (img, mask) in train_generator:
        yield (img, mask)

test_datagen = image_mask_generator(test_images_datagen, test_masks_datagen)
for i in range(0, info.splits['test'].num_examples):
    batch = next(test_datagen)

train_datagen = image_mask_generator(train_images_datagen, train_masks_datagen)
for i in range(0, info.splits['train'].num_examples):
    batch = next(train_datagen)

'''
path = 'C:/Users/Jonas/tensorflow_datasets/downloads/manual/suns/'
mpath = (path + 'test/mask')
ipath = (path + 'test/image')

# we create two instances with the same arguments
data_gen_args = dict(horizontal_flip=True,
                     vertical_flip=True,
                     rescale=1./255)
train_image_datagen = ImageDataGenerator(**data_gen_args)
train_mask_datagen = ImageDataGenerator(**data_gen_args)
# Provide the same seed and keyword arguments to the fit and flow methods
SEED = 1
image_generator = train_image_datagen.flow_from_directory(
    ipath,
    target_size=(128,128),
    class_mode=None,
    interpolation='bicubic',
    shuffle=False,
    seed=SEED)
mask_generator = train_mask_datagen.flow_from_directory(
    mpath,
    target_size=(128,128),
    class_mode=None,
    interpolation='nearest',
    shuffle=False,
    seed=SEED)



def image_mask_generator(image_data_generator, mask_data_generator):
    train_generator = zip(image_data_generator, mask_data_generator)
    for (img, mask) in train_generator:
        yield (img, mask)
test_generator = image_mask_generator(image_generator, mask_generator)
print(len(list(test_generator)))

for image, mask in test_generator.take(12):
    display(image, mask)

'''
'''
i = -1
temp_img = None
for img in os.listdir(path):
    if i % 2 != 0:
        temp_img = img
    else:
        print("Image", str(int(i/2)))
        print(path + temp_img)
        print(path + img)
        print()

    i += 1
'''

'''
MANUAL_DOWNLOAD_INSTRUCTIONS=""" dsfghggfh fgh"""
dl_manager = tfds.download.DownloadManager(download_dir="C:/Users/Jonas/Downloads/", manual_dir="C:/Users/Jonas/tensorflow_datasets/download/manual")
for filename in os.listdir(dl_manager.manual_dir / "suns/train"):
    print(os.fsdecode(filename))
'''

'''dl_manager = tfds.download.DownloadManager(download_dir="C:/Users/Jonas/Downloads/")
iterator = dl_manager.iter_archive('C:/Users/Jonas/tensorflow_datasets/downloads/manual/suns.zip')

for filename, fobj in iterator:
    print(filename)    
    print(fobj.read())
'''
