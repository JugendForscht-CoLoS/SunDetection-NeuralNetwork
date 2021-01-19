import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

pred = np.zeros(shape=(8, 8, 1))
true = np.array([np.zeros(shape=(8, 1)),
                 np.zeros(shape=(8, 1)),
                 np.zeros(shape=(8, 1)),
                 [[0],[0],[0],[1],[0],[0],[0],[0]],
                 [[0],[0],[1],[1],[1],[0],[0],[0]],
                 [[0],[0],[0],[1],[0],[0],[0],[0]],
                 np.zeros(shape=(8, 1)),
                 np.zeros(shape=(8, 1)),
                 ])

pred = tf.constant(pred, dtype='float32')
true = tf.constant(true, dtype='float32')

display_list = [true, pred]
title = ['true', 'predicted']

for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
plt.show()

#Crossentropy
bce = tf.keras.losses.BinaryCrossentropy()
print('Binary Crossentropy')
print(bce(true, pred).numpy())
print()

sfce0 = tfa.losses.SigmoidFocalCrossEntropy(alpha=0.25)
sfce1 = tfa.losses.SigmoidFocalCrossEntropy(alpha=0.75)
sfce2 = tfa.losses.SigmoidFocalCrossEntropy(alpha=1.5)
sfce3 = tfa.losses.SigmoidFocalCrossEntropy(alpha=2.5)
#gamma hat keine Auswirkung auf die schwere des Losses

print('Sigmoid Focal Crossentropy (Alpha=0.25)')
print(sfce0(true, pred).numpy())
print('Sigmoid Focal Crossentropy (Alpha=0.75)')
print(sfce1(true, pred).numpy())
print('Sigmoid Focal Crossentropy (Alpha=1.5)')
print(sfce2(true, pred).numpy())
print('Sigmoid Focal Crossentropy (Alpha=2.5))')
print(sfce3(true, pred).numpy())
print()

#Dice Loss