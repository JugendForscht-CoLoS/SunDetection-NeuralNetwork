import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

SIZE=8
pred = np.zeros(shape=(SIZE, SIZE, 1))
true = np.zeros(shape=(SIZE, SIZE, 1))

for h in range(0, SIZE):
    for w in range(0, SIZE):
        if 0.02>np.random.rand():
            pred[h][w][0]=1

sun = [SIZE/2-SIZE/20, SIZE/2+SIZE/20]
for h in range(0, SIZE):
    for w in range(0, SIZE):
        if sun[0]<h<sun[1] and sun[0]<w<sun[1]:
            true[h][w][0]=1

pred = tf.constant(pred, dtype='float32')
true = tf.constant(true, dtype='float32')

def display(display_list):
    title = ['true', 'predicted']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

ones = tf.constant(np.ones((SIZE, SIZE, 1)),dtype='float32')
inverted_pred = tf.math.subtract(ones, pred)
inverted_true = tf.math.subtract(ones, true)

display([true, pred])
display([inverted_true, inverted_pred])

#Crossentropy
# bce = tf.keras.losses.BinaryCrossentropy()
# print('Binary Crossentropy')
# print(bce(true, pred).numpy())
# print()

#find max value for alpha 
for i in range(0,10):
    i /= 40
    sfce = tfa.losses.SigmoidFocalCrossEntropy(alpha=i)
    print('Sigmoid Focal Crossentropy (alpha=',i,')')
    print(sfce(true, pred).numpy())
    print()
    print(sfce(inverted_true, inverted_pred).numpy())
    print()

#Dice Loss
