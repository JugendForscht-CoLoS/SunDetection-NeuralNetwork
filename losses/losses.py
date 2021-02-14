import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

# SIZE=8
# pred = np.zeros(shape=(SIZE, SIZE, 1))
# true = np.zeros(shape=(SIZE, SIZE, 1))

# for h in range(0, SIZE):
#     for w in range(0, SIZE):
#         if 0.3>np.random.rand():
#             pred[h][w][0]=1

# sun = [SIZE/2-SIZE/20, SIZE/2+SIZE/20]
# for h in range(0, SIZE):
#     for w in range(0, SIZE):
#         if sun[0]<h<sun[1] and sun[0]<w<sun[1]:
#             true[h][w][0]=1

# pred = tf.constant(pred, dtype='float32')
# true = tf.constant(true, dtype='float32')

# def display(display_list):
#     title = ['true', 'predicted']

#     for i in range(len(display_list)):
#         plt.subplot(1, len(display_list), i+1)
#         plt.title(title[i])
#         plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
#     plt.show()

# ones = tf.constant(np.ones((SIZE, SIZE, 1)),dtype='float32')
# inverted_pred = tf.math.subtract(ones, pred)
# inverted_true = tf.math.subtract(ones, true)

# display([true, pred])
# display([inverted_true, inverted_pred])

# def print_loss(loss_func, name):
#     print(name)
#     print(loss_func(true, pred).numpy())
#     print(loss_func(inverted_true, inverted_pred).numpy())
#     print()

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1 - score

#print_loss(dice_loss, 'Dice Loss')

def log_cosh_dice_loss(y_true, y_pred):
    x = dice_loss(y_true, y_pred)
    return tf.math.log((tf.exp(x) + tf.exp(-x)) / 2.0)

#print_loss(log_cosh_dice_loss, 'Log Cosh Dice Loss')

def bce_dsc_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    o = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred) + dice_loss(y_true, y_pred)
    return tf.reduce_mean(o)

#print_loss(bce_dsc_loss, 'Binary Crossentropy and Dice Loss')

# def focal_tversky(y_true, y_pred, gamma=0.4):
#         pt_1 = tversky_index(y_true, y_pred)
#         return K.pow((1 - pt_1), gamma)

#print_loss(focal_tversky, 'Focal Tversky Loss')

class TverskyLoss(tf.keras.losses.Loss):
    def __init__(self, alpha, beta=None, smooth=1):
        super(TverskyLoss, self).__init__()

        self.alpha = alpha
        self.beta = beta
        if self.beta == None: self.beta = 1 - alpha
        self.smooth = smooth
        
    def call(self, y_true, y_pred):
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)

        tp = K.sum(y_true * y_pred)
        fn = K.sum(y_true * (1 - y_pred))
        fp = K.sum((1 - y_true) * y_pred)

        index = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        return 1 - index
    
    def get_config(self):
        config = super(TverskyLoss, self).get_config()
        return config #{"alpha": self.alpha, "beta": self.beta, "smooth": self.smooth}

# tversky = TverskyLoss(alpha=0.7)
# print(tversky(true, pred))