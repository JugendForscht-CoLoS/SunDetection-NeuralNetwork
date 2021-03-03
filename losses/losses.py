import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

#Klasse, die den Tversky-Loss berechnet
#Erbt von tf.keras.losses.Loss
class TverskyLoss(tf.keras.losses.Loss):
    def __init__(self, alpha, beta=None, smooth=1):
        super(TverskyLoss, self).__init__()
        #Parameter des Losses
        self.alpha = alpha
        self.beta = beta
        if self.beta == None: self.beta = 1 - alpha
        self.smooth = smooth
        
    def call(self, y_true, y_pred): #y_true: Maske des Datensatzes ; y_pred: Maske des neuronalen Netzes
        '''
                 TP
        1 -  ------------
             TP+a*FN+b*FP
        '''
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)

        tp = K.sum(y_true * y_pred) #True Positives
        fn = K.sum(y_true * (1 - y_pred)) #False Negatives
        fp = K.sum((1 - y_true) * y_pred) #False Positives

        index = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        return 1 - index
    
    def get_config(self):
        config = super(TverskyLoss, self).get_config()
        return config 
