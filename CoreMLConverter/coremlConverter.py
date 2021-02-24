import coremltools as ct
import tensorflow as tf

from model.model import getModel
from losses.losses import TverskyLoss

pathML = input('Pfad des .mlmodels: ')
author = input('Autor: ')
modelLicense = input('Lizenz: ')
description = input('kurze Beschreibung: ')
version = input('Version: ')

model = getModel()
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005),
              loss=TverskyLoss(alpha=0.9835, smooth=2e4),
              metrics=['acc'])
model.load_weights('neuronalnet/saved_weights')

mlModel = ct.convert(model)
mlModel.author = author
mlModel.license = modelLicense
mlModel.short_description = description
mlModel.version = version
mlModel.save(pathML)
