import coremltools as ct
import tensorflow as tf

pathH5 = input('Pfad des .h5 models: ')
pathML = input('Pfad des .mlmodels: ')
author = input('Autor: ')
modelLicense = input('Lizenz: ')
description = input('kurze Beschreibung: ')
version = input('Version: ')

keras_model = tf.keras.models.load_model(pathH5)
mlModel = ct.convert(keras_model)
mlModel.author = author
mlModel.license = modelLicense
mlModel.short_description = description
mlModel.version = version
mlModel.save(pathML)
