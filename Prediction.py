import tensorflow as tf
from model.unet import getModel
from losses.losses import TverskyLoss

model = getModel()
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005),
              loss=TverskyLoss(alpha=0.9835, smooth=2e4),
              metrics=['acc'])
model.load_weights('neuralnet/saved_weights')

def display(display_list):

  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()

img = PIL.Image.open('/Users/timjaeger/Desktop/resizedPixelBuffer.png')
imgArray = np.array(img).astype(np.float32)

out = model.predict(imgArray)

display([imgArray, out])
