from tensorflow.keras.layers import MaxPooling2D, UpSampling2D, Conv2D, Conv2DTranspose, Input, Add
from tensorflow.keras.models import Model

OUTPUT_CHANNELS = 1
ACTIVATION='relu'
PICTURE_SIZE=224

inputs = Input(shape=(PICTURE_SIZE, PICTURE_SIZE, 3))
def downsample(input, filters=8, third_conv=False, padding='valid'):
  model = Conv2D(filters=filters, kernel_size=[3,3], activation=ACTIVATION, padding=padding)(input)
  model = Conv2D(filters=(filters+4), kernel_size=[3,3], activation=ACTIVATION, padding=padding)(model)
  if third_conv:
    model = Conv2D(filters=(filters+8), kernel_size=[3,3], activation=ACTIVATION, padding=padding)(model)

  model = MaxPooling2D(pool_size=[2,2])(model)
  return model

def upsample(input, filters=20, third_conv=False, padding='same'):
  model = UpSampling2D(size=[2,2])(input)
  model = Conv2DTranspose(filters=filters, kernel_size=[3,3], activation=ACTIVATION, padding=padding)(model)
  if third_conv:
    model = Conv2DTranspose(filters=filters-4, kernel_size=[3,3], activation=ACTIVATION, padding=padding)(model)
    filters-=4
  model = Conv2DTranspose(filters=filters-4, kernel_size=[3,3], activation=ACTIVATION, padding=padding)(model)
  return model

def upsample2(input, filters):
  model = UpSampling2D(size=[2,2])(input)
  model = Conv2DTranspose(filters=filters, kernel_size=[3,3], activation=ACTIVATION, padding='same')(model)  
  return model

def getModel():
    output0 = downsample(inputs, padding='same')
    output1 = downsample(output0, filters=16, padding='same')
    output2 = downsample(output1, filters=32, third_conv=True, padding='same')
    output3 = downsample(output2, filters=64, padding='same')
    output4 = downsample(output3, filters=128, third_conv=True, padding='same')

    model = Conv2D(filters=200, kernel_size=[3,3], activation=ACTIVATION)(model)  
    model = Conv2D(filters=210, kernel_size=[3,3], activation=ACTIVATION)(model) 
    model = Conv2D(filters=220, kernel_size=[3,3], activation=ACTIVATION)(model) 

    model = Conv2DTranspose(filters=220, kernel_size=[3,3], activation=ACTIVATION)(model)  
    model = Conv2DTranspose(filters=210, kernel_size=[3,3], activation=ACTIVATION)(model) 
    model = Conv2DTranspose(filters=200, kernel_size=[3,3], activation=ACTIVATION)(model) 

    model = upsample(model, filters=72) 
    model = Add()([model, output3])
    model = upsample(model, filters=44)
    model = Add()([model, output2])
    model = upsample(model, filters=24)
    model = Add()([model, output1])
    model = upsample(model, filters=16) 
    model = Add()([model, output0])
    model = upsample(model, filters= 12)

    output = Conv2DTranspose(filters=OUTPUT_CHANNELS, kernel_size=[1,1], activation='sigmoid', padding='same')(model)

    model = Model(inputs, output)
    return model