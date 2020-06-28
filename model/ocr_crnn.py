import tensorflow as tf

'''
if you use tensorflow 1.x
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
'''

from keras.layers import LeakyReLU, Conv2D, BatchNormalization, MaxPool2D, ZeroPadding2D, Permute
from keras.layers import Bidirectional, LSTM, TimeDistributed, Dense, Reshape, ReLU, Input
from keras.models import Model
from model.keys import alphabetChinese
import numpy as np
from PIL import Image

def keras_crnn(nclass, nh=256, leakyRelu=False):

    data_format = 'channels_first'
    # data_format = 'channels_last'
    ks = [3, 3, 3, 3, 3, 3, 2]
    ps = [1, 1, 1, 1, 1, 1, 0]
    ss = [1, 1, 1, 1, 1, 1, 1]
    nm = [64, 128, 256, 256, 512, 512, 512]
    imgInput = Input(shape=(1, 32, None), name='imgInput')
    # imgInput = Input(shape=(32, None, 1), name='imgInput')

    def convRelu(i, x, batchNormalization=False):
        # nIn = nc if i == 0 else nm[i - 1]
        nOut = nm[i]
        if leakyRelu:
            activation = LeakyReLU(alpha=0.2)
        else:
            activation = ReLU(name='relu{0}'.format(i))

        x = Conv2D(filters=nOut,
                   kernel_size=ks[i],
                   strides=(ss[i], ss[i]),
                   padding='valid' if ps[i] == 0 else 'same',
                   dilation_rate=(1, 1),
                   activation=None, use_bias=True, data_format=data_format,
                   name='cnn.conv{0}'.format(i))(x)

        if batchNormalization:
            x = BatchNormalization(epsilon=1e-05, axis=1, momentum=0.1, name='cnn.batchnorm{0}'.format(i))(x)

        x = activation(x)
        return x

    x = imgInput
    x = convRelu(0, batchNormalization=False, x=x)

    x = MaxPool2D(pool_size=(2, 2), name='cnn.pooling{0}'.format(0), padding='valid', data_format=data_format)(x)

    x = convRelu(1, batchNormalization=False, x=x)
    x = MaxPool2D(pool_size=(2, 2), name='cnn.pooling{0}'.format(1), padding='valid', data_format=data_format)(x)

    x = convRelu(2, batchNormalization=True, x=x)
    x = convRelu(3, batchNormalization=False, x=x)
    x = ZeroPadding2D(padding=(0, 1), data_format=data_format)(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 1), padding='valid', name='cnn.pooling{0}'.format(2),
                  data_format=data_format)(x)

    x = convRelu(4, batchNormalization=True, x=x)
    x = convRelu(5, batchNormalization=False, x=x)
    x = ZeroPadding2D(padding=(0, 1), data_format=data_format)(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 1), padding='valid', name='cnn.pooling{0}'.format(3),
                  data_format=data_format)(x)
    x = convRelu(6, batchNormalization=True, x=x)

    x = Permute((3, 2, 1))(x)

    x = Reshape((-1, 512))(x)

    out = None
    x = Bidirectional(LSTM(nh, return_sequences=True, use_bias=True, recurrent_activation='sigmoid'))(x)
    x = TimeDistributed(Dense(nh))(x)
    x = Bidirectional(LSTM(nh, return_sequences=True, use_bias=True, recurrent_activation='sigmoid'))(x)
    out = TimeDistributed(Dense(nclass))(x)

    return Model(imgInput, out)

def resizeNormalize(img, imgH=32):
    scale = img.size[1] * 1.0 / imgH
    w = img.size[0] / scale
    w = int(w)
    img = img.resize((w, imgH), Image.BILINEAR)
    w, h = img.size
    img = (np.array(img) / 255.0 - 0.5) / 0.5
    return img

def strLabelConverter(res, alphabet):
    N = len(res)
    raw = []
    for i in range(N):
        if res[i] != 0 and (not (i > 0 and res[i - 1] == res[i])):
            raw.append(alphabet[res[i] - 1])
    return ''.join(raw)

if __name__ == '__main__':
    import cv2
    import matplotlib.pyplot as plt

    model = keras_crnn(nclass=len(alphabetChinese)+1)
    model.load_weights('D:/coursera/OCR_Series/pretrain/ocr_lstm.h5')

    path = 'D:\\coursera\\OCR_Series\\dataset\\val\\4.jpg'
    img = np.array(Image.open(path).convert('RGB'))
    img = Image.fromarray(np.uint8(img))
    img = img.convert('L')

    image = resizeNormalize(img, 32)
    image = image.astype(np.float32)
    image = np.array([[image]])

    preds = model.predict(image)

    preds = np.argmax(preds, axis=2).reshape((-1,))
    raw = strLabelConverter(preds, alphabetChinese)

    print(raw)
