import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dense, ReLU, AveragePooling2D, TimeDistributed
from tensorflow.keras.layers import Flatten, Permute, Dropout, Concatenate, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model

def conv_block(input, growth_rate, dropout_rate=None, weight_decay=1e-4):
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(input)
    x = tf.keras.layers.ReLU()(x)
    x = Conv2D(growth_rate, (3, 3), padding='same', kernel_regularizer=l2(weight_decay))(x)
    if (dropout_rate):
        x = Dropout(dropout_rate)(x)
    return x

def dense_block(x, nb_layers, nb_filter, growth_rate, droput_rate=0.2, weight_decay=1e-4):
    for i in range(nb_layers):
        cb = conv_block(x, growth_rate, droput_rate, weight_decay)
        x = Concatenate(axis=-1)([x, cb])
        nb_filter += growth_rate
    return x, nb_filter

def transition_block(input, nb_filter, dropout_rate=None, weight_decay=1e-4):
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(input)
    x = ReLU()(x)
    x = Conv2D(nb_filter, (1, 1), padding='same', use_bias=False, kernel_regularizer=l2(weight_decay))(x)

    if (dropout_rate):
        x = Dropout(dropout_rate)(x)

    x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    return x, nb_filter

def dense_ocr(nclass):
    input = Input(shape=(32, None, 1), name='the_input')

    _dropout_rate = 0.2
    _weight_decay = 1e-4

    _nb_filter = 64
    # conv 64  5*5 s=2
    x = Conv2D(_nb_filter, (5, 5), strides=(2, 2), kernel_initializer='he_normal', padding='same',
               use_bias=False, kernel_regularizer=l2(_weight_decay))(input)

    # 64 +  8 * 8 = 128
    x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None, _weight_decay)
    # 128
    x, _nb_filter = transition_block(x, 128, _dropout_rate, _weight_decay)

    # 128 + 8 * 8 = 192
    x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None, _weight_decay)
    # 192->128
    x, _nb_filter = transition_block(x, 128, _dropout_rate, _weight_decay)

    # 128 + 8 * 8 = 192
    x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None, _weight_decay)

    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    x = ReLU()(x)

    x = Permute((2, 1, 3), name='permute')(x)
    x = TimeDistributed(Flatten(), name='flatten')(x)
    y_pred = Dense(nclass, name='out', activation='softmax')(x)

    model = Model(inputs=input,outputs=y_pred)
    return model

if __name__ == '__main__':
    model = dense_ocr(8696)
    model.load_weights('D:\\coursera\\OCR_Series\\pretrain\\weights_ocr.hdf5')
