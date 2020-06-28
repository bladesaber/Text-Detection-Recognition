import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import ZeroPadding2D, Add, UpSampling2D, Concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

def DarknetConv2D_BN_Leaky(x, filters, kernel_size, stride=(1, 1), use_bias=False):
    x = DarknetConv2D(x, filters, kernel_size, stride=stride, use_bias=use_bias)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x

def DarknetConv2D(x, filters, kernel_size, stride, use_bias):
    if stride==(2, 2):
        padding = 'valid'
    else:
        padding = 'same'
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=stride,
                           use_bias=use_bias, padding=padding, kernel_regularizer=l2(5e-4))(x)
    return x

def resblock_body(x, num_filters, num_blocks):
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)

    x = DarknetConv2D_BN_Leaky(x, filters=num_filters, kernel_size=(3, 3), stride=(2, 2))

    for i in range(num_blocks):
        y = DarknetConv2D_BN_Leaky(x, filters=num_filters//2, kernel_size=(1, 1))
        y = DarknetConv2D_BN_Leaky(y, filters=num_filters, kernel_size=(3, 3))

        x = Add()([x, y])
    return x

def darknet_body(x):
    x = DarknetConv2D_BN_Leaky(x, filters=32, kernel_size=(3, 3))

    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)

    return x

def make_last_layers(x, num_filters, out_filters):
    x = DarknetConv2D_BN_Leaky(x, filters=num_filters, kernel_size=(1, 1))
    x = DarknetConv2D_BN_Leaky(x, filters=num_filters*2, kernel_size=(3, 3))
    x = DarknetConv2D_BN_Leaky(x, filters=num_filters, kernel_size=(1, 1))
    x = DarknetConv2D_BN_Leaky(x, filters=num_filters*2, kernel_size=(3, 3))
    x = DarknetConv2D_BN_Leaky(x, filters=num_filters, kernel_size=(1, 1))

    y = DarknetConv2D_BN_Leaky(x, num_filters*2, (3,3))
    y = DarknetConv2D(y, out_filters, (1,1), stride=(1, 1), use_bias=True)
    return x, y

def yolo_text(num_classes, anchors, train=False):
    imgInput = Input(shape=(None, None, 3))

    darknet = Model(imgInput, darknet_body(imgInput))
    num_anchors = len(anchors) // 3
    x, y1 = make_last_layers(darknet.output, 512, num_anchors * (num_classes + 5))

    x = DarknetConv2D_BN_Leaky(x, filters=256, kernel_size=(1, 1))
    x = UpSampling2D(2)(x)

    x = Concatenate()([x, darknet.layers[152].output])
    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5))

    x = DarknetConv2D_BN_Leaky(x, filters=128, kernel_size=(1, 1))
    x = UpSampling2D(2)(x)

    x = Concatenate()([x, darknet.layers[92].output])
    x, y3 = make_last_layers(x, 128, num_anchors * (num_classes + 5))

    out = [y1, y2, y3]

    if train:
        # num_anchors = len(anchors)
        # y_true = [Input(shape=(None, None, num_anchors // 3, num_classes + 5)) for l in range(3)]
        #
        # loss = Lambda(yolo_loss, output_shape=(4,), name='loss',
        #               arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5, })(out + y_true)
        #
        # def get_loss(loss, index):
        #     return loss[index]
        #
        # lossName = ['class_loss', 'xy_loss', 'wh_loss', 'confidence_loss']
        # lossList = [Lambda(get_loss, output_shape=(1,), name=lossName[i], arguments={'index': i})(loss) for i in
        #             range(4)]
        # textModel = Model([imgInput, *y_true], lossList)
        # return textModel
        pass

    else:
        textModel = Model([imgInput], out)
        return textModel

if __name__ == '__main__':
    import numpy as np
    from PIL import Image

    anchors = '8,11, 8,16, 8,23, 8,33, 8,48, 8,97, 8,139, 8,198, 8,283'
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
    num_anchors = len(anchors)

    model = yolo_text(2, anchors, False)
    model.load_weights('D:/coursera/OCR_Series/pretrain/weights_yolo.hdf5')

    p = 'D:/coursera/OCR_Series/dataset/val/1.jpg'
    img = np.array(Image.open(p))
    if img.shape[-1]>3:
        img = img[:, :, :3]

    im = Image.fromarray(img)
    w_, h_ = 608, 608

    boxed_image = im.resize((w_, h_), Image.BICUBIC)
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)

    print(image_data.shape)
    out = model(image_data)

