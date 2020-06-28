from model.ocr_denesnet import dense_ocr
import cv2
import numpy as np
from PIL import Image
import tensorflow.keras.backend as K
from model.ocr_crnn import keras_crnn, resizeNormalize, strLabelConverter
from model.keys import alphabetChinese

class OcrAgent(object):
    def __init__(self, modelPath):
        char = ''
        with open('D:\\coursera\\OCR_Series\\model\\char.txt',
                  encoding='utf-8') as f:
            for ch in f.readlines():
                ch = ch.strip('\r\n')
                char = char + ch

        nclass = len(char) + 1
        print('nclass:', len(char))
        self.id_to_char = {i: j for i, j in enumerate(char)}

        self.model = dense_ocr(nclass)
        self.model.load_weights(modelPath)

    def predict(self, img):
        # IMAGE_MEAN = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 3))
        # img = img - IMAGE_MEAN

        img = Image.fromarray(np.uint8(img))
        img = img.convert('L')
        img = np.array(img)

        h, w = img.shape
        ratio = 32.0/h
        text_img = cv2.resize(img, (int(w*ratio), int(h*ratio)))
        text_img = text_img/255.0
        text_img = np.expand_dims(text_img, 2)
        text_img = np.expand_dims(text_img, 0)

        y_pred = self.model.predict(text_img)
        out = K.get_value(K.ctc_decode(y_pred,
                                       input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1])[0][0]
                          )[:, :]

        out = u''.join([self.id_to_char[x] for x in out[0]])
        return out

class Ocr_Lstm_Agent:
    def __init__(self, modelPath):
        self.model = keras_crnn(nclass=len(alphabetChinese) + 1)
        self.model.load_weights('D:/coursera/OCR_Series/pretrain/ocr_lstm.h5')

    def predict(self, img):
        img = Image.fromarray(np.uint8(img))
        img = img.convert('L')

        image = resizeNormalize(img, 32)
        image = image.astype(np.float32)
        image = np.array([[image]])

        preds = self.model.predict(image)

        preds = np.argmax(preds, axis=2).reshape((-1,))
        raw = strLabelConverter(preds, alphabetChinese)

        return raw

if __name__ == '__main__':
    img = cv2.imread('D:\\coursera\\OCR_Series\\dataset\\val\\4.jpg')
    im = Image.fromarray(img)

    # agent = OcrAgent(modelPath='D:\\coursera\\OCR_Series\\pretrain\\weights_ocr.hdf5')
    # agent.predict(img)

    agent = Ocr_Lstm_Agent('D:/coursera/OCR_Series/pretrain/ocr_lstm.h5')
    result = agent.predict(im)

    print(result)