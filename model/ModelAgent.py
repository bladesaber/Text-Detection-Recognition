from model.ocr_agent import OcrAgent, Ocr_Lstm_Agent
from model.yolo_agent import YoloAgent
import cv2
import matplotlib.pyplot as plt

if __name__ == '__main__':
    p = 'D:/coursera/OCR_Series/dataset/val/1.jpg'
    img = cv2.imread(p)

    yolomodel = YoloAgent(path='D:/coursera/OCR_Series/pretrain/weights_yolo.hdf5', train=False)

    # ocrmodel = OcrAgent(modelPath='D:\\coursera\\OCR_Series\\pretrain\\weights_ocr.hdf5')
    ocrmodel = Ocr_Lstm_Agent('D:/coursera/OCR_Series/pretrain/ocr_lstm.h5')

    img_list = yolomodel.predict(img)
    for img in img_list:
        result = ocrmodel.predict(img)
        print(result)