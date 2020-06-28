from model.yolo3 import yolo_text
import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# K = tf.keras.backend

import cv2
import matplotlib.pyplot as plt
from PIL import Image
from Utils import utils
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from model.detector.TextDetector import TextDetector, get_boxes

class YoloAgent(object):
    def __init__(self, path, train):
        anchors = '8,11, 8,16, 8,23, 8,33, 8,48, 8,97, 8,139, 8,198, 8,283'
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)
        num_anchors = len(anchors)

        class_names = ['none', 'text', ]
        num_classes = len(class_names)

        self.textModel = yolo_text(num_classes, anchors, train=train)

        if path:
            self.textModel.load_weights(path)

        # self.image_shape = K.placeholder(shape=(2,))  ##图像原尺寸:h,w
        # self.input_shape = K.placeholder(shape=(2,))  ##图像resize尺寸:h,w
        self.image_shape = tf.keras.layers.Input(shape=(2, ), batch_size=1, dtype=tf.float32)
        self.input_shape = tf.keras.layers.Input(shape=(2, ), batch_size=1, dtype=tf.float32)

        self.box_score = self.box_layer([*self.textModel.output], anchors, num_classes)
        self.yolo_model = Model([self.textModel.input, self.input_shape, self.image_shape], self.box_score)

    def yolo_head(self, feats, anchors, num_classes, input_shape, calc_loss=False):
        """Convert final layer features to bounding box parameters."""
        num_anchors = len(anchors)
        # Reshape to batch, height, width, num_anchors, box_params.
        anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

        grid_shape = K.shape(feats)[1:3]  # height, width
        grid_y = tf.tile(
            K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], 1, 1])
        grid_x = tf.tile(
            K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, 1, 1])

        grid = K.concatenate([grid_x, grid_y])
        grid = K.cast(grid, K.dtype(feats))

        feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

        # Adjust preditions to each spatial grid point and anchor size.
        # 百分比化
        box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
        box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
        box_confidence = K.sigmoid(feats[..., 4:5])
        box_class_probs = K.sigmoid(feats[..., 5:])

        if calc_loss == True:
            return grid, feats, box_xy, box_wh

        return box_xy, box_wh, box_confidence, box_class_probs

    def box_layer(self, inputs, anchors, num_classes):
        y1, y2, y3 = inputs
        out = [y1, y2, y3]

        num_layers = len(out)
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        boxes = []
        scores = []

        # input_shape = K.cast(self.input_shape, tf.float32)
        # image_shape = K.cast(self.image_shape, tf.float32)
        input_shape = K.reshape(self.input_shape, shape=(2, ))
        image_shape = K.reshape(self.image_shape, shape=(2, ))

        for lay in range(num_layers):
            box_xy, box_wh, box_confidence, box_class_probs = self.yolo_head(out[lay], anchors[anchor_mask[lay]],
                                                                             num_classes, input_shape)
            # box_xy = (box_xy - offset) * scale
            # box_wh = box_wh*scale

            box_score = box_confidence * box_class_probs
            box_score = K.reshape(box_score, [-1, num_classes])

            box_mins = box_xy - (box_wh / 2.)
            box_maxes = box_xy + (box_wh / 2.)
            box = K.concatenate([
                box_mins[..., 0:1],  # xmin
                box_mins[..., 1:2],  # ymin
                box_maxes[..., 0:1],  # xmax
                box_maxes[..., 1:2]  # ymax
            ], axis=-1)

            box = K.reshape(box, [-1, 4])
            boxes.append(box)
            scores.append(box_score)

        boxes = K.concatenate(boxes, axis=0)
        scores = K.concatenate(scores, axis=0)

        boxes *= K.concatenate([image_shape[::-1], image_shape[::-1]])

        return [boxes, scores[..., 1]]

    def plot_box(self, img, boxes):
        blue = (255, 0, 0)
        tmp = np.copy(img)
        for box in boxes:
            cv2.rectangle(tmp, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), blue, 2)  # 19
        plt.imshow(tmp)
        plt.show()

    def text_detect(self, boxes, scores, img):

        MAX_HORIZONTAL_GAP = 30
        # MAX_HORIZONTAL_GAP = 1000

        MIN_V_OVERLAPS = 0.6
        MIN_SIZE_SIM = 0.6

        # TEXT_PROPOSALS_MIN_SCORE = 0.7
        # TEXT_PROPOSALS_NMS_THRESH = 0.3
        # TEXT_LINE_NMS_THRESH = 0.3

        TEXT_PROPOSALS_MIN_SCORE = 0.35
        TEXT_PROPOSALS_NMS_THRESH = 0.15
        TEXT_LINE_NMS_THRESH = 0.15

        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
        textdetector = TextDetector(MAX_HORIZONTAL_GAP, MIN_V_OVERLAPS, MIN_SIZE_SIM)
        shape = img.shape[:2]

        boxes, scores = textdetector.detect(boxes,
                                    scores[:, np.newaxis],
                                    shape,
                                    TEXT_PROPOSALS_MIN_SCORE,
                                    TEXT_PROPOSALS_NMS_THRESH,
                                    TEXT_LINE_NMS_THRESH,
                                    )

        text_recs = get_boxes(boxes)
        newBox = []
        rx = 1
        ry = 1
        for box in text_recs:
            x1, y1 = (box[0], box[1])
            x2, y2 = (box[2], box[3])
            x3, y3 = (box[6], box[7])
            x4, y4 = (box[4], box[5])
            newBox.append([x1 * rx, y1 * ry, x2 * rx, y2 * ry, x3 * rx, y3 * ry, x4 * rx, y4 * ry])
        return newBox, scores
        # return boxes, scores

    def predict(self, img, prob=0.05):
        original_img = img
        IMAGE_MEAN = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 3))
        img = img - IMAGE_MEAN

        h, w, channel = img.shape
        w_, h_ = utils.resize_im(w, h, scale=416, max_scale=2048)  ##短边固定为608,长边max_scale<4000
        # w_, h_ = 608, 608

        boxed_image = cv2.resize(img, (w_, h_))
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)

        boxes, scores = self.yolo_model([
            image_data,
            np.array([h_, w_], dtype=np.float).reshape((1, 2)),
            np.array([h, w], dtype=np.float).reshape((1, 2))
        ])
        boxes, scores = boxes.numpy(), scores.numpy()

        keep = np.where(scores > prob)

        boxes[:, 0:4][boxes[:, 0:4] < 0] = 0
        boxes[:, 0][boxes[:, 0] >= w] = w - 1
        boxes[:, 1][boxes[:, 1] >= h] = h - 1
        boxes[:, 2][boxes[:, 2] >= w] = w - 1
        boxes[:, 3][boxes[:, 3] >= h] = h - 1
        boxes = boxes[keep[0]]

        scores = scores[keep[0]]

        # yolo result
        # return boxes, scores

        sorted_indices = np.argsort(scores.ravel())[::-1]
        box, scores = boxes[sorted_indices], scores[sorted_indices]

        boxes, scorse = self.text_detect(box, scores, img)

        boxes = utils.sort_box(boxes)

        # for plot boxes
        # return boxes
        return self.get_clip_imgs(original_img, boxes)

    def get_clip_imgs(self, img, boxes, leftAdjustAlph=0.0, rightAdjustAlph=0.0):
        clip_images = []
        for box in boxes:
            clip_image = utils.cut_img(img, box, leftAdjustAlph, rightAdjustAlph)
            clip_images.append(clip_image)
        return clip_images

if __name__ == '__main__':
    import cv2

    model = YoloAgent(path='D:/coursera/OCR_Series/pretrain/weights_yolo.hdf5', train=False)

    p = 'D:/coursera/OCR_Series/dataset/val/1.jpg'
    img = cv2.imread(p)

    # boxes = model.predict(img)
    # model.plot_box(img, boxes)

    # boxes, scores = model.predict(img)
    # utils.draw_box(img, boxes)