import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def resize_im(w, h, scale=416, max_scale=608):
    f = float(scale) / min(h, w)
    if max_scale is not None:
        if f * max(h, w) > max_scale:
            f = float(max_scale) / max(h, w)
    newW, newH = int(w * f), int(h * f)
    return newW - (newW % 32), newH - (newH % 32)


def nms(boxes, threshold, method='Union'):
    if boxes.size == 0:
        return np.empty((0, 3))
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    s = boxes[:, 4]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    I = np.argsort(s)
    pick = np.zeros_like(s, dtype=np.int16)
    counter = 0
    while I.size > 0:
        i = I[-1]
        pick[counter] = i
        counter += 1
        idx = I[0:-1]
        xx1 = np.maximum(x1[i], x1[idx])
        yy1 = np.maximum(y1[i], y1[idx])
        xx2 = np.minimum(x2[i], x2[idx])
        yy2 = np.minimum(y2[i], y2[idx])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if method is 'Min':
            o = inter / np.minimum(area[i], area[idx])
        else:
            o = inter / (area[i] + area[idx] - inter)
        I = I[np.where(o <= threshold)]
    pick = pick[0:counter]
    return pick


def sort_box(box):
    """
    对box排序,及页面进行排版
        box[index, 0] = x1
        box[index, 1] = y1
        box[index, 2] = x2
        box[index, 3] = y2
        box[index, 4] = x3
        box[index, 5] = y3
        box[index, 6] = x4
        box[index, 7] = y4
    """

    box = sorted(box, key=lambda x: sum([x[1], x[3], x[5], x[7]]))
    return list(box)

def cut_img(im, box, leftAdjustAlph=0.0, rightAdjustAlph=0.0):
    hight, width, channel = im.shape
    angle, w, h, cx, cy = solve(box)
    degree_ = angle * 180.0 / np.pi

    box = (max(1, cx - w / 2 - leftAdjustAlph * (w / 2)),  ##xmin
           cy - h / 2,  ##ymin
           min(cx + w / 2 + rightAdjustAlph * (w / 2), width - 1),  ##xmax
           cy + h / 2)  ##ymax
    newW = box[2] - box[0]
    newH = box[3] - box[1]

    clip_img = im[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :]
    # clip_img = clip_img.astype(np.int)

    # cx = int((box[2]-box[0])/2)
    # cy = int((box[3]-box[1])/2)
    # clip_h, clip_w, channel = clip_img.shape
    # M = cv2.getRotationMatrix2D((cx, cy), 90, 1.0)
    # rotated_img = cv2.warpAffine(clip_img, M, (clip_w, clip_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # tmpImg = im.rotate(degree_, center=(cx, cy)).crop(box)
    # box = {'cx': cx, 'cy': cy, 'w': newW, 'h': newH, 'degree': degree_, }
    # return tmpImg, box

    return clip_img

def solve(box):
    """
    绕 cx,cy点 w,h 旋转 angle 的坐标
    x = cx-w/2
    y = cy-h/2
    x1-cx = -w/2*cos(angle) +h/2*sin(angle)
    y1 -cy= -w/2*sin(angle) -h/2*cos(angle)

    h(x1-cx) = -wh/2*cos(angle) +hh/2*sin(angle)
    w(y1 -cy)= -ww/2*sin(angle) -hw/2*cos(angle)
    (hh+ww)/2sin(angle) = h(x1-cx)-w(y1 -cy)

    """
    x1, y1, x2, y2, x3, y3, x4, y4 = box[:8]
    cx = (x1 + x3 + x2 + x4) / 4.0
    cy = (y1 + y3 + y4 + y2) / 4.0
    w = (np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) + np.sqrt((x3 - x4) ** 2 + (y3 - y4) ** 2)) / 2
    h = (np.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2) + np.sqrt((x1 - x4) ** 2 + (y1 - y4) ** 2)) / 2
    # x = cx-w/2
    # y = cy-h/2

    sinA = (h * (x1 - cx) - w * (y1 - cy)) * 1.0 / (h * h + w * w) * 2
    if abs(sinA) > 1:
        angle = None
    else:
        angle = np.arcsin(sinA)
    return angle, w, h, cx, cy

def draw_box(img, boxes):
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    for xmin, ymin, xmax, ymax in boxes:
        rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, edgecolor='r', linewidth=1, facecolor='none')
        ax.add_patch(rect)
    plt.show()