import os
import argparse
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import onnxruntime as ort


class OnnxModel:
    def __init__(self, onnx_path):
        self.sess = ort.InferenceSession(onnx_path)
    def run(self, x):
        return self.sess.run(None, {'image': x})[0]


def transformation_from_points(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2
    U, S, Vt = np.linalg.svd(points1.T * points2)
    R = (U * Vt).T
    return np.vstack([np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)), 
                     np.matrix([0., 0., 1.])])


def face_alignment(img, landmark):
    landmark = landmark.reshape(2, 5).T
    coord5point = [[38.29459953, 51.69630051],
                   [73.53179932, 51.50139999],
                   [56.02519989, 71.73660278],
                   [41.54930115, 92.3655014 ],
                   [70.72990036, 92.20410156]]

    pts1 = np.float64(np.matrix([[point[0], point[1]] for point in landmark]))
    pts2 = np.float64(np.matrix([[point[0], point[1]] for point in coord5point]))
    M = transformation_from_points(pts1, pts2)
    aligned_image = cv2.warpAffine(img, M[:2], (img.shape[1], img.shape[0]))
    crop_img = aligned_image[0:112, 0:112]
    return crop_img


def process_image(img):
    img = img.transpose(2, 0, 1).astype(np.float32)
    img = (img - 128) / 128
    img = img[None, ...]
    return img


onnx_model = OnnxModel('MobileFaceNet.onnx')
image_file = "first.jpg"
targets = torch.load('facebank/facebank.pth').numpy()
names = np.load('facebank/names.npy')
threshold = 50
show_score = True
image = cv2.imread(image_file)

bboxes = np.load('first_bboxes.npy')
landmarks = np.load('first_landmarks.npy')
embs = []
for landmark in landmarks:
    face = face_alignment(image, landmark=landmark)
    input_data = process_image(face)
    embs.append(onnx_model.run(input_data))


source_embs = np.concatenate(embs, axis=0)  # num_faces x 512
# i.e. 3 x 512 x 1 - 1 x 512 x 2 = 3 x 512 x 2
diff = source_embs[..., None] - targets.transpose(1, 0)[None, ...]
dist = np.sum(np.power(diff, 2), axis=1)
minimum = dist.min(axis=1)
min_idx = dist.argmin(axis=1)
min_idx[minimum > ((threshold-156)/(-80))] = -1  # if no match, set idx to -1
score = minimum
results = min_idx


score_100 = np.clip(score*-80+156,0,100)
image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(image)
font = ImageFont.truetype('utils/simkai.ttf', 20)
for i, b in enumerate(bboxes):
    draw.rectangle([(b[0], b[1]), (b[2], b[3])], outline='blue', width=5)
    draw.text((int(b[0]), int(b[1]-25)), 
              names[results[i] + 1] + ' score:{:.0f}'.format(score_100[i]),
            fill=(255,255,0), font=font)

for p in landmarks:
    for i in range(5):
        draw.ellipse([(p[i] - 2.0, p[i + 5] - 2.0), (p[i] + 2.0, p[i + 5] + 2.0)], 
                     outline='blue')

image.show()
