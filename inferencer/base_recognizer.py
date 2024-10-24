import os
import os.path as osp
import cv2
import numpy as np
from utils import extend_box, face_align, revert_points


class BaseRecognizer:
    def __init__(self, recognizer_config):
        self.input_size = recognizer_config['input_size']
        self.norm_mean = np.array(recognizer_config['normalization']['mean'],
                                  np.float32).reshape[3, 1, 1]
        self.norm_std = np.array(recognizer_config['normalization']['std'],
                                 np.float32).reshape[3, 1, 1]
        self.box_extend_cfg = recognizer_config['extend']
        self.threshold = recognizer_config['threshold']
        self.align_pattern = np.array(recognizer_config['align_pattern'],
                                      np.float32)

        self.db_feat_dir = recognizer_config['db_features']
        self.db_feats = None

    def set_db_feats(self):
        db_feats = []
        for feat_file in os.listdir(self.db_feat_dir):
            feat_path = osp.join(self.db_feat_dir, feat_file)
            if feat_file.endswith('.npy'):
                db_feat = np.load(feat_path)
            elif feat_file.endswith('.'):
                db_feat = np.fromfile(feat_path, np.float32)
            elif feat_file.endswith('.'):
                db_feat = np.loadtxt(feat_path)
            else:
                raise RuntimeError('Can not parse file type:',
                                   osp.splitext(feat_file)[-1])

            db_feats.append(db_feat.reshape(1, -1))
        self.db_feats = np.vstack(db_feats).T

    def preprocess(self, img, det_box, landmarks):
        """data preprocess"""

        img_size = (img.shape[1], img.shape[0])
        input_data = []

        for det_box, one_box_five_points in zip(det_boxes, five_points):
            # crop
            input_box = extend_box(det_box, self.box_extend_cfg, img_size)
            l, t, r, b = input_box  # left, top, right, bottom
            # image_croped = img[t:b, l:r, :]

            # face align
            image_croped = face_align(
                img, landmark, self.align_pattern, self.input_size)
            
            # HWC -> CHW, uint8 -> float32
            image_croped = image_croped.transpose(2, 0, 1).astype(np.float32)

            # normalization
            image_croped = (image_croped - self.norm_mean) / self.norm_std

            input_data.append(image_croped[None, ...])

        input_data = np.concatenate(input_data, axis=0)
        return input_data
    
    def infer(self, input_data):
        """model inference"""

        raise NotImplementedError()
    
    def postprocess(self, output_data):
        norm = np.linalg.norm(output_data[0], axis=1, keepdims=True)
        features = output_data[0] / norm
        return features

    def extract_features(self, img, det_boxes, five_points):
        input_data = self.preprocess(img, det_boxes, five_points)
        output_data = self.infer(input_data)
        features = self.postprocess(output_data)
        return features

    def recognize(self, features):
        similarities = features @ self.db_feats
        flags = (similarities.max(axis=1) > self.threshold).tolist()
        return similarities, flags

    def predict(self, img, det_boxes, five_points):
        features = self.extract_features(img, det_boxes, five_points)
        similarities, flags = self.recognize(features)
        return features, similarities, flags
