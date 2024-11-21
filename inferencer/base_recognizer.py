import os
import os.path as osp
import cv2
import numpy as np
from utils import extend_box, extract_face_embedding, face_align, revert_points


class BaseRecognizer:
    def __init__(self, recognizer_config):
        self.input_size = recognizer_config['input_size']
        self.norm_mean = np.array(recognizer_config['normalization']['mean'],
                                  np.float32).reshape(3, 1, 1)
        self.norm_std = np.array(recognizer_config['normalization']['std'],
                                 np.float32).reshape(3, 1, 1)
        self.box_extend_cfg = recognizer_config.get('extend', None)
        
        self.input_crop = recognizer_config['align']['input_crop']
        self.align_pattern = np.array(
            recognizer_config['align']['pattern'], np.float32)

        recog_cfg = recognizer_config['recognize']
        assert recog_cfg['judge_mode'] in ['L2', 'cosine']
        self.judge_mode = recog_cfg['judge_mode']
        self.threshold = recog_cfg['threshold']
        self.db_image_dir = recog_cfg['db_images']
        self.db_feat_dir = recog_cfg['db_features']
        self.db_feats = None
        self.db_names = None

    def set_db(self, detector=None, lmk_model=None):
        os.makedirs(self.db_feat_dir, exist_ok=True)
        db_feats = []
        names = []
        for image_file in os.listdir(self.db_image_dir):
            name = image_file.rsplit('-', 1)[0]
            names.append(name)

            image_path = osp.join(self.db_image_dir, image_file)
            feat_path = osp.join(self.db_feat_dir, 
                                 image_file.replace('.jpg', '.npy'))
            
            if osp.isfile(feat_path):
                db_feat = np.load(feat_path)
            else:
                db_feat = extract_face_embedding(image_path, detector, 
                                                 lmk_model, self)
                np.save(feat_path, db_feat)

            db_feats.append(db_feat.reshape(1, -1))
        
        db_feats = np.vstack(db_feats).T
        if self.judge_mode == 'L2':
            db_feats = db_feats[None, ...]

        self.db_feats = db_feats
        self.names = np.array(names)

    def preprocess(self, img, det_boxes, landmarks):
        """data preprocess"""

        img_size = (img.shape[1], img.shape[0])
        input_data = []

        for det_box, one_box_five_points in zip(det_boxes, landmarks):
            # crop
            if self.input_crop and self.box_extend_cfg:
                input_box = extend_box(det_box, self.box_extend_cfg, img_size)
            else:
                input_box = None

            # face align
            image_croped = face_align(img, one_box_five_points, 
                                      self.align_pattern, self.input_size, 
                                      self.input_crop, input_box)
            
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
        features = output_data[0]
        if self.judge_mode == 'cosine':
            norm = np.linalg.norm(features, axis=1, keepdims=True)
            features = features / norm
        return features

    def extract_features(self, img, det_boxes, five_points):
        input_data = self.preprocess(img, det_boxes, five_points)
        output_data = self.infer(input_data)
        features = self.postprocess(output_data)
        return features

    def recognize(self, features):
        if self.judge_mode == 'cosine':
            similarities = features @ self.db_feats
            indices = similarities.argmax(axis=1)
            pred_names = self.names[indices]
            confs = similarities.max(axis=1)
            flags = (confs > self.threshold)
            
        elif self.judge_mode == 'L2':
            diff = features[..., None] - self.db_feats
            dist = np.sum(np.power(diff, 2), axis=1)
            indices = dist.argmin(axis=1)
            confs = dist.min(axis=1)
            confs = np.clip(confs * -80 + 156, 0, 100)
            pred_names = self.names[indices]
            flags = (confs >= self.threshold)
        
        else:
            raise ValueError('Unknown judge mode:', self.judge_mode)

        return pred_names, confs, flags

    def predict(self, img, det_boxes, five_points):
        features = self.extract_features(img, det_boxes, five_points)
        return self.recognize(features)
