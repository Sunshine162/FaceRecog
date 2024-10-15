import cv2
import numpy as np
from utils import extend_box, revert_points


class BaseLandmark:
    def __init__(self, landmark_config):
        """
        Args:
            landmark_config: a dict, contain keys:
                model_path: 
                input_size: width and height of model input
                normalization: contain mean and std for normalization
                extend: box extending configuration, a dict
                five_point_indices: 
        """

        self.input_size = landmark_config['input_size']
        self.norm_mean = np.array(landmark_config['normalization']['mean'], 
                                  np.float32).reshape(3, 1, 1)
        self.norm_std = np.array(landmark_config['normalization']['std'],
                                 np.float32).reshape(3, 1, 1)
        self.box_extend_cfg = landmark_config['extend']
        self.five_point_indices = landmark_config['five_point_indices']
    
    def preprocess(self, img, det_box):
        """data preprocess"""

        img_size = (img.shape[1], img.shape[0])

        # crop
        input_box = extend_box(det_box, self.box_extend_cfg, img_size)
        l, t, r, b = input_box   # left, top, right, bottom
        image_croped = img[t:b, l:r, :]

        # resize
        image_croped = cv2.resize(image_croped, (self.input_size[1],
                                             self.input_size[0]))
        # HWC -> CHW, uint8 -> float32
        image_croped = image_croped.transpose((2, 0, 1)).astype(np.float32)

        # normalization
        image_croped = (image_croped - self.norm_mean) / self.norm_std

        input_data = np.expand_dims(image_croped, axis=0)
        meta = {'input_box': input_box}

        return input_data, meta
    
    def infer(self, input_data):
        """model inference"""

        raise NotImplementedError()
    
    def postprocess(self, output_data, meta):
        """process model outputs

        Args: 
            output_data: model outputs
            meta: some information of preprocessing
        
        Return:
            boxes: prediction boundding boxes
            scores: confidence of boxes
        """
        
        assert len(output_data) == 2
        landmarks, scores = output_data
        assert landmarks.shape[0] == scores.shape[0] == 1

        landmarks = landmarks.reshape(-1, 2)
        scores = scores.flatten()

        five_points = landmarks[self.five_point_indices, :]
        five_points = revert_points(five_points, meta['input_box'])

        confidence = scores.mean()

        return five_points, confidence
        
    def predict(self, img, det_box):
        """predict one image by end2end"""

        input_data, meta = self.preprocess(img, det_box)
        output_data = self.infer(input_data)
        five_points, score = self.postprocess(output_data, meta)
        return five_points, score
