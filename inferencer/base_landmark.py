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
        self.conf_threshold = landmark_config['conf_threshold']
    
    def preprocess(self, img, det_boxes):
        """data preprocess"""

        img_size = (img.shape[1], img.shape[0])
        input_data = []
        metas = []
            
        for det_box in det_boxes:
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

            input_data.append(image_croped[None, ...])
            metas.append({'input_box': input_box})

        input_data = np.concatenate(input_data, axis=0)
        return input_data, metas
    
    def infer(self, input_data):
        """model inference"""

        raise NotImplementedError()
    
    def postprocess(self, output_data, metas):
        """process model outputs

        Args: 
            output_data: model outputs
            meta: some information of preprocessing
        
        Return:
            boxes: prediction boundding boxes
            confidences: confidence of boxes
            flags: is box valid
        """
        
        assert len(output_data) == 2
        landmarks, scores = output_data
        num_faces = landmarks.shape[0]

        landmarks = landmarks.reshape(num_faces, -1, 2)
        five_points = landmarks[:, self.five_point_indices, :]
        for i in range(landmarks.shape[0]):
            five_points[i] = revert_points(five_points[i], metas[i]['input_box'])

        confidences = scores.mean(axis=1)
        flags = (confidences > self.conf_threshold).tolist()

        return five_points, confidences, flags
        
    def predict(self, imgs, det_boxes):
        """predict one image by end2end"""

        input_data, metas = self.preprocess(imgs, det_boxes)
        output_data = self.infer(input_data)
        five_points, confidences, flags = self.postprocess(output_data, metas)
        return five_points, confidences, flags
