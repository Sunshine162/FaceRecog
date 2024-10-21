import cv2
import numpy as np

from utils import xywh2xyxy, py_nms, revert_boxes


class BaseDetector:
    def __init__(self, detector_config):
        """
        Args:
            detector_config: a dict, contain keys:
                model_path: 
                input_size: width and height of model input
                norm_cfg: mean and std for normalization
                conf_threshold:
                nms_threshold:
                min_face: 
                max_outputs: 
        """

        self.input_size = detector_config['input_size']
        self.norm_mean = np.array(detector_config['normalization']['mean'], 
                                  np.float32).reshape(3, 1, 1)
        self.norm_std = np.array(detector_config['normalization']['std'], 
                                 np.float32).reshape(3, 1, 1)
        self.conf_threshold = detector_config['conf_threshold']
        self.nms_threshold = detector_config['nms_threshold']
        self.min_face = detector_config['min_face']
        self.max_outputs = detector_config['max_outputs']
    
    def preprocess(self, imgs):
        """convert raw image to input data of model.

        Args:
            imgs: raw images readed by OpenCV, three channel, numpy.ndarray
        
        Return:
            input_data: input data for model
            metas: list of dict, like: 
                [{'scale_factor': 0.5, 'left_pad': 0, 'top_pad': 200}, ...]
        """
        input_data = []
        metas = []
        for img in imgs:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, c = img.shape

            # scale image
            scale_factor = min(self.input_size[0] / h, self.input_size[1] / w)
            img = cv2.resize(img, (int(w * scale_factor), int(h * scale_factor)))
            h, w, c = img.shape

            # letter box
            dh = (self.input_size[1] - h) / 2
            dw = (self.input_size[0] - w) / 2
            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
            img = cv2.copyMakeBorder(img, top, bottom, 
                left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))  # add border

            # HWC -> CHW, uint8 -> float32
            img = img.transpose(2, 0, 1).astype(np.float32)

            # normalization
            img = (img - self.norm_mean) / self.norm_std

            input_data.append(np.expand_dims(img, axis=0))
            metas.append(dict(
                scale_factor=scale_factor, left_pad=left, top_pad=top))
        
        input_data = np.concatenate(input_data, axis=0)

        return input_data, metas
    
    def infer(self, input_data):
        """model inference"""
        
        raise NotImplementedError()
    
    def postprocess(self, output_data, metas):
        """process model outputs

        Args: 
            output_data: model outputs
            metas: some information of preprocessing
        
        Return:
            det_results: 
        """
        output_data = output_data[0]
        if output_data.ndim != 3:
            raise RuntimeError('Can not parse the output data')
        
        output_data[..., :4] = xywh2xyxy(output_data[..., :4])

        det_results = []
        for one_image_data, meta in zip(output_data, metas):
            # non maximum suppression
            one_image_data = py_nms(
                one_image_data, self.nms_threshold, self.conf_threshold)
            
            one_image_boxes = one_image_data[..., :4]
            one_image_boxes = revert_boxes(one_image_boxes, meta['left_pad'], 
                                        meta['top_pad'], meta['scale_factor'])
            one_image_scores = one_image_data[..., 4]

            one_image_boxes, one_image_scores = self.filter_boxes(
                one_image_boxes, one_image_scores)
            one_image_flags = [True] * one_image_boxes.shape[0]
            det_results.append((
                one_image_boxes, one_image_scores, one_image_flags))

        return det_results


    def filter_boxes(self, boxes, scores=None, sorted=True):
        """find the top_k max bboxes, and filter the small face"""

        if self.min_face:
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            select_index = area > self.min_face
            if scores is not None:
                boxes = boxes[select_index, :]
        
        if self.max_outputs:
            if boxes.shape[0] > self.max_outputs:
                if not sorted:
                    descending_indices = scores.argsort()[::-1]
                    boxes = boxes[descending_indices, :]
                    scores = scores[descending_indices]
                
                boxes = boxes[:self.max_outputs, :]
                if scores is not None:
                    scores = scores[:self.max_outputs]

        if scores is None:
            return boxes
        return boxes, scores

    def predict(self, imgs):
        """predict one image by end2end"""

        input_data, metas = self.preprocess(imgs)
        output_data = self.infer(input_data)
        det_results = self.postprocess(output_data, metas)
        return det_results
