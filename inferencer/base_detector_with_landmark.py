import cv2
import numpy as np

from utils import xywh2xyxy, py_nms, revert_dets
from .base_detector import BaseDetector


class BaseDetectorWithLandmark(BaseDetector):
    def __init__(self, detector_config):
        super(BaseDetectorWithLandmark, self).__init__(detector_config)
    
    def postprocess(self, output_data, metas):
        """process model outputs

        Args: 
            output_data: model outputs
            metas: some information of preprocessing
        
        Return:
            det_results: bboxes, scores, and keypoints
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
            one_image_data = revert_dets(one_image_data, meta['left_pad'], 
                                          meta['top_pad'], meta['scale_factor'])

            one_image_boxes = one_image_data[..., :4]
            one_image_scores = one_image_data[..., 4]
            one_image_kpts = one_image_data[..., 5:15].reshape(-1, 5, 2)
            valid_flags = self.check_boxes(one_image_boxes, one_image_scores)
            
            det_results.append((
                one_image_boxes, one_image_scores, one_image_kpts, valid_flags))

        return det_results
