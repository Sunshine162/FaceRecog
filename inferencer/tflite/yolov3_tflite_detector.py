import numpy as np
import tensorflow as tf
from ..base_detector import BaseDetector


class Yolov3TFLiteDetector(BaseDetector):
    def __init__(self, detector_config):
        super(Yolov3TFLiteDetector, self).__init__(detector_config)
        interpreter = tf.lite.Interpreter(detector_config.model_path)
        interpreter.allocate_tensors()
        self.interpreter = interpreter
        self.input_details = interpreter.get_input_details()
        self.output_details = interpreter.get_output_details()
        assert len(self.input_details) == 1
        self.input_name = self.input_details[0]['index']
        self.output_names = [o['index'] for o in self.output_details]

    def infer(self, input_data):
        """model inference"""
        self.interpreter.set_tensor(self.input_name, input_data)
        self.interpreter.invoke()
        output_data = [
            self.interpreter.get_tensor(name) for name in self.output_names]
        return output_data

    def postprocess(self, output_data, metas):
        """process model outputs

        Args: 
            output_data: model outputs
            metas: some information of preprocessing
        
        Return:
            det_results: 
        """
        assert len(output_data) == 2
        bboxes, scores = output_data
        bboxes = xywh2xyxy(bboxes)

        det_results = []
        for one_image_boxes, one_image_scores, meta in zip(bboxes, scores, metas):
            one_image_boxes
            # non maximum suppression
            one_image_data = np.hstack([one_image_boxes, one_image_scores])
            one_image_data = py_nms(
                one_image_data, self.nms_threshold, self.conf_threshold)
            
            one_image_boxes = one_image_data[..., :4]
            one_image_boxes = revert_dets(one_image_boxes, meta['left_pad'], 
                                        meta['top_pad'], meta['scale_factor'])
            one_image_scores = one_image_data[..., 4]
            one_image_flags = self.check_boxes(one_image_boxes, one_image_scores)
            det_results.append((
                one_image_boxes, one_image_scores, one_image_flags))

        return det_results