from .create_session import create_session
from ..base_detector_with_landmark import BaseDetectorWithLandmark


class Yolov5OnnxDetectorWithLandmark(BaseDetectorWithLandmark):
    def __init__(self, detector_config):
        super(Yolov5OnnxDetectorWithLandmark, self).__init__(detector_config)
        
        self.session = create_session(detector_config['model_path'], 
                                      detector_config['provider'])

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def infer(self, input_data):
        """model inference"""

        return self.session.run(
            self.output_names, {self.input_name: input_data})
