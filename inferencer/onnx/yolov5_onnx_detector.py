import onnxruntime as ort

from ..base_detector import BaseDetector


class Yolov5OnnxDetector(BaseDetector):
    def __init__(self, detector_config):
        super(Yolov5OnnxDetector, self).__init__(detector_config)
        
        device = detector_config['device']
        assert device == 'cpu' or device == 'cuda'
        if device == 'cuda':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = [ 'CPUExecutionProvider' ]

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = detector_config['num_threads']

        self.session = ort.InferenceSession(
            detector_config['model_path'], sess_options, providers=providers
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def infer(self, input_data):
        """model inference"""

        return self.session.run(
            self.output_names, {self.input_name: input_data})
    