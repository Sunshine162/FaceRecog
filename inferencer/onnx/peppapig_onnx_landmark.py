import onnxruntime as ort

from ..base_landmark import BaseLandmark


class PeppaPigOnnxLandmark(BaseLandmark):
    def __init__(self, landmark_config):
        super(PeppaPigOnnxLandmark, self).__init__(landmark_config)
        
        device = landmark_config['device']
        assert device == 'cpu' or device == 'cuda'
        if device == 'cuda':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = [ 'CPUExecutionProvider' ]

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = landmark_config['num_threads']

        self.session = ort.InferenceSession(
            landmark_config['model_path'], sess_options, providers=providers
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def infer(self, input_data):
        """model inference"""

        return self.session.run(
            self.output_names, {self.input_name: input_data})
