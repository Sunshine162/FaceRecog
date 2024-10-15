from .create_session import create_session
from ..base_landmark import BaseLandmark



class PeppaPigOnnxLandmark(BaseLandmark):
    def __init__(self, landmark_config):
        super(PeppaPigOnnxLandmark, self).__init__(landmark_config)
        
        self.session = create_session(landmark_config['model_path'],
                                      landmark_config['provider'],
                                      landmark_config['num_threads'])
        print(self.session.get_providers())

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def infer(self, input_data):
        """model inference"""

        return self.session.run(
            self.output_names, {self.input_name: input_data})
