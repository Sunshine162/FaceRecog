from .create_session import create_session
from ..base_recognizer import BaseRecognizer


class MobileFacenetOnnxRecognizer(BaseRecognizer):
    def __init__(self, recognizer_config):
        super(MobileFacenetOnnxRecognizer, self).__init__(recognizer_config)
        
        self.session = create_session(recognizer_config['model_path'], 
                                      recognizer_config['provider'])

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def infer(self, input_data):
        """model inference"""

        return self.session.run(
            self.output_names, {self.input_name: input_data})
