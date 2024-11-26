import tensorflow as tf
from ..base_recognizer import BaseRecognizer


class MobileFacenetTFLiteRecognizer(BaseRecognizer):
    def __init__(self, recognizer_config):
        super(MobileFacenetTFLiteRecognizer, self).__init__(recognizer_config)
        interpreter = tf.lite.Interpreter(recognizer_config.model_path)
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
