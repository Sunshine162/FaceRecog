cfg = {
    'detector': {
        'model_path': 'models/onnx/yolov5n-0.5.onnx',
        'engine': 'onnxruntime',
        'provider': 'cpu',
        'num_threads': 1,
        'input_size': (640, 384),
        'normalization': {
            'mean': [0.0, 0.0, 0.0],
            'std': [255.0, 255.0, 255.0],
        },
        'conf_threshold': 0.72,
        'nms_threshold': 0.1,
        'min_face': 100,
        'max_outputs': 20,
    },

    'landmark': {
        'model_path': 'models/onnx/kps_student.onnx',
        'engine': 'onnxruntime',
        'provider': 'cpu',
        'num_threads': 1,
        'input_size': (256, 256),

        
        'extend': {
            'square': True,
            'side': 'width',  # choices = ['width', 'height', 'max', 'min']
            'ratio': 1.4
        },
        'normalization': {
            'mean': [0.0, 0.0, 0.0],
            'std': [255.0, 255.0, 255.0],
        },
        'five_point_indices': [96, 97, 54, 76, 82],
    },

    'recognizer': {
        'model_path': 'models/onnx/mobilefacenet.onnx',
        'engine': 'onnxruntime',
        'provider': 'cpu',
        'input_size': (112, 112),
        'extend': {
            'square': False,
            'ratio': 1.1
        }
    }
}