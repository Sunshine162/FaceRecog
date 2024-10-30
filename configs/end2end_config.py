cfg = {
    'detector': {
        'model_path': 'models/onnx/yolov5n-0.5.onnx',
        'engine': 'onnxruntime',
        'provider': 'cpu',
        'batch_size': 1,  # detector only support batchsize=1
        'input_size': (640, 384),
        'normalization': {
            'mean': [0.0, 0.0, 0.0],
            'std': [255.0, 255.0, 255.0],
        },
        'conf_threshold': 0.5,
        'nms_threshold': 0.6,
        'min_face': 100,
        'max_outputs': 20,
    },

    'landmark': {
        # 'model_path': 'models/onnx/kps_student_256.onnx',
        'model_path': 'models/onnx/kps_student_256_dyn_batch.onnx',
        'engine': 'onnxruntime',
        'provider': 'cpu',
        'batch_size': 1,
        'max_batch_size': 32,
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
        'conf_threshold': 0.6,
    },

    'recognizer': {
        # 'model_path': 'models/onnx/MobileFaceNet_L2.onnx',
        'model_path': 'models/onnx/MobileFaceNet_L2_dyn_batch.onnx',
        'engine': 'onnxruntime',
        'provider': 'cpu',
        'batch_size': 1,
        'max_batch_size': 32,
        'input_size': (112, 112),
        'extend': {
            'square': False,
            'ratio': 1.1
        },
        'normalization': {
            'mean': [128.0, 128.0, 128.0],
            'std': [128.0, 128.0, 128.0],
        },
        'align': {
            'input_crop': False,
            'pattern': [
                [38.29459953, 51.69630051],
                [73.53179932, 51.50139999],
                [56.02519989, 71.73660278],
                [41.54930115, 92.3655014 ],
                [70.72990036, 92.20410156]
            ]
        },
        'recognize': {
            'db_images': 'faces/images/',
            'db_features': 'faces/features',
            'judge_mode': 'L2',  # choices = ['L2', 'cosine']
            'threshold': 50,
        },
    },

    'pipeline': {
        'queue_max_length': 24,
        'num_workers': 16,
        'wait_time': 1e-4,
        'frame_skipping': 1,  # process 1 frame every 2 frames
    }
}
