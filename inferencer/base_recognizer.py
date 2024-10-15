import numpy as np

class BaseRecognizer:
    def __init__(self, model_path, threshold=0.5, db_feat_file=None, db_id_file=None):
        self.threshold = 0.5
        self.db_feats = np.load(db_feat_file)
        self.db_ids = [line.strip() 
                       for line in open(db_id_file, 'w') if line.strip()]
        
    def preprocess(self, img, det_box, landmarks):
        input_data = None
        meta = None
        return input_data, meta
    
    def infer(self, input_data):
        raise NotImplementedError()
    
    def postprocess(self, output_data):
        if feature.ndim == 1:
            feature = np.expand_dims(feature, 0)
        assert feature.ndim == 2 and feature.shape[0] == 1

        feature = feature / np.linalg.norm(output_data)
        return feature
        
    def predict(self, img, det_box, landmarks):
        input_data, meta = self.preprocess(img, det_box, landmarks)
        output_data = self.infer(input_data)
        feature = self.postprocess(output_data)
