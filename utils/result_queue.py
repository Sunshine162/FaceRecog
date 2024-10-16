class Face:
    def __init__(self):
        self.id = None
        self.det_box = None
        self.det_score = None
        self.det_flag = None
        self.keypoints = None
        self.kpt_score = None
        self.kpt_flag = None
        self.feature = None
        self.name = None
        self.person_id = None
        self.finish = False


class Frame:
    def __init__(self, img, id):
        self.image = img
        self.id = id
        self.faces = []
        self.finish = False


class ResultQueue():
    def __init__(self):
        pass
    
    def put_frame_with_boxes(self):
        pass
    
    def get_frame_with_results(self):
        """阻塞"""
        pass
    
    def read_batch_boxes(self):
        pass

    def read_batch_keypoints(self):
        pass
    
    def store_batch_keypoints(self):
        pass
    
    def store_batch_names(self):
        pass

