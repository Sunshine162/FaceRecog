# the progress of processing a single human face
PASS = 0
LANDMARK_TO_DO = 1
LANDMARK_DOING = 2
RECOGNIZE_TO_DO = 3
RECOGNIZE_DOING = 4
DONE = 5


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
        self.person_name = None
        self.person_id = None
        self.similiarity = None
        self.rec_flag = None

        self.progress = None


class Frame:
    def __init__(self, id, img, det_results):
        self.image = img
        self.id = id

        self.faces = []
        det_boxes, det_scores, det_flags = det_results
        for i, (box, score, flag) in enumerate(zip(det_boxes, det_scores)):
            face = Face()
            face.id = i
            face.det_box = box
            face.det_score = score
            face.det_flag = flag
            face.progress = LANDMARK_TO_DO if flag else PASS

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

