from collections import deque
from ctypes import cast, py_object
from dataclasses import dataclass
from queue import Queue
import threading
from typing import Sequence, Union
import numpy as np


# the progress of processing a single human face
PASS = 0
LANDMARK_TO_DO = 1
LANDMARK_DOING = 2
RECOGNIZE_TO_DO = 3
RECOGNIZE_DOING = 4
DONE = 5


@dataclass
class Face:
    face_id: int

    det_box: Union[Sequence, np.ndarray]
    det_score: float
    det_flag: bool

    keypoints: Union[Sequence, np.ndarray]
    kpt_score: float
    kpt_flag = bool

    feature: np.ndarray
    person_name: str
    person_id = int
    confidence = float
    rec_flag = bool

    progress = int


class Frame:
    def __init__(self, frame_id, img):
        self.image = img
        self.frame_id = frame_id
        self.faces = []
        self.is_finish = False

    def put_face_boxes(self, det_boxes, det_scores, det_flags):
        for i, (box, score, flag) in enumerate(zip(det_boxes, det_scores, det_flag)):
            face = Face()
            face.id = i
            face.det_box = box
            face.det_score = score
            face.det_flag = flag
            face.progress = LANDMARK_TO_DO if flag else PASS
            self.faces.append(face)

    def put_keypoints(self, start, keypoints, kpt_scores, kpt_flags):
        cur_idx = start
        for one_face_keypoints, score, flag in zip(keypoints, kpt_scores, kpt_flags):
            while self.faces[cur_idx].progress != LANDMARK_DOING:
                cur_idx += 1
            self.faces[cur_idx].keypoints = one_face_keypoints
            self.faces[cur_idx].kpt_score = score
            self.faces[cur_idx].kpt_flag = flag
            self.faces[cur_idx].progress = RECOGNIZE_TO_DO if flag else PASS

    def put_recog_results(self, start, recog_ids, recog_names, recog_confs, recog_flags):
        cur_idx = start
        for pid, pname, conf, flag in zip(recog_ids, recog_names, recog_confs, recog_flags):
            while self.faces[cur_idx].progress != RECOGNIZE_DOING:
                cur_idx += 1
            self.faces[cur_idx].person_id = pid
            self.faces[cur_idx].person_name = pname
            self.faces[cur_idx].confidence = conf
            self.faces[cur_idx].rec_flag = flag
            self.faces[cur_idx].progress = DONE if flag else PASS
        
        if self.faces[-1].progress == DONE or self.faces[-1].progress == PASS:
            self.is_finish = True


class FrameQueue():
    def __init__(self, max_length=30, det_batch_size=1, lmk_batch_size=1, rec_batch_size=1):
        self.max_length = max_length
        self.frame_queue = Queue(maxsize=max_length)
        self.address_list = []
        self.det_start = {"frame_index": 0}
        self.lmk_start = {"frame_index": 0, "face_index": 0}
        self.rec_start = {"frame_index": 0, "face_index": 0}
        self.get_lock = threading.Lock()
        self.det_lock = threading.Lock()
        self.lmk_lock = threading.Lock()
        self.rec_lock = threading.Lock()
        self.det_batch_size = det_batch_size
        self.lmk_batch_size = lmk_batch_size
        self.rec_batch_size = rec_batch_size
    
    def put_frame(self, frame_id, img):
        frame_obj = Frame(frame_id, img)
        self.frame_queue.put([frame_id, img], block=True, timeout=3)
        assert len(self.address_list) < self.max_length
        self.address_list.append(id(frame_obj))
    
    def get_result(self):
        """
        1. self.frame_queue 为空
        2. 最早的 frame 处理中
        3. 最早的 frame 处理完毕
        """

        if self.frame_queue.empty():
            return None
        
        frame = cast(self.address_list[0], py_object).value

        timeout = 3
        start_time = time.time()
        wait_time = 0
        while not frame.is_finish:
            time.sleep(0.00001)
            wait_time += (time.time() - start_time)
            if wait_time > timeout:
                raise RuntimeError("get result timeout!!!")

        frame = self.queue.get(block=True, timeout=timeout)
        self.address_list.pop(0)

        with self.get_lock:
            self.det_start['frame_index'] -= 1
            self.lmk_start['frame_index'] -= 1
            self.rec_start['frame_index'] -= 1

        return frame
    
    def get_batch_images(self):
        """互斥"""
        with self.det_lock:
            frame_info = []
            images = []
            frame_idx = self.det_start['frame_index']
            
            for address in self.address_list[frame_idx:]:
                frame = cast(frame_address, py_object).value
                frame_info.append(address)
                images.append(frame.image)

                if len(images) == self.det_batch_size:
                    break

        return frame_info, images
    
    def get_batch_boxes(self):
        """互斥"""
        with self.lmk_lock:
            face_info = []
            boxes = []
            frame_idx = self.lmk_start['frame_index']
            face_idx = self.lmk_start['face_index']
            new_frame_idx = frame_idx
            new_face_idx = face_idx

            for address in self.address_list[frame_idx:]:
                frame = cast(frame_address, py_object).value
                
                cnt = 0
                for face in frame.faces[face_idx:]:
                    if face.progress == LANDMARK_TO_DO:
                        boxes.append(face.box)
                        cnt += 1
                    new_face_idx = (new_face_idx + 1) % len(frame.faces)
                    if len(boxes) == self.lmk_batch_size:
                        break
                    
                if cnt:
                    face_info.append((address, face_idx, cnt))

                face_idx = new_face_idx
                if new_face_idx == 0:
                    new_frame_idx += 1
                
                if len(boxes) == self.lmk_batch_size:
                    break
            
            self.lmk_start['frame_index'] = new_frame_idx
            self.lmk_start['face_index'] = new_face_idx

        return face_info, boxes

    def get_batch_keypoints(self):
        """互斥"""
        with self.rec_lock:
            face_info = []
            boxes = []
            keypoints = []
            frame_idx = self.rec_start['frame_index']
            face_idx = self.rec_start['face_index']
            new_frame_idx = frame_idx
            new_face_idx = face_idx

            for address in self.address_list[frame_idx:]:
                frame = cast(frame_address, py_object).value
                
                cnt = 0
                for face in frame.faces[face_idx:]:
                    if face.progress == RECOGNIZE_TO_DO:
                        boxes.append(face.box)
                        keypoints.append(face.keypoints)
                        cnt += 1
                    new_face_idx = (new_face_idx + 1) % len(frame.faces)
                    if len(boxes) == self.rec_batch_size:
                        break
                    
                if cnt:
                    face_info.append((address, face_idx, cnt))

                face_idx = new_face_idx
                if new_face_idx == 0:
                    new_frame_idx += 1
                
                if len(boxes) == self.rec_batch_size:
                    break
            
            self.rec_start['frame_index'] = new_frame_idx
            self.rec_start['face_index'] = new_face_idx

        return face_info, boxes, keypoints

    def put_detect_results(self, det_results):
        frame_address, det_boxes, det_scores, det_flags = det_results
        frame = cast(frame_address, py_object).value
        frame.put_face_boxes(det_boxes, det_scores, det_flags)

    def put_batch_keypoints(self, batch_keypoints):
        """一对一、一拆多、多合一"""

        face_info, keypoints, kpt_scores, kpt_flags = batch_keypoints
        _start = 0
        for (frame_address, start, size) in face_info:
            frame = cast(frame_address, py_object).value
            frame.put_keypoints(start, 
                                keypoints[_start: _start + size], 
                                kpt_scores[_start: _start + size], 
                                kpt_flags[_start: _start + size])
            _start += size

    def put_batch_recog_results(self, batch_recog_results):
        """一对一、一拆多、多合一"""

        face_info, recog_ids, recog_names, recog_confs, recog_flags = batch_recog_results
        _start = 0
        for (frame_address, start, size) in face_info:
            frame = cast(frame_address, py_object).value
            frame.put_recog_results(start,
                                    recog_ids[_start: _start + size],
                                    recog_names[_start: _start + size], 
                                    recog_confs[_start: _start + size], 
                                    recog_flags[_start: _start + size])
            _start += size

