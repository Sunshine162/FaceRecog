import os.path as osp
import time
from typing import List

import cv2
from munch import Munch
import numpy as np
from tqdm import tqdm
from yaml import safe_load

from inferencer import Yolov5OnnxDetectorWithLandmark, MobileFacenetOnnxRecognizer
from tracker.byte_tracker import BYTETracker
from utils import copy_audio, draw_mosaic


class TrackInfo:
    def __init__(self, track_id):
        self.track_id: int = track_id
        self.bboxes: List[np.ndarray] = []
        self.det_scores: List[float] = []
        self.keypoints: List[np.ndarray] = []
        self.features: List[np.ndarray] = []
        self.rec_names: List[str] = []
        self.rec_confidences: List[float] = []
        self.current_index: int = 0  # 当前 track_id 第几次出现
        self.best_name: str = None
        self.best_conf: float = 0.0


class Results:
    def __init__(self, frame_interval):
        self.data = {}
        self.faces = []
        self.frame_interval = frame_interval

    def update_recognition_information(self, track_results, src, recognizer):
        result = []
        for track_obj in track_results:
            track_id = track_obj.track_id
            score = track_obj.score.item()

            if track_id in self.data:
                track_info = self.data[track_id]
            else:
                track_info = TrackInfo(track_id=track_id)
                self.data[track_id] = track_info

            # print(track_obj.tlbr.astype(np.int64).tolist())
            track_info.bboxes.append(track_obj.tlbr.astype(np.int64).tolist().copy())
            track_info.det_scores.append(track_obj.score.item())
            track_info.keypoints.append(track_obj.pt5.astype(np.int64).tolist().copy())

            if track_info.current_index % self.frame_interval == 0:
                pred_names, rec_confs, rec_flags = \
                        recognizer.predict(src, [track_obj.tlbr], [track_obj.pt5])
                pred_name = str(pred_names[0]) if rec_flags[0] else 'Unknown'
                track_info.rec_names.append(pred_name)
                track_info.rec_confidences.append(rec_confs[0].item())
                if track_info.best_conf < score * rec_confs[0].item():
                    track_info.best_name = pred_name
                    track_info.best_conf = score * rec_confs[0].item()
            else:
                track_info.rec_names.append(None)
                track_info.rec_confidences.append(0.0)

            result.append((track_id, track_info.current_index))
            track_info.current_index += 1

        self.faces.append(result)


def load_config_and_models(config_path):
    with open(config_path, 'r') as f:
        cfg = Munch.fromDict(safe_load(f))

    detector = Yolov5OnnxDetectorWithLandmark(cfg.detector)
    recognizer = MobileFacenetOnnxRecognizer(cfg.recognizer)
    tracker = BYTETracker(cfg.tracker)
    recognizer.set_db()
    return cfg, detector, tracker, recognizer


def save_results(video_path, results, save_path, preview=False):
    video_capture=cv2.VideoCapture(video_path)
    total = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = video_capture.get(cv2.CAP_PROP_FPS)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    radius = max(3, int(max(width, height) / 1000))
    wait_time = 1 / src_fps

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    if osp.isfile(save_path):
        os.remove(save_path)
    out = cv2.VideoWriter(save_path, fourcc, src_fps, (width, height))

    for i in tqdm(range(total), desc="Display results"):
        if i >= len(results.faces):
            break

        ret, img = video_capture.read()

        result = results.faces[i]
        for (track_id, index) in result:
            det_box = results.data[track_id].bboxes[index]
            # draw boundding box
            # cv2.rectangle(img, det_box[0:2], det_box[2:4], (0, 255, 0))

            # pt5 = results.data[track_id].keypoints[index]
            # draw five points
            # for point in pt5:
            #     cv2.circle(img, point, radius, (0, 0, 255), -1)

            pred_name = results.data[track_id].best_name
            # draw name of recognition
            # if pred_name != "Unknown":
            #     cv2.putText(img, pred_name, det_box[:2], None, 1, (254, 241, 2), 2)
            if pred_name is None or pred_name == "Unknown":
                draw_mosaic(img, det_box)
            
        out.write(img)
                
        if preview:
            cv2.namedWindow("capture", 0)
            cv2.imshow("capture", img)
            time.sleep(wait_time)

            key = cv2.waitKey(1)
            if key == ord('q'):
                exit()
    
    # write audio
    copy_audio(video_path, save_path)


def predict_video(video_path, cfg, detector, tracker, recognizer):
    video_capture=cv2.VideoCapture(video_path)
    total = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_interval = round(src_fps * cfg.recognizer.recognize.time_interval)
    results = Results(frame_interval)

    # face and landmark detection + face tracking + face recognition
    start = time.time()
    cnt = 0
    for i in tqdm(range(total), desc="Predicting"):
        ret, src = video_capture.read()
        dst = src.copy()
        cnt += 1
        if not ret:
            break

        det_boxes, det_scores, keypoints, flags = detector.predict([src])[0]
        track_results = tracker.update(det_boxes, det_scores, keypoints)
        results.update_recognition_information(track_results, src, recognizer)
        
    duration = time.time() - start
    process_fps = round(cnt / duration)
    print("predict fps:", process_fps)

    return results


def main():
    config_path = 'configs/end2end_config.yml'
    video_path = 'videos/Trump3.mp4'
    save_path = 'videos/Trump3_mosaic.mp4'

    cfg, detector, tracker, recognizer = load_config_and_models(config_path)
    results = predict_video(video_path, cfg, detector, tracker, recognizer)
    save_results(video_path, results, save_path, preview=False)


if __name__ == "__main__":
    main()
