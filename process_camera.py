import time
import cv2
import numpy as np
from munch import Munch
from tqdm import tqdm
from yaml import safe_load

from inferencer import Yolov5OnnxDetectorWithLandmark, MobileFacenetOnnxRecognizer
from tracker.byte_tracker import BYTETracker


def load_config_and_models(config_path):
    with open(config_path, 'r') as f:
        cfg = Munch.fromDict(safe_load(f))

    detector = Yolov5OnnxDetectorWithLandmark(cfg.detector)
    recognizer = MobileFacenetOnnxRecognizer(cfg.recognizer)
    tracker = BYTETracker(cfg.tracker)
    recognizer.set_db(detector)
    return cfg, detector, tracker, recognizer


def update_recognition_information(rec_info, track_results, src, recognizer, frame_interval):
    """
    人脸识别执行条件：
        1. 上一次未识别到人名，且距离上一次识别超过15帧 -- DONE
        2. 遇到更优的人脸：人脸更优、更清晰、遮挡更少，表情更正常 -- TODO
    """
    rest_ids = set(rec_info)
    for track_obj in track_results:
        track_id = track_obj.track_id
        score = track_obj.score.item()

        if track_id in rest_ids:
            rest_ids.remove(track_id)
            if rec_info[track_id]['name'] is not None:
                continue
            if rec_info[track_id]['count'] < frame_interval:
                rec_info[track_id]['count'] += 1
                continue
        else:
            rec_info[track_id] = {'name': None, 'count': 0}

        pred_names, rec_confs, rec_flags = \
                    recognizer.predict(src, [track_obj.tlbr], [track_obj.pt5])
        if rec_flags[0]:
            rec_info[track_id]['name'] = str(pred_names[0])
        rec_info[track_id]['count'] = 0

    for track_id in rest_ids:
        del rec_info[track_id]


def predict_image(src, rec_info, detector, tracker, recognizer, frame_interval):
    dst = src.copy()
    radius = max(3, int(max(dst.shape[:2]) / 1000))

    imgs = [src]
    det_boxes, det_scores, keypoints, flags = detector.predict(imgs)[0]

    rest_ids = set(rec_info)
    track_results = tracker.update(det_boxes, det_scores, keypoints)
    update_recognition_information(rec_info, track_results, src, recognizer, frame_interval)
    for track_obj in track_results:
        det_box = track_obj.tlbr.astype(np.int64).tolist()
        score = track_obj.score.item()
        track_id = track_obj.track_id
        cv2.rectangle(dst, det_box[0:2], det_box[2:4], (0, 255, 0))
        for point in track_obj.pt5.astype(np.int64).tolist():
            cv2.circle(dst, point, radius, (0, 0, 255), -1)
        pred_name = rec_info[track_id]['name']
        if pred_name is None:
            pred_name = str(track_id)
        cv2.putText(dst, str(pred_name), det_box[:2], None, 1, (254, 241, 2), 2)

    return dst


def predict_video(video_path, cfg, detector, tracker, recognizer):
    rec_info = {}
    video_capture=cv2.VideoCapture(video_path)
    total = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_interval = round(src_fps * cfg.recognizer.recognize.time_interval)

    for i in tqdm(range(total), desc="Display results"):
        ret, src = video_capture.read()
        if not ret:
            continue

        start = time.time()
        dst = predict_image(src, rec_info, detector, tracker, recognizer, frame_interval)
        duration = time.time() - start

        process_fps = 1 / duration
        cv2.putText(dst, f"fps: {process_fps:.0f}", (20, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), thickness=2)
        cv2.namedWindow("capture", 0)
        cv2.imshow("capture", dst)

        key = cv2.waitKey(1)
        if key == ord('q'):
            exit()


def main():
    cfg, detector, tracker, recognizer = load_config_and_models('configs/end2end_config.yml')
    predict_video('videos/Trump3.mp4', cfg, detector, tracker, recognizer)


if __name__ == "__main__":
    main()
