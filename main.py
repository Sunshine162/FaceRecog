from math import ceil
from queue import Queue
import time
from threading import Thread

import cv2
from munch import Munch
import numpy as np
from yaml import safe_load

from inferencer import Yolov5OnnxDetector, PeppaPigOnnxLandmark, \
                       MobileFacenetOnnxRecognizer


# stop flag of multiple threads
END_OF_VIDEO = -1
PREDICT_FINISH = False


def put_frame(video_capture, input_queue, frame_skipping=1):
    global END_OF_VIDEO

    frame_index = 0
    while True:
        ret, image = video_capture.read()
        if not ret:
            input_queue.put((END_OF_VIDEO, None), block=True)
            break
        if frame_index % frame_skipping == 0:
            input_queue.put((frame_index, image), block=True)
        frame_index += 1


def predict_image(cfg, input_queue, output_dict, detector, lmk_model, recognizer):
    global END_OF_VIDEO
    global PREDICT_FINISH

    lmk_bs = cfg.landmark.batch_size
    rec_bs = cfg.recognizer.batch_size

    while not PREDICT_FINISH:
        frame_index, src = input_queue.get(block=True)
        if frame_index == END_OF_VIDEO:
            PREDICT_FINISH = True
            output_dict[frame_index] = None
            break

        det_results = detector.predict(src[None, ...])
        det_boxes, det_confs, det_flags = det_results[0]
        det_boxes = det_boxes[det_flags]

        five_points, lmk_confs, lmk_flags = [], [], []
        for i in range(0, det_boxes.shape[0], lmk_bs):
            part_five_points, part_lmk_confs, part_lmk_flags = \
                lmk_model.predict(src, det_boxes[i : i+lmk_bs, ...])
            five_points.append(part_five_points)
            lmk_confs.append(part_lmk_confs)
            lmk_flags.append(part_lmk_flags)
        five_points = np.concatenate(five_points, axis=0)
        lmk_confs = np.concatenate(lmk_confs, axis=0)
        lmk_flags = np.concatenate(lmk_flags, axis=0)

        _det_boxes = det_boxes[lmk_flags]
        pred_names, rec_confs, rec_flags = [], [], []
        for i in range(0, _det_boxes.shape[0], rec_bs):
            part_pred_names, part_rec_confs, part_rec_flags = \
                recognizer.predict(src, _det_boxes[i : i+rec_bs, ...], 
                                   five_points[i : i+rec_bs, ...])
            pred_names.append(part_pred_names)
            rec_confs.append(part_rec_confs)
            rec_flags.append(part_rec_flags)
        if pred_names:
            pred_names = np.concatenate(pred_names, axis=0)
            rec_confs = np.concatenate(rec_confs, axis=0)
            rec_flags = np.concatenate(rec_flags, axis=0)

        det_boxes = det_boxes.astype(np.int64).tolist()
        five_points = five_points.astype(np.int64).tolist()
        dst = src.copy()
        radius = max(3, int(max(dst.shape[:2]) / 1000))

        i = 0
        for det_box, points, lmk_flag in zip(det_boxes, five_points, lmk_flags):
            # draw detection boundding box
            cv2.rectangle(dst, det_box[0:2], det_box[2:4], (0, 255, 0))

            if not lmk_flag:
                continue

            # draw five key points
            for point in points:
                cv2.circle(dst, point, radius, (0, 0, 255), -1)
            
            pred_name = str(pred_names[i]) if rec_flags[i] else 'Unknown'
            cv2.putText(dst, pred_name, det_box[:2], None, 1, (254, 241, 2), 2)
            
            i += 1
        
        output_dict[frame_index] = dst


def predict_video(video_path_or_cam, cfg):
    # load models
    detector = Yolov5OnnxDetector(cfg.detector)
    lmk_model = PeppaPigOnnxLandmark(cfg.landmark)
    recognizer = MobileFacenetOnnxRecognizer(cfg.recognizer)
    recognizer.set_db(detector, lmk_model)

    ppl_cfg = cfg.pipeline
    # create input_queue for receiving frames
    max_length = ppl_cfg.queue_max_length
    input_queue = Queue(maxsize=max_length)
    # create output_dict for receiving recognition result
    output_dict = {}

    # get basic information of vieo
    video_capture = cv2.VideoCapture(video_path_or_cam)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # read frame from video file or stream
    frame_skipping = ppl_cfg.frame_skipping
    put_thread = Thread(target=put_frame, 
                        args=[video_capture, input_queue, frame_skipping])
    put_thread.start()

    # run models
    predict_threads = []
    for i in range(ppl_cfg.num_workers):
        predict_thread = Thread(
            target=predict_image, 
            args=(cfg, input_queue, output_dict, detector, lmk_model, recognizer)
        )
        predict_thread.start()
        predict_threads.append(predict_thread)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>> display >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    start = time.time()
    time_interval = 1 / fps * frame_skipping
    frame_index = 0
    end_frame_index = ceil(total / frame_skipping) * frame_skipping
    while frame_index != end_frame_index:
        frame_start = time.time()
        while frame_index not in output_dict:
            time.sleep(ppl_cfg.wait_time)
        dst = output_dict.pop(frame_index)

        duration = max(time.time() - frame_start, time_interval)
        curr_fps = 1 / duration
        time.sleep(duration)
        cv2.namedWindow("capture", 0)
        cv2.imshow("capture", dst)
        cv2.waitKey(1)

        frame_index += frame_skipping

    time_spent = time.time() - start
    avg_fps = total / time_spent
    qml = ppl_cfg.queue_max_length
    nw = ppl_cfg.num_workers
    print(f"max_length={qml} workers={nw} frames={total} time={time_spent:.1f} "
          f"fps={avg_fps:.1f}")
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # release resource
    cv2.destroyAllWindows()
    put_thread.join()
    for t in predict_threads:
        t.join()


def parse_args():
    import argparse
    parser = argparse.ArgumentParser("face detection and recognition")
    parser.add_argument('-c', '--config-file',
                        default="configs/end2end_config.yml",
                        type=str, help="path to config file")
    parser.add_argument('-i', '--input-video',
                        default="videos/Trump3.mp4",
                        # default="videos/GodofGamblers.mp4",
                        type=str, help="path to input video")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    with open(args.config_file, 'r') as f:
        cfg = Munch.fromDict(safe_load(f))
    predict_video(args.input_video, cfg)


if __name__ == "__main__":
    main()
