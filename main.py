from queue import Queue
import time
from threading import Thread

import cv2
import numpy as np

from configs.end2end_config import cfg
from inferencer import Yolov5OnnxDetector, PeppaPigOnnxLandmark, \
                       MobileFacenetOnnxRecognizer


# stop flag of multiple threads
LAST_FRAME = -1
PREDICT_FINISH = False


def put_frame(video_capture, input_queue):
    global LAST_FRAME

    frame_index = 0
    while True:
        ret, image = video_capture.read()
        if not ret:
            input_queue.put((LAST_FRAME, None), block=True)
            break
        input_queue.put((frame_index, image), block=True)
        frame_index += 1


def predict_image(input_queue, output_dict, detector, lmk_model, recognizer):
    global LAST_FRAME
    global PREDICT_FINISH

    while not PREDICT_FINISH:
        frame_index, src = input_queue.get(block=True)
        if frame_index == LAST_FRAME:
            PREDICT_FINISH = True
            output_dict[frame_index] = None
            break

        det_results = detector.predict(src[None, ...])
        det_boxes, det_confs, det_flags = det_results[0]

        dst = src.copy()
        for i, det_flag in enumerate(det_flags):
            det_box = det_boxes[i]
            if not det_flag:
                continue

            # predict landmark
            five_points, lmk_confs, lmk_flags = lmk_model.predict(src, [det_box])
            if not lmk_flags[0]:
                continue

            # draw detection boundding box
            l, t, r, b = det_box.astype(np.int64).tolist()
            cv2.rectangle(dst, (l, t), (r, b), (0, 255, 0))

            # draw five key points
            radius = max(3, int(max(dst.shape[:2]) / 1000))
            for point in five_points[0]:
                point = point.astype(np.int64).tolist()
                cv2.circle(dst, point, radius, (0, 0, 255), -1)
            
            # recognize face
            pred_names, confs, flags = recognizer.predict(src, [det_box], 
                                                          five_points)
            pred_name = str(pred_names[0]) if flags[0] else 'Unknown'
            cv2.putText(dst, pred_name, 
                        det_box[:2].astype(np.int64).tolist(), None, 
                        1, (254, 241, 2), 2)
        
        output_dict[frame_index] = dst


def predict_video(video_path_or_cam):
    # load models
    detector = Yolov5OnnxDetector(cfg['detector'])
    lmk_model = PeppaPigOnnxLandmark(cfg['landmark'])
    recognizer = MobileFacenetOnnxRecognizer(cfg['recognizer'])
    recognizer.set_db(detector, lmk_model)

    # create input_queue for receiving frames
    max_length = cfg['pipeline']['queue_max_length']
    input_queue = Queue(maxsize=max_length)
    # create output_dict for receiving recognition result
    output_dict = {}

    # get basic information of vieo
    video_capture = cv2.VideoCapture(video_path_or_cam)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # read frame from video file or stream
    put_thread = Thread(target=put_frame, args=[video_capture, input_queue])
    put_thread.start()

    # run models
    predict_threads = []
    for i in range(cfg['pipeline']['num_workers']):
        predict_thread = Thread(
            target=predict_image, 
            args=(input_queue, output_dict, detector, lmk_model, recognizer)
        )
        predict_thread.start()
        predict_threads.append(predict_thread)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>> display >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    start = time.time()
    time_interval = 1 / fps 
    frame_index = 0
    while frame_index != total:
        frame_start = time.time()
        while frame_index not in output_dict:
            time.sleep(cfg['pipeline']['wait_time'])
        dst = output_dict.pop(frame_index)

        duration = max(time.time() - frame_start, time_interval)
        curr_fps = 1 / duration
        time.sleep(duration)
        cv2.namedWindow("capture", 0)
        cv2.imshow("capture", dst)
        cv2.waitKey(1)

        frame_index += 1

    time_spent = time.time() - start
    avg_fps = total / time_spent
    qml = cfg['pipeline']['queue_max_length']
    nw = cfg['pipeline']['num_workers']
    print(f"max_length={qml} workers={nw} frames={total} time={time_spent:.1f} "
          f"fps={avg_fps:.1f}")
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # release resource
    cv2.destroyAllWindows()
    put_thread.join()
    for t in predict_threads:
        t.join()


def main():
    predict_video('videos/Trump3.mp4')


if __name__ == "__main__":
    main()
