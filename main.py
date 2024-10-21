import time
import cv2
import numpy as np

from multiprocessing import Process
from queue import Queue
from threading import Thread

from configs.end2end_config import cfg
from inferencer import Yolov5OnnxDetector, PeppaPigOnnxLandmark
from utils import DataQueue


END = -1


def put_frame(video_path_or_cam, input_queue):
    global END

    vide_capture = cv2.VideoCapture(video_path_or_cam)
    frame_index = 0
    while True:
        ret, image = vide_capture.read()
        if not ret:
            input_queue.put((END, None), block=True)
            break
        input_queue.put((frame_index, image), block=True)
        frame_index += 1


def predict_image(input_queue, output_dict, detector, lmk_model):
    global END

    while True:
        frame_index, src = input_queue.get(block=True)
        if frame_index == END:
            output_dict[frame_index] = None
            break

        det_results = detector.predict(src[None, ...])
        det_boxes, det_confs, det_flags = det_results[0]

        dst = src.copy()
        for i, det_flag in enumerate(det_flags):
            if not det_flag:
                continue

            det_box = det_boxes[i]
            l, t, r, b = det_box.astype(np.int64).tolist()
            cv2.rectangle(dst, (l, t), (r, b), (0, 255, 0))

            five_points, lmk_confs, lmk_flags = lmk_model.predict(src, [det_box])
            if lmk_flags[0]:
                radius = max(3, int(max(dst.shape[:2]) / 1000))
                for point in five_points[0]:
                    point = point.astype(np.int64).tolist()
                    cv2.circle(dst, point, radius, (0, 0, 255), -1)
        
        output_dict[frame_index] = dst


def predict_video(video_path_or_cam):
    max_length = cfg['pipeline']['queue_max_length']
    input_queue = Queue(maxsize=max_length)
    output_dict = {}
    detector = Yolov5OnnxDetector(cfg['detector'])
    lmk_model = PeppaPigOnnxLandmark(cfg['landmark'])

    # read frame from video file or stream
    put_thread = Thread(target=put_frame, args=[video_path_or_cam, input_queue])
    put_thread.start()

    # run models
    predict_threads = []
    for i in range(cfg['pipeline']['num_workers']):
        predict_thread = Thread(
            target=predict_image, 
            args=(input_queue, output_dict, detector, lmk_model)
        )
        predict_thread.start()
        predict_threads.append(predict_thread)

    # display
    global END
    frame_index = 0
    while END not in output_dict:
        start = time.time()
        while frame_index not in output_dict:
            time.sleep(cfg['pipeline']['wait_time'])
        dst = output_dict.pop(frame_index)
        duration = max(time.time() - start, 1e-5)
        fps = 1 / duration
        cv2.putText(dst, f"fps: {fps:.0f}", (20, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), thickness=2)
        cv2.namedWindow("capture", 0)
        cv2.imshow("capture", dst)

        key = cv2.waitKey(1)
        if key == ord('q'):
            return
        
        frame_index += 1


def main():
    predict_video('videos/GodofGamblers.mp4')


if __name__ == "__main__":
    main()
