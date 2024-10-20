import time
import cv2
import numpy as np

from concurrent.futures import ThreadPoolExecutor, wait,\
    ALL_COMPLETED, FIRST_COMPLETED, as_completed
from configs.end2end_config import cfg
from ctypes import cast, py_object
from inferencer import Yolov5OnnxDetector, PeppaPigOnnxLandmark
from utils import DataQueue


def put_frame(data_queue, frame_id, image):
    data_queue.put_frame(frame_id, image)


def detect_face(data_queue, detector):
    frame_info, images = data_queue.get_batch_images()
    if images:
        det_results = detector.predict(images)
        data_queue.put_batch_detect_results(frame_info, det_results)


def detect_keypoint(data_queue, landmark_model):
    face_info, boxes = data_queue.get_batch_boxes()
    if boxes:
        new_info = []
        for (frame_address, start, size) in face_info:
            frame = cast(frame_address, py_object).value
            new_info.append((frame.image, size))
        five_points, confidences, flags = landmark_model.predict(new_info, boxes)
        data_que.put_batch_keypoints(face_info, 
                                     (five_points, confidences, flags))


def predict_video(video_path_or_cam, data_queue, detector, lmk_model):
    is_finish = False

    # put frame to data queue
    pool1= ThreadPoolExecutor(max_workers=1)
    vide_capture=cv2.VideoCapture(video_path_or_cam)
    frame_id = 0
    while True and not is_finish:
        ret, image = vide_capture.read()
        # if not ret or frame_id >= 10000:
        if not ret:
            break
        task = pool1.submit(
            lambda all_args: put_frame(*all_args), (data_queue, frame_id, image))
        frame_id += 1
    print("AAAA")

    # run detect
    pool2= ThreadPoolExecutor(max_workers=cfg['detector']['num_workers'])
    while True and not is_finish:
        task = pool2.submit(
            lambda all_args: detect_face(*all_args), (data_queue, detector))
    print("BBBB")

    # run landmark
    pool3= ThreadPoolExecutor(max_workers=cfg['landmark']['num_workers'])
    while True and not is_finish:
        task = pool3.submit(
            lambda all_args: detect_keypoint(*all_args), (data_queue, lmk_model))
    print("CCCC")

    # draw detection and recognition results
    start = time.time()
    while True: 
        frame = data_queue.get_result()
        if not frame:
            is_finish = True
            break
        dst = frame.image
        end = time.time()
        fps = 1 / (end - start)
        start = end
        cv2.putText(dst, f"fps: {fps:.0f}", (20, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), thickness=2)
        cv2.namedWindow("capture", 0)
        cv2.imshow("capture", dst)

        key = cv2.waitKey(1)
        if key == ord('q'):
            return

    pool1.shutdown()
    pool2.shutdown()
    pool3.shutdown()


def main():

    detector = Yolov5OnnxDetector(cfg['detector'])
    lmk_model = PeppaPigOnnxLandmark(cfg['landmark'])
    data_queue = DataQueue(max_length=cfg['data_queue']['max_length'],
                            det_batch_size=cfg['detector']['batch_size'],
                            lmk_batch_size=cfg['landmark']['batch_size'],
                            rec_batch_size=cfg['recognizer']['batch_size'])

    # src = cv2.imread('images/test1.jpg')
    # dst = predict_image(src, detector, lmk_model)
    # cv2.imwrite('images/test1_out.jpg', dst)

    print('inital finish')
    predict_video('videos/GodofGamblers.mp4', data_queue, detector, lmk_model)


if __name__ == "__main__":
    main()
