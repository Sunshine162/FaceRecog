import time
import cv2
import numpy as np

from configs.end2end_config import cfg
from ctypes import cast, py_object
from multiprocessing import Process
from inferencer import Yolov5OnnxDetector, PeppaPigOnnxLandmark
from threading import Thread
from utils import DataQueue


IS_FINISH = False


def put_frame(video_path_or_cam, data_queue):
    vide_capture = cv2.VideoCapture(video_path_or_cam)
    frame_id = 0
    while True:
        ret, image = vide_capture.read()
        if not ret:
            break
        data_queue.put_frame(frame_id, image)
        frame_id += 1


def detect_face(data_queue, detector):
    global IS_FINISH
    while not IS_FINISH:
        frame_info, images = data_queue.get_batch_images()
        if not images:
            continue
        det_results = detector.predict(images)
        data_queue.put_batch_detect_results(frame_info, det_results)


def detect_keypoint(data_queue, lmk_model):
    global IS_FINISH
    while not IS_FINISH:
        face_info, boxes = data_queue.get_batch_boxes()
        if not boxes:
            continue

        new_info = []
        for (frame_address, start, size) in face_info:
            frame = cast(frame_address, py_object).value
            new_info.append((frame.image, size))

        five_points, confidences, flags = lmk_model.predict(new_info, boxes)
        data_queue.put_batch_keypoints(
            face_info, (five_points, confidences, flags))


def predict_video(video_path_or_cam, data_queue):
    global IS_FINISH

    # put frame to data queue
    put_thread = Thread(target=put_frame, args=(video_path_or_cam, data_queue))
    put_thread.start()
    while data_queue.frame_queue.empty():
        time.sleep(0.1)

    # run detect
    det_threads = []
    for i in range(cfg['detector']['num_workers']):
        detector = Yolov5OnnxDetector(cfg['detector'])
        det_thread = Thread(target=detect_face, args=(data_queue, detector))
        det_thread.start()
        det_threads.append(det_thread)

    # run landmark
    lmk_threads = []
    for i in range(cfg['landmark']['num_workers']):
        lmk_model = PeppaPigOnnxLandmark(cfg['landmark'])
        lmk_thread = Thread(target=detect_keypoint, args=(data_queue, lmk_model))
        lmk_thread.start()
        lmk_threads.append(lmk_thread)

    # draw detection and recognition results
    start = time.time()
    while True: 
        frame = data_queue.get_result()
        if not frame:
            IS_FINISH = True
            break

        dst = frame.image
        if frame.faces:
            for face in frame.faces:
                l, t, r, b = face.det_box.astype(np.int64).tolist()
                cv2.rectangle(dst, (l, t), (r, b), (0, 255, 0))
        
                radius = max(3, int(max(dst.shape[:2]) / 1000))
                if face.keypoints is None:
                    continue
                for point in face.keypoints:
                    point = point.astype(np.int64).tolist()
                    cv2.circle(dst, point, radius, (0, 0, 255), -1)

        end = time.time()
        fps = 1 / (end - start)
        start = end
        print(f'done={frame.frame_id}  fps={fps}')
        cv2.putText(dst, f"fps: {fps:.0f}", (20, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), thickness=2)
        cv2.putText(dst, f"id: {frame.frame_id}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), thickness=2)

        cv2.namedWindow("capture", 0)
        cv2.imshow("capture", dst)

        key = cv2.waitKey(1)
        if key == ord('q'):
            return


def main():

    # detector = Yolov5OnnxDetector(cfg['detector'])
    # lmk_model = PeppaPigOnnxLandmark(cfg['landmark'])
    data_queue = DataQueue(max_length=cfg['data_queue']['max_length'],
                            det_batch_size=cfg['detector']['batch_size'],
                            lmk_batch_size=cfg['landmark']['batch_size'],
                            rec_batch_size=cfg['recognizer']['batch_size'])

    # src = cv2.imread('images/test1.jpg')
    # dst = predict_image(src, detector, lmk_model)
    # cv2.imwrite('images/test1_out.jpg', dst)

    print('inital finish')
    predict_video('videos/GodofGamblers.mp4', data_queue)


if __name__ == "__main__":
    main()
