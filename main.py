import time
import cv2
import numpy as np

from inferencer import Yolov5OnnxDetector, PeppaPigOnnxLandmark
from configs.end2end_config import cfg


def predict_image(src, detector, lmk_model):
    dst = src.copy()
    latency = {}
    
    t0 = time.time()
    det_boxes, det_scores = detector.predict(src)
    t1 = time.time()
    latency['det'] = (t1 - t0) * 1000

    for i, det_box in enumerate(det_boxes):
        l, t, r, b = det_box.astype(np.int64).tolist()
        cv2.rectangle(dst, (l, t), (r, b), (0, 255, 0))

        five_points, lmk_score = lmk_model.predict(src, det_box)
        radius = max(3, int(max(dst.shape[:2]) / 1000))
        for point in five_points:
            point = point.astype(np.int64).tolist()
            cv2.circle(dst, point, radius, (0, 0, 255), -1)
    latency['lmk'] = (time.time() - t1) * 1000
    
    return dst, latency, det_boxes.shape[0]


def predict_video(video_path_or_cam, detector, lmk_model):
    vide_capture=cv2.VideoCapture(video_path_or_cam)
    while 1:
        ret, image = vide_capture.read()
        if not ret:
            continue

        start = time.time()
        dst, latency, cnt = predict_image(image, detector, lmk_model)

        duration = time.time() - start
        fps = 1 / duration
        cv2.putText(dst, f"fps: {fps:.0f}", (20, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), thickness=2)
        cv2.putText(dst, f"det: {latency['det']:.0f}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), thickness=2)
        cv2.putText(dst, f"lmk: {latency['lmk']:.0f}", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), thickness=2)
        cv2.putText(dst, f"cnt: {cnt}", (20, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), thickness=2)

        cv2.namedWindow("capture", 0)
        cv2.imshow("capture", dst)

        key = cv2.waitKey(1)
        if key == ord('q'):
            return


def main():
    detector = Yolov5OnnxDetector(cfg['detector'])
    lmk_model = PeppaPigOnnxLandmark(cfg['landmark'])

    # src = cv2.imread('images/test1.jpg')
    # dst = predict_image(src, detector, lmk_model)
    # cv2.imwrite('images/test1_out.jpg', dst)

    predict_video('videos/GodofGamblers.mp4', detector, lmk_model)
    # predict_video('videos/Trump1.mp4', detector, lmk_model)
    # predict_video('videos/Trump2.mp4', detector, lmk_model)


if __name__ == "__main__":
    main()
