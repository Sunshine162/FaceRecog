import cv2
import numpy as np
from skimage import transform


def transformation_from_points(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2
    U, S, Vt = np.linalg.svd(points1.T * points2)
    R = (U * Vt).T
    return np.vstack([np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)), 
                     np.matrix([0., 0., 1.])])


def face_align(img, landmark, pattern, out_size, do_crop=True, input_box=None):
    """face alignment"""

    if do_crop:
        l, t, r, b = input_box  # left, top, right, bottom
        crop_img = img[t:b, l:r, :]
        landmark -= np.array([[l, t]], np.float32)

        if isinstance(landmark, (list, tuple)):
            landmark = np.array(landmark)
        if isinstance(pattern, (list, tuple)):
            pattern = np.array(pattern)
        
        landmark = landmark.astype(np.float32)
        pattern = pattern.astype(np.float32)

        tform = transform.SimilarityTransform()
        tform.estimate(landmark, pattern)
        M = tform.params[0:2, :]
        crop_img = cv2.warpAffine(crop_img, M, out_size, borderValue=0.0)

    else:
        pts1 = np.float64(
            np.matrix([[point[0], point[1]] for point in landmark]))
        pts2 = np.float64(
            np.matrix([[point[0], point[1]] for point in pattern]))
        M = transformation_from_points(pts1, pts2)
        aligned_image = cv2.warpAffine(img, M[:2], (img.shape[1], img.shape[0]))
        w, h = out_size
        crop_img = aligned_image[0:h, 0:w]

    return crop_img


def extract_face_embedding(face_image_path, face_detector, lmk_detector, 
                           face_recognizer):
    src = cv2.imread(face_image_path)

    src_h, src_w = src.shape[:2]
    in_w, in_h, = face_detector.input_size
    if src_h < 0.75 * in_h and src_w < 0.75 * in_w:
        new_h = in_h
        new_w = in_w
    else:
        new_h = int(src_h * 1.5)
        new_w = int(src_w * 1.5)

    src_pad = np.full((new_h, new_w, 3), 0).astype(np.uint8)
    top_start = (new_h - src_h) // 2
    left_start = (new_w - src_w) // 2
    src_pad[top_start:top_start+src_h, left_start:left_start+src_w, :] = src

    det_results = face_detector.predict([src_pad])[0]
    if len(det_results) == 3:
        det_boxes, det_confs, det_flags = det_results
        det_boxes = det_boxes[det_flags]
        five_points, *_ = lmk_detector.predict(src_pad, det_boxes)
    elif len(det_results) == 4:
        det_boxes, det_confs, five_points, det_flags = det_results

    if len(det_boxes) == 0:
        raise RuntimeError("No face detected in the given image!")
    if len(det_boxes) > 1:
        raise RuntimeError("Multiple faces detected in the given image!")

    feature = face_recognizer.extract_features(src_pad, det_boxes, five_points)
    return feature.squeeze()
