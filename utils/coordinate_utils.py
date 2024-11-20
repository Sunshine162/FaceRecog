import numpy as np

def xywh2xyxy(boxes):
    """
    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where 
    xy1=top-left, xy2=bottom-right
    
    Reference: 
        https://github.com/610265158/Peppa_Pig_Face_Landmark

    Args:
        boxes: format is [[x_center, y_center, width, height], ...]
    
    Return:
        new_boxes: format is [[x_left, y_top, x_right, y_bottom]]
    """

    new_boxes =  np.copy(boxes)
    new_boxes[..., 0] = boxes[..., 0] - boxes[..., 2] / 2  # top left x
    new_boxes[..., 1] = boxes[..., 1] - boxes[..., 3] / 2  # top left y
    new_boxes[..., 2] = boxes[..., 0] + boxes[..., 2] / 2  # bottom right x
    new_boxes[..., 3] = boxes[..., 1] + boxes[..., 3] / 2  # bottom right y
    return new_boxes

def revert_dets(dets, left_pad, top_pad, scale_factor):
    """revert coordinates of box to original image

    Args:
        dets: 6D-array result for detection or 16D-array for detection and landmark
        meta: a dict like {'scale_factor': 0.5, 'left_pad': 0, 'top_pad': 200}
    
    Return:
        new_box: format is [x_left, y_top, x_right, y_bottom]
    """

    dets[..., 0:4] -= np.array([[left_pad, top_pad, left_pad, top_pad]])
    dets[..., 0:4] /= scale_factor
    
    if dets.shape[-1] == 16:
        dets[..., 5:15] -= np.array([[left_pad, top_pad] * 5])
        dets[..., 5:15] /= scale_factor

    return dets

def revert_points(points, box):
    """convert coordinates of points
    do two things:
        1. relative coordinates in box  ==>  absolute coordinates in box
        2. absolute coordinates in box  ==>  absolute coordinates in full image
    
    Args:
        points: N x 2 numpy.ndarray, like [[x1, y1], [x2, y2], ...]
        box: format is [x_left, y_top, x_right, y_bottom]
    
    Return:
        points:  N x 2 numpy.ndarray, like [[x1, y1], [x2, y2], ...]
    """
    assert points.shape[-1] == 2

    l, t, r, b = box  # left, top, right, bottom
    w = r - l
    h = b - t
    points[..., 0] = points[..., 0] * w + l
    points[..., 1] = points[..., 1] * h + t

    return points


def extend_square_box(box, ratio, img_size, side='max'):
    """extend box to a square
    
    Args:
        box: tuple or list, format is [x_left, y_top, x_right, y_bottom]
        ratio: ratio for extending
        img_size: tuple or list, image width and image height
        side: select a square's side
    
    Return:
        new_box: result, which format is same to input box
    """

    assert ratio >= 1
    box_width = box[2] - box[0]
    box_height = box[3] - box[1]
    center_x = (box[2] + box[0]) / 2
    center_y = (box[3] + box[1]) / 2
    img_width, img_height = img_size

    if side == 'width':
        side = box_width * ratio
    elif side == 'height':
        side = box_height * ratio
    elif side == 'max':
        side = max(box_width, box_height) * ratio
    elif side == 'min':
        side = min(box_width, box_height) * ratio
    else:
        raise ValueError(
            f"side must be one of ['width', 'height', 'max', 'min']")
    
    new_box = [None, None, None, None]
    new_box[0] = max(0, int(center_x - side / 2))
    new_box[1] = max(0, int(center_y - side / 2))
    new_box[2] = min(img_width, int(center_x + side / 2))
    new_box[3] = min(img_height, int(center_y + side / 2))

    return new_box

def extend_box_by_ratio(box, ratio, img_size):
    """extend box
    Args:
        box: tuple or list, format is [x_left, y_top, x_right, y_bottom]
        ratio: ratio for extending
        img_size: tuple or list, image width and image height
    
    Return:
        new_box: result, which format is same to input box
    """

    assert ratio >= 1
    box_width = box[2] - box[0]
    box_height = box[3] - box[1]
    center_x = (box[2] + box[0]) / 2
    center_y = (box[3] + box[1]) / 2
    img_width, img_height = img_size

    new_width = box_width * ratio
    new_height= box_height * ratio

    new_box = [None, None, None, None]
    new_box[0] = max(0, int(center_x - new_width / 2))
    new_box[1] = max(0, int(center_y - new_height / 2))
    new_box[2] = min(img_width, int(center_x + new_width / 2))
    new_box[3] = min(img_height, int(center_y + new_height / 2))

    return new_box

def extend_box(box, cfg, img_size):
    """extend box"""
    if cfg.get('square', False):
        return extend_square_box(box, cfg['ratio'], img_size, cfg['side'])
    else:
        return extend_box_by_ratio(box, cfg['ratio'], img_size)

