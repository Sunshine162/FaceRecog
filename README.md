# FaceRecog
Face recognition on some device

## Install
```
conda create -n FaceRecog python=3.10
conda activate FaceRecog
pip install -r requirements.txt
```

## Example
```
python main.py videos/Trump3.mp4
```

## Feature


- [X] implement single ONNX end2end demo of face detect->landmark->recognize.
- [X] inference with multiple threads.
- [X] implement frame skipping.
- [ ] implement multiple batch size for landmark inference and recognizer inference.
- [ ] ONNX model optimization.
- [ ] implemet face tracing.
- [ ] implemet openvino.


## Reference

- [https://github.com/610265158/Peppa_Pig_Face_Landmark](https://github.com/610265158/Peppa_Pig_Face_Landmark)
- [https://github.com/xuexingyu24/MobileFaceNet_Tutorial_Pytorch](https://github.com/xuexingyu24/MobileFaceNet_Tutorial_Pytorch)
