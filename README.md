# FaceRecog
Face recognition on some device

## Install
```bash
conda create -n FaceRecog python=3.10
conda activate FaceRecog
pip install -r requirements.txt
```

## Example
```bash
python main.py -c configs/onnx_end2end_config.yml -i videos/Trump3.mp4
```

## Feature
- [X] implement single ONNX end2end demo of face detect->landmark->recognize.
- [X] inference with multiple threads.
- [X] implement frame skipping.
- [X] implement multiple batch size for landmark inference and recognizer inference.
- [ ] ONNX model optimization.
- [X] implement face tracing.
- [X] implement openvino.
- [X] face detection with landmark detection
- [X] face mosaic
- [ ] multiple process test
- [ ] profile count
- [ ] asyncio


## Reference

- [https://github.com/610265158/Peppa_Pig_Face_Landmark](https://github.com/610265158/Peppa_Pig_Face_Landmark)
- [https://github.com/xuexingyu24/MobileFaceNet_Tutorial_Pytorch](https://github.com/xuexingyu24/MobileFaceNet_Tutorial_Pytorch)
- [https://github.com/ifzhang/ByteTrack/tree/main/yolox/tracker](https://github.com/ifzhang/ByteTrack/tree/main/yolox/tracker)
