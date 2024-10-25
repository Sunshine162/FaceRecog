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
[X] 1. implement single ONNX end2end demo of face detect->landmark->recognize.
[X] 2. inference with multiple threads.
[ ] 3. implement frame skipping.
[ ] 4. implement multiple batch size for landmark inference and recognizer inference.
[ ] 5. ONNX model optimization.
[ ] 6. implemet face tracing. 
