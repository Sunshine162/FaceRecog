import sys
import numpy as np
import onnxruntime as ort

onnx1 = sys.argv[1]  # single batch ONNX
onnx2 = sys.argv[2]  # dynamic batch ONNX
batch_size = int(sys.argv[3])

sess1 = ort.InferenceSession(onnx1)
sess2 = ort.InferenceSession(onnx2)

all_inputs = []
all_outputs = []
for i in range(batch_size):
	inp = np.random.randn(1, 3, 112, 112).astype(np.float32)
	outp = sess1.run(None, {'image': inp})[0]
	all_inputs.append(inp)
	all_outputs.append(outp)

all_inputs = np.concatenate(all_inputs, axis=0)
all_outputs1 = np.concatenate(all_outputs, axis=0)
all_outputs2 = sess2.run(None, {'image': all_inputs})

print("max-diff:", all_outputs1.shape, np.abs(all_outputs1 - all_outputs2).max())
