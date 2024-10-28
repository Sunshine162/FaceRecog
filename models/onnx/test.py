import sys
import numpy as np
import onnxruntime as ort

onnx1 = sys.argv[1]
onnx2 = sys.argv[2]
input_size = int(sys.argv[3])
batch_size = int(sys.argv[4])

sess1 = ort.InferenceSession(onnx1)
sess2 = ort.InferenceSession(onnx2)

all_inputs = []
all_outputs = []
for i in range(batch_size):
	inp = np.random.randn(1, 3, input_size, input_size).astype(np.float32)
	outp = sess1.run(None, {'input': inp})
	all_inputs.append(inp)
	all_outputs.append(outp)

all_inputs = np.concatenate(all_inputs, axis=0)
all_outputs1 = []
all_outputs1.append(np.concatenate([x[0] for x in all_outputs], axis=0))
all_outputs1.append(np.concatenate([x[1] for x in all_outputs], axis=0))

all_outputs2 = sess2.run(None, {'input': all_inputs})

print(all_outputs1[0].shape, np.abs(all_outputs1[0] - all_outputs2[0]).max())
print(all_outputs1[1].shape, np.abs(all_outputs1[1] - all_outputs2[1]).max())
