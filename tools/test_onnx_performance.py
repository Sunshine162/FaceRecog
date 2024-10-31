import sys
import time

import numpy as np
import onnxruntime as ort


DATA_TYPE_MAPPING = {
    "tensor(float)": np.float32,
}


def test_preformance(onnx_path, batch_size, warmup_times=5, repeat_times=100):
    sess = ort.InferenceSession(onnx_path)
    inputs = sess.get_inputs()
    assert len(inputs) == 1
    input_name = inputs[0].name
    input_type = DATA_TYPE_MAPPING[inputs[0].type]
    input_shape = inputs[0].shape
    assert isinstance(input_shape[0], str)

    for i in range(warmup_times):
        # breakpoint()
        inp = np.random.randn(batch_size, *input_shape[1:]).astype(input_type)
        outp = sess.run(None, {input_name: inp})

    times = []
    for i in range(repeat_times):
        t0 = time.time()
        inp = np.random.randn(batch_size, *input_shape[1:]).astype(input_type)
        outp = sess.run(None, {input_name: inp})
        times.append(time.time() - t0)
    
    fps = batch_size / (np.array(times).mean().item())

    return fps
    

def main():
    onnx_path = sys.argv[1]
    repeat_times = int(sys.argv[2])
    output_csv = sys.argv[3]
    wait_time = 5

    best_bs, best_fps = 0, 0.0
    with open(output_csv, 'w', encoding='utf-8') as f:
        f.write("batch_size,fps\n")
        for bs in range(1, 65):
            fps = test_preformance(onnx_path, bs, repeat_times=repeat_times)
            f.write(f"{bs},{fps}\n")
            print(f"PERFORMANCE  ===>  bs={bs}, fps={fps:.2f}\n")

            if fps > best_fps:
                best_bs = bs
                best_fps = fps
            
            time.sleep(5)
            
        print(f"PERFORMANCE  ===>  best_bs={best_bs}, best_fps={best_fps:.2f}\n")


if __name__ == "__main__":
    main()
