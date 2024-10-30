import sys
import numpy as np
import onnx
from onnx import shape_inference
from onnx.tools import update_model_dims
from onnxsim import simplify


def modify_model():
    """
    convert single batchsize ONNX convert to dynamic batchsize ONNX.
    """
    input_onnx = sys.argv[1]
    output_onnx = sys.argv[2]

    model = onnx.load(input_onnx)
    g = model.graph

    for node in g.node:
        if node.name == '/conv_6_flatten/Constant':
            for attr in node.attribute:
                if attr.name == 'value':
                    new_value = np.array([-1, 512], np.int64).tobytes()
                    attr.t.raw_data = new_value
                    break
            break

    model = update_model_dims.update_inputs_outputs_dims(
        model, 
        {"image": ["bs", 3, 112, 112]}, 
        {"feature": ["bs", 512]}
    )
    model = shape_inference.infer_shapes(model)
    model, _ = simplify(model)
    onnx.save(model, output_onnx)


if __name__ == "__main__":
    modify_model()
