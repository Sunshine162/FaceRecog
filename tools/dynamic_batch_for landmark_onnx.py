import sys
import numpy as np
import onnx
from onnx import helper, shape_inference
from onnx.tools import update_model_dims
from onnxsim import simplify


def add_batch_size_node(g):
    shape_node = helper.make_node(
        'Shape', 
        [g.input[0].name], 
        ['input_data_shape_output'], 
        'input_data_shape',
    )
    g.node.insert(0, shape_node)
    
    slice_start = helper.make_node(
        'Constant', [], ['bs_slice_start_output'], 'bs_slice_start',
        value=onnx.numpy_helper.from_array(np.array([0], np.int64)),
    )
    g.node.insert(1, slice_start)
    slice_end = helper.make_node(
        'Constant', [], ['bs_slice_end_output'], 'bs_slice_end',
        value=onnx.numpy_helper.from_array(np.array([1], np.int64)),
    )
    g.node.insert(2, slice_end)
    slice_axis = helper.make_node(
        'Constant', [], ['bs_slice_axis_output'], 'bs_slice_axis',
        value=onnx.numpy_helper.from_array(np.array([0], np.int64)),
    )
    g.node.insert(3, slice_axis)
    slice_node = helper.make_node(
        'Slice',
        [shape_node.output[0], slice_start.output[0], slice_end.output[0], 
         slice_axis.output[0]], 
        ['input_batch_size'],
        'bs_slice',
    )
    g.node.insert(4, slice_node)

    return slice_node.output[0]


def modify_resize(g, resize_name, batch_size_output, resize_size):
    dst_shape_part2_node = helper.make_node(
        'Constant', 
        [], 
        ['resize_dst_shape_part2_output'], 
        'resize_dst_shape_part2',
        value=onnx.numpy_helper.from_array(
            np.array([64, resize_size, resize_size], np.int64)),
    )
    g.node.insert(5, dst_shape_part2_node)
    cat_node = helper.make_node(
        'Concat',
        [batch_size_output, dst_shape_part2_node.output[0]],
        ['resize_dst_shape_output'],
        f'resize_dst_shape',
        axis=0
    )
    g.node.insert(6, cat_node)

    for i, node in enumerate(g.node):
        if node.name == resize_name:
            node.input[3] = cat_node.output[0]
            break


def dynamic_reshape1(g, reshape_names, batch_size_output, last_dim):
    dst_shape_part2_node = helper.make_node(
        'Constant', 
        [], 
        ['reshape_dst_shape_part2_output'], 
        'reshape_dst_shape_part2',
        value=onnx.numpy_helper.from_array(np.array([98, last_dim], np.int64)),
    )
    g.node.insert(7, dst_shape_part2_node)
    cat_node = helper.make_node(
        'Concat',
        [batch_size_output, dst_shape_part2_node.output[0]],
        ['reshape_dst_shape_output'],
        'reshape_dst_shape',
        axis=0
    )
    g.node.insert(8, cat_node)

    for node in g.node:
        if node.name in reshape_names:
            node.input[1] = cat_node.output[0]


def replace_reshape(g, reshape_names):
    indices = []
    for i, node in enumerate(g.node):
        if node.name in reshape_names:
            indices.append(i)
    indices.sort(reverse=True)

    for i in indices:
        reshape_node = g.node[i]
        unsqueeze_node = helper.make_node(
            'Unsqueeze', 
            [reshape_node.input[0]],
            [reshape_node.output[0]],
            f'{reshape_node.name}_unsqueeze',
            axes=[-1]
        )
        g.node.pop(i)
        g.node.insert(i, unsqueeze_node)


def modify_model_tail(g, start_node_name, end_node_name, div_name, batch_size_output):

    start_node, end_node, div_node = None, None, None
    end_index = None
    for i, node in enumerate(g.node):
        if node.name == start_node_name:
            start_node = node
        elif node.name == end_node_name:
            end_node = node
            end_index = i
        elif node.name == div_name:
            div_node = node
    assert all([x is not None for x in [start_node, end_node, div_node]])
    div_node.input[0] = start_node.output[0]
    end_node.input[0] = div_node.output[0]

    dst_shape_part2_node = helper.make_node(
        'Constant', 
        [], 
        ['reshape1_dst_shape_part2_output'], 
        'reshape1_dst_shape_part2',
        value=onnx.numpy_helper.from_array(np.array([196], np.int64)),
    )
    g.node.insert(9, dst_shape_part2_node)
    cat_node = helper.make_node(
        'Concat',
        [batch_size_output, dst_shape_part2_node.output[0]],
        ['reshape1_dst_shape_output'],
        'reshape1_dst_shape',
        axis=0
    )
    g.node.insert(10, cat_node)
    end_node.input[1] = cat_node.output[0]


def modify_model(input_onnx, output_onnx, input_size, last_dim, resize_size, score_name):
    model = onnx.load(input_onnx)
    model, _ = simplify(model)

    model = update_model_dims.update_inputs_outputs_dims(
        model, 
        {"input": ["bs", 3, input_size, input_size]}, 
        {"output": ["bs", 196], score_name: ["bs", 98]}
    )
    model.graph.output[0].type.tensor_type.shape.dim[0].dim_param = "bs"
    model = shape_inference.infer_shapes(model)

    batch_size_output = add_batch_size_node(model.graph)
    modify_resize(model.graph, 
        '/student/decoder/aspp/fm_pool/Resize', batch_size_output, resize_size)
    dynamic_reshape1(model.graph, 
        ['/Reshape', '/Reshape_1', '/Reshape_2'], batch_size_output, last_dim)
    replace_reshape(model.graph, ['/Reshape_3', '/Reshape_5'])
    modify_model_tail(model.graph, '/Concat', '/Reshape_7', '/Div_1', batch_size_output)

    for info in model.graph.value_info:
        if info.name in ['/Reshape_output_0', '/Reshape_1_output_0', '/Reshape_2_output_0']:
            if info.type.tensor_type.shape.dim[-1].dim_param.startswith('unk'):
                info.type.tensor_type.shape.dim[-1].dim_value = last_dim
        if len(info.type.tensor_type.shape.dim) > 0:
            info.type.tensor_type.shape.dim[0].dim_param = 'bs'

    model = update_model_dims.update_inputs_outputs_dims(
        model, 
        {"input": ["bs", 3, input_size, input_size]}, 
        {"output": ["bs", 196], score_name: ["bs", 98]}
    )

    model = shape_inference.infer_shapes(model)
    model, _ = simplify(model)
    onnx.save(model, output_onnx)


def main():
    """
    convert single batchsize ONNX convert to dynamic batchsize ONNX.
    """
    input_onnx = sys.argv[1]
    output_onnx = sys.argv[2]
    if '256' in input_onnx:
        modify_model(input_onnx, output_onnx, 256, 4096, 16, 'score')
    else:
        modify_model(input_onnx, output_onnx, 128, 1024, 8, '2658')


if __name__ == "__main__":
    main()

