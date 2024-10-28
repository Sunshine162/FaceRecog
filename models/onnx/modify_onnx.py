import sys
import numpy as np
import onnx
from onnx import helper, shape_inference
from onnx.tools import update_model_dims


def dynamic_reshape(g, reshape_name, start, end, last_dim):

    index = 0
    for node in g.node:
        if node.name == reshape_name:
            break
        index += 1
    reshape_node = g.node[index]

    shape_node = helper.make_node(
        'Shape', 
        [reshape_node.input[0]], 
        [f'{reshape_node.input[0]}_shape_output'], 
        f'{reshape_node.input[0]}_shape',
    )
    g.node.insert(index, shape_node)
    index += 1

    slice_start = helper.make_node(
        'Constant', [], [f'{reshape_node.name}_dst_shape_part1_start_output'], f'{reshape_node.name}_dst_shape_part1_start',
        value=onnx.numpy_helper.from_array(np.array([start], np.int64)),
    )
    g.node.insert(index, slice_start)
    index += 1
    slice_end = helper.make_node(
        'Constant', [], [f'{reshape_node.name}_dst_shape_part1_end_output'], f'{reshape_node.name}_dst_shape_part1_end',
        value=onnx.numpy_helper.from_array(np.array([end], np.int64)),
    )
    g.node.insert(index, slice_end)
    index += 1
    slice_axis = helper.make_node(
        'Constant', [], [f'{reshape_node.name}_dst_shape_part1_axis_output'], f'{reshape_node.name}_dst_shape_part1_axis',
        value=onnx.numpy_helper.from_array(np.array([0], np.int64)),
    )
    g.node.insert(index, slice_axis)
    index += 1
    slice_node = helper.make_node(
        'Slice',
        [f'{reshape_node.input[0]}_shape_output', f'{reshape_node.name}_dst_shape_part1_start_output', 
         f'{reshape_node.name}_dst_shape_part1_end_output', f'{reshape_node.name}_dst_shape_part1_axis_output'], 
        [f'{reshape_node.name}_dst_shape_part1_output'],
        f'{reshape_node.name}_dst_shape_part1',
    )
    g.node.insert(index, slice_node)
    index += 1

    last_dim_node = helper.make_node(
        'Constant', [], [f'{reshape_node.name}_last_dim_output'], f'{reshape_node.name}_last_dim',
        value=onnx.numpy_helper.from_array(np.array([last_dim], np.int64)),
    )
    g.node.insert(index, last_dim_node)
    index += 1
    cat_node = helper.make_node(
        'Concat',
        [f'{reshape_node.name}_dst_shape_part1_output', f'{reshape_node.name}_last_dim_output'],
        [f'{reshape_node.name}_dst_shape_output'],
        f'{reshape_node.name}_dst_shape',
        axis=0
    )
    g.node.insert(index, cat_node)
    index += 1

    # g.node.extend([shape_node, slice_start, slice_end, slice_axis, slice_node, last_dim_node, cat_node])
    reshape_node.input[1] = f'{reshape_node.name}_dst_shape_output'



def replace_reshape(g, reshape_name, scatternd_name, shape_out_name, del_names):

    reshape_index, scatternd_index = 0, 0
    flag = True
    scatternd_node = None
    for node in g.node:
        if node.name == reshape_name:
            flag = False
        if flag:
            reshape_index += 1
        if node.name == scatternd_name:
            break
        scatternd_index += 1
    reshape_node = g.node[reshape_index]
    scatternd_node = g.node[scatternd_index]

    unsqueeze_node = helper.make_node(
        'Unsqueeze', 
        [reshape_node.input[0]],
        [reshape_node.output[0]],
        f'{reshape_node.name}_unsqueeze',
        axes=[-1]
    )
    g.node.insert(reshape_index, unsqueeze_node)
    scatternd_index += 1
    



    slice_start = helper.make_node(
        'Constant', [], [f'{reshape_node.name}_dst_shape_part1_start_output'], f'{reshape_node.name}_dst_shape_part1_start',
        value=onnx.numpy_helper.from_array(np.array([0], np.int64)),
    )
    g.node.insert(scatternd_index, slice_start)
    scatternd_index += 1
    slice_end = helper.make_node(
        'Constant', [], [f'{reshape_node.name}_dst_shape_part1_end_output'], f'{reshape_node.name}_dst_shape_part1_end',
        value=onnx.numpy_helper.from_array(np.array([1], np.int64)),
    )
    g.node.insert(scatternd_index, slice_end)
    scatternd_index += 1
    slice_axis = helper.make_node(
        'Constant', [], [f'{reshape_node.name}_dst_shape_part1_axis_output'], f'{reshape_node.name}_dst_shape_part1_axis',
        value=onnx.numpy_helper.from_array(np.array([0], np.int64)),
    )
    g.node.insert(scatternd_index, slice_axis)
    scatternd_index += 1
    slice_node = helper.make_node(
        'Slice',
        [shape_out_name, f'{reshape_node.name}_dst_shape_part1_start_output', 
         f'{reshape_node.name}_dst_shape_part1_end_output', f'{reshape_node.name}_dst_shape_part1_axis_output'], 
        [f'{reshape_node.name}_dst_shape_part1_output'],
        f'{reshape_node.name}_dst_shape_part1',
    )
    g.node.insert(scatternd_index, slice_node)
    scatternd_index += 1

    last_dim_node = helper.make_node(
        'Constant', [], [f'{reshape_node.name}_last_dim_output'], f'{reshape_node.name}_last_dim',
        value=onnx.numpy_helper.from_array(np.array([98, 1, 3], np.int64)),
    )
    g.node.insert(scatternd_index, last_dim_node)
    scatternd_index += 1
    cat_node = helper.make_node(
        'Concat',
        [f'{reshape_node.name}_dst_shape_part1_output', f'{reshape_node.name}_last_dim_output'],
        [f'{reshape_node.name}_dst_shape_output'],
        f'{reshape_node.name}_dst_shape',
        axis=0
    )
    g.node.insert(scatternd_index, cat_node)
    scatternd_index += 1

    expand_node = helper.make_node(
        'Expand',
        [scatternd_node.input[1], f'{reshape_node.name}_dst_shape_output'],
        [f'{reshape_name}_expand_output'],
        f'{reshape_name}_expand',
    )
    g.node.insert(scatternd_index, expand_node)
    scatternd_index += 1
    scatternd_node.input[1] = f'{reshape_name}_expand_output'


    del_ids = []
    del_names = del_names + [reshape_name]
    for i, node in enumerate(g.node):
        if node.name in del_names:
            del_ids.append(i)
    del_ids.sort(reverse=True)
    for i in del_ids:
        g.node.pop(i)


def modify_256x256():
    input_onnx = sys.argv[1]
    output_onnx = sys.argv[2]

    model = onnx.load(input_onnx)
    model = update_model_dims.update_inputs_outputs_dims(
        model, 
        {"input": ["bs", 3, 256, 256]}, 
        {"output": ["bs", 196], "score": ["bs", 98]}
    )
    model.graph.output[0].type.tensor_type.shape.dim[0].dim_param = "bs"
    model = shape_inference.infer_shapes(model)


    dynamic_reshape(model.graph, '/Reshape', 0, 2, 4096)
    dynamic_reshape(model.graph, '/Reshape_1', 0, 2, 4096)
    dynamic_reshape(model.graph, '/Reshape_2', 0, 2, 4096)
    dynamic_reshape(model.graph, '/Reshape_7', 0, 1, 196)
    replace_reshape(model.graph, '/Reshape_3', '/ScatterND', '/Shape_output_0', ['/Shape_1', '/Slice_3', '/Concat_2'])
    replace_reshape(model.graph, '/Reshape_4', '/ScatterND_1', '/Shape_2_output_0', ['/Shape_3', '/Slice_4', '/Concat_4'])
    replace_reshape(model.graph, '/Reshape_5', '/ScatterND_2', '/Shape_4_output_0', ['/Shape_5', '/Slice_5', '/Concat_6'])
    replace_reshape(model.graph, '/Reshape_6', '/ScatterND_3', '/Shape_6_output_0', ['/Shape_7', '/Slice_6', '/Concat_8'])

    for info in model.graph.value_info:
        if info.name == '/student/decoder/aspp/fm_pool/Resize_output_0':
            for i, dim in enumerate(['bs', 64, 16, 16]):
                if isinstance(dim, str):
                    info.type.tensor_type.shape.dim[i].dim_param = dim
                elif isinstance(dim, int):
                    info.type.tensor_type.shape.dim[i].dim_value = dim
                else:
                    raise TypeError()
        if info.name in ['/Reshape_output_0', '/Reshape_1_output_0', '/Reshape_2_output_0']:
            if info.type.tensor_type.shape.dim[-1].dim_param.startswith('unk'):
                info.type.tensor_type.shape.dim[-1].dim_value = 4096
        if len(info.type.tensor_type.shape.dim) > 0:
            info.type.tensor_type.shape.dim[0].dim_param = 'bs'


    model = update_model_dims.update_inputs_outputs_dims(
        model, 
        {"input": ["bs", 3, 256, 256]}, 
        {"output": ["bs", 196], "score": ["bs", 98]}
    )
    model = shape_inference.infer_shapes(model)
    onnx.save(model, output_onnx)



def modify_128x128(input_onnx, output_onnx):

    model = onnx.load(input_onnx)
    model = update_model_dims.update_inputs_outputs_dims(
        model, 
        {"input": ["bs", 3, 128, 128]}, 
        {"output": ["bs", 196], "score": ["bs", 98]}
    )
    model.graph.output[0].type.tensor_type.shape.dim[0].dim_param = "bs"
    model = shape_inference.infer_shapes(model)


    dynamic_reshape(model.graph, '/Reshape', 0, 2, 1024)
    dynamic_reshape(model.graph, '/Reshape_1', 0, 2, 1024)
    dynamic_reshape(model.graph, '/Reshape_2', 0, 2, 1024)
    dynamic_reshape(model.graph, '/Reshape_7', 0, 1, 196)
    replace_reshape(model.graph, '/Reshape_3', '/ScatterND', '/Shape_output_0', ['/Shape_1', '/Slice_3', '/Concat_2'])
    replace_reshape(model.graph, '/Reshape_4', '/ScatterND_1', '/Shape_2_output_0', ['/Shape_3', '/Slice_4', '/Concat_4'])
    replace_reshape(model.graph, '/Reshape_5', '/ScatterND_2', '/Shape_4_output_0', ['/Shape_5', '/Slice_5', '/Concat_6'])
    replace_reshape(model.graph, '/Reshape_6', '/ScatterND_3', '/Shape_6_output_0', ['/Shape_7', '/Slice_6', '/Concat_8'])

    for info in model.graph.value_info:
        if info.name == '/student/decoder/aspp/fm_pool/Resize_output_0':
            for i, dim in enumerate(['bs', 64, 8, 8]):
                if isinstance(dim, str):
                    info.type.tensor_type.shape.dim[i].dim_param = dim
                elif isinstance(dim, int):
                    info.type.tensor_type.shape.dim[i].dim_value = dim
                else:
                    raise TypeError()
        if info.name in ['/Reshape_output_0', '/Reshape_1_output_0', '/Reshape_2_output_0']:
            if info.type.tensor_type.shape.dim[-1].dim_param.startswith('unk'):
                info.type.tensor_type.shape.dim[-1].dim_value = 1024
        if len(info.type.tensor_type.shape.dim) > 0:
            info.type.tensor_type.shape.dim[0].dim_param = 'bs'


    model = update_model_dims.update_inputs_outputs_dims(
        model, 
        {"input": ["bs", 3, 128, 128]}, 
        {"output": ["bs", 196], "score": ["bs", 98]}
    )
    model = shape_inference.infer_shapes(model)
    onnx.save(model, output_onnx)



def main():
    input_onnx = sys.argv[1]
    output_onnx = sys.argv[2]
    if '256' in input_onnx:
        modify_256x256(input_onnx, output_onnx)
    else:
        modify_128x128(input_onnx, output_onnx)



if __name__ == "__main__":
    main()

