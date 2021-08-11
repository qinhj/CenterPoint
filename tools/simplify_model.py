#!/usr/bin/env python
# coding: utf-8

import onnx
from onnx import numpy_helper, helper
import numpy as np
from copy import deepcopy


## map: name -> index
def map_name_to_index(graph_prop_list):
    m = {}
    for i, n in enumerate(graph_prop_list):
        m[n.name] = i
    return m


## transpose input data from nhwc to nchw
def convert_input_nhwc_nchw(model):
    batch_dim = 1

    # input tensor dim
    dim_list = [dim_val.dim_value for dim_val in model.graph.input[0].type.tensor_type.shape.dim]
    dim_list.insert(0, batch_dim) # as nhwc
    dim_list = np.array(dim_list)[[0,3,1,2]] # -> nchw

    # reset input tensor value info
    input_node = onnx.helper.make_tensor_value_info(
        'input.1', onnx.TensorProto.FLOAT, dim_list.tolist())
    model.graph.input.pop()
    model.graph.input.append(input_node)

    # output tensor dim
    dim_list = [dim_val.dim_value for dim_val in model.graph.output[0].type.tensor_type.shape.dim]
    dim_list.insert(0, batch_dim) # as nhwc
    dim_list.insert(2, 1)
    dim_list = np.array(dim_list)[[0,3,1,2]] # -> nchw

    # reset output tensor value info
    out_node = onnx.helper.make_tensor_value_info(
        model.graph.output[0].name, onnx.TensorProto.FLOAT, dim_list.tolist())
    model.graph.output.pop()
    model.graph.output.append(out_node)


def simplify_model(model_path):
    model = onnx.load(model_path)
    if model is None:
        print("File %s not found!" % model_path)
    from onnxsim import simplify
    return simplify(model)


def simplify_pfn_rpn_model(pfn_model_path, rpn_model_path):
    """
    @param: [in/out]pfn_model_path
    @param: [in/out]rpn_model_path
    """
    model, check = simplify_model(pfn_model_path)
    if not check:
        print("[ERROR]: Simplify %s error!" % pfn_model_path)
    onnx.save(model, pfn_model_path)

    model, check = simplify_model(rpn_model_path)
    if not check:
        print("[ERROR]: Simplify %s error!" % rpn_model_path)    
    onnx.save(model, rpn_model_path)


if __name__ == "__main__":

    pfn_model_path = "./onnx_model/pfn.onnx"
    rpn_model_path = "./onnx_model/rpn.onnx"

    # simplify pfn && rpn model inplace
    simplify_pfn_rpn_model(pfn_model_path, rpn_model_path)
    
    ## ---------------- optimize pfn model ---------------- ##
    
    # load pfn model
    model = onnx.load(pfn_model_path)

    # delete nodes
    delete_dict = {}
    for node in model.graph.node:
        if node.op_type in {"Transpose", "Expand", "Squeeze"}:
            delete_dict[node.output[0]] = node

    # pop all model value info
    val_len = len(model.graph.value_info)
    for idx in range(val_len):
        model.graph.value_info.pop()

    convert_input_nhwc_nchw(model)

    # map: name -> index
    map_init = map_name_to_index(model.graph.initializer)

    # modify some op nodes
    for i, node in enumerate(model.graph.node):
        # update MatMul as Conv2D
        if node.op_type == "MatMul":
            node.op_type = "Conv"
            # tensor -> array -> expand_dims -> tensor
            weight_name = node.input[1]
            weight_index = map_init[weight_name]
            weight_tensor = model.graph.initializer[weight_index]
            model.graph.initializer.remove(weight_tensor)
            weight_array = numpy_helper.to_array(weight_tensor).transpose(1,0)
            weight_array = np.expand_dims(weight_array, axis=-1)
            weight_array = np.expand_dims(weight_array, axis=-1)
            weight_tensor = numpy_helper.from_array(weight_array, name=weight_name)
            model.graph.initializer.insert(weight_index, weight_tensor)
        
        # skip delete nodes by resetting the input tensor
        if node.input[0] in delete_dict.keys():
            node.input[0] = delete_dict[node.input[0]].input[0]
        
        # replace ReduceMax by MaxPool
        if node.op_type == "ReduceMax":
            model.graph.node.remove(node)
            node = helper.make_node(op_type="MaxPool",
                inputs=node.input, outputs=node.output, name=node.name,
                ceil_mode = 0, kernel_shape = [1,20], pads = [0,0,0,0], strides=[1,1])
            model.graph.node.insert(i, node)
        
        # update input tensor("repeats") of Tile
        if node.op_type == "Tile":
            arr_name = node.input[1]
            arr_index = map_init[arr_name]
            arr_tensor = model.graph.initializer[arr_index]
            model.graph.initializer.remove(arr_tensor)
            arr_array = np.array([1,1,1,20], np.int64)
            arr_tensor = numpy_helper.from_array(arr_array, name=arr_name)
            model.graph.initializer.insert(arr_index, arr_tensor)
        
        # update "axis" to 1 of Concat
        if node.op_type == "Concat":
            node.attribute[0].i = 1

    # update/check output tensor value info
    for node in model.graph.output:
        if node.name in delete_dict.keys():
            node.name = delete_dict[node.name].input[0]

    # remove target nodes
    for keys, node in delete_dict.items():
        model.graph.node.remove(node)

    # do infer onnx model's shapes
    model = onnx.shape_inference.infer_shapes(model)
    # check and simplify onnx model
    import onnxsim
    model, _ = onnxsim.simplify(model)

    pfn_model_save_path = "./onnx_model/pfn.opt.onnx"
    onnx.save(model, pfn_model_save_path)
    print("Done")
