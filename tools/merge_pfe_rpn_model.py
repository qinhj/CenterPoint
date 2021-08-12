#!/usr/bin/env python
# coding: utf-8

import onnx
import onnxsim  # for: onnx.numpy_helper
import numpy as np
import copy


def make_scatterND(model, rpn_input_shape, indices_shape, pfn_out_maxpool_name, batch_size, save_for_trt=False):
    """
    @brief: 1. add new input tensor?; 2. add new ScatterND node; 3. ...
    @param: model
    @param: rpn_input_shape e.g. 1x64x512x512
    @param: indices_shape   e.g. 1x30000x2
    @param: pfn_out_maxpool_name
    @param: batch_size
    @param: save_for_trt    True for trt, default False
    """
    #### Step1: Create nodes before op "ScatterND" ####
    squeeze_axes = [3]
    squeeze_tensor = np.array(squeeze_axes, dtype=np.int32)
    # Create tensor "axes" which binds to Squeeze's attribute
    squeeze_tensor = onnx.numpy_helper.from_array(squeeze_tensor, name="axes")
    # Append tensor to model initializer
    model.graph.initializer.append(squeeze_tensor)

    # Remove single-dimensional entries from the shape of a tensor.
    squeeze_node = onnx.helper.make_node(op_type="Squeeze",
        inputs=[pfn_out_maxpool_name], outputs=['pfn_squeeze_1'],
        name="pfn_Squeeze_1", axes=[3], # e.g. 1x64x5268x1 -> 1x64x5268
    )
    # Transpose the input tensor similar to numpy.transpose.
    transpose_node_1 = onnx.helper.make_node(op_type="Transpose",
        inputs=['pfn_squeeze_1',], outputs=['pfn_transpose_1'],
        name="pfn_Transpose_1", perm=[0,2,1], # e.g. 1x64x5268 -> 1x5268x64
    )

    #### Step2: Create node for op "ScatterND" ####
    # e.g. 1x64x512x512 -> 1x262144x64
    data_shape = [batch_size, rpn_input_shape[2]*rpn_input_shape[3], rpn_input_shape[1]]
    data_tensor = np.zeros(data_shape, dtype=np.float32)
    # Create tensor "scatter_data" which binds to ScatterND's attribute
    data_tensor = onnx.numpy_helper.from_array(data_tensor, name="scatter_data")
    # Append tensor to model initializer
    model.graph.initializer.append(data_tensor)

    attr = {"output_shape":data_shape, "index_shape":indices_shape} if save_for_trt else {}
    # create node: ScatterND
    scatter_node = onnx.helper.make_node(op_type="ScatterND",
        inputs=['scatter_data', 'indices_input', 'pfn_transpose_1'],
        outputs=['scatter_1'], name="ScatterND_1", **attr)

    #### Step3: Create nodes after op "ScatterND" ####
    # Create tensor "pfn_reshape_shape" which binds to Reshape's input
    reshape_shape = np.array(rpn_input_shape, dtype=np.int64)
    reshape_tensor = onnx.numpy_helper.from_array(reshape_shape, name="pfn_reshape_shape")
    # Append tensor to model initializer
    model.graph.initializer.append(reshape_tensor)    

    # create nodes after ScatterND
    transpose_node_2 = onnx.helper.make_node(op_type="Transpose",
        inputs=['scatter_1',], outputs=['pfn_transpose_2'],
        name="pfn_Transpose_2", perm=[0,2,1],
    )
    reshape_node = onnx.helper.make_node(op_type="Reshape",
        inputs=["pfn_transpose_2", "pfn_reshape_shape"],
        outputs=['rpn_input'], name="pfn_reshape_1"
    )

    #### Step4: Update graph input && node ####
    idx_type = onnx.TensorProto.INT32 if save_for_trt else onnx.TensorProto.INT64
    input_node = onnx.helper.make_tensor_value_info('indices_input', idx_type, indices_shape)
    model.graph.input.append(input_node)
    # Append additional nodes
    model.graph.node.append(squeeze_node)
    model.graph.node.append(transpose_node_1)    
    model.graph.node.append(scatter_node)
    model.graph.node.append(transpose_node_2)    
    model.graph.node.append(reshape_node)


def main(pfn_model_path, rpn_model_path, save_model_path, trt=False):
    """
    @param: pfn_model_path  simplified pfn model path
    @param: rpn_model_path  simplified rpn model path
    @param: save_model_path path to save merged model
    @param: trt             trt output flag
    """
    # load pfn && rpn model
    pfn_model = onnx.load(pfn_model_path)
    rpn_model = onnx.load(rpn_model_path)

    # update node name of pfn model
    for node in pfn_model.graph.node:
        node.name = "pfn_" + node.name

    # Todo: model parameters
    batch_size = 1
    pfn_output_name = pfn_model.graph.output[0].name
    rpn_input_shape = [batch_size, 64, 512, 512]
    indices_shape = [batch_size, pfn_model.graph.input[0].type.tensor_type.shape.dim[2].dim_value, 2]

    # add additional nodes/initializers/...
    make_scatterND(pfn_model, rpn_input_shape, indices_shape, pfn_output_name, batch_size, trt)

    # update rpn model input
    for node in rpn_model.graph.node:
        if rpn_model.graph.input[0].name == node.input[0]:
            node.input[0] = pfn_model.graph.node[-1].output[0]
            break

    # merge rpn model nodes/initializers/...
    pfn_model.graph.node.extend(rpn_model.graph.node)
    pfn_model.graph.initializer.extend(rpn_model.graph.initializer)
    # replace pfn model output with rpn model output
    while 0 < len(pfn_model.graph.output):
        pfn_model.graph.output.pop()
    pfn_model.graph.output.extend(rpn_model.graph.output)

    # infer model shape then check and simplify model
    pfn_model = onnx.shape_inference.infer_shapes(pfn_model)
    if not trt:
        pfn_model, _ = onnxsim.simplify(pfn_model)

    onnx.save(pfn_model, save_model_path)
    print("Done")


if __name__ == "__main__":
    # input model path
    pfn_sim_model_path = "./onnx_model/pfn.opt.onnx"
    rpn_sim_model_path = "./onnx_model/rpn.onnx"

    # output model path
    pointpillars_save_path = "./onnx_model/pointpillars.onnx"
    pointpillars_trt_save_path = "./onnx_model/pointpillars_trt.onnx"

    main(pfn_sim_model_path, rpn_sim_model_path, pointpillars_save_path)
    main(pfn_sim_model_path, rpn_sim_model_path, pointpillars_trt_save_path, True)
