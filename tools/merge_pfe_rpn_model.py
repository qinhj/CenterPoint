#!/usr/bin/env python
# coding: utf-8

import onnx
import onnxsim  # for: onnx.numpy_helper
import numpy as np
import copy


def make_scatterND(model, rpn_input_shape, indices_shape, pfe_out_maxpool_name, batch_size, save_for_trt=False):
    """
    @brief: 1. add new input tensor?; 2. add new ScatterND node; 3. ...
    @param: model
    @param: rpn_input_shape e.g. 1x64x512x512
    @param: indices_shape   e.g. 1x30000x2
    @param: pfe_out_maxpool_name
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
        inputs=[pfe_out_maxpool_name], outputs=['pfe_squeeze_1'],
        name="pfe_Squeeze_1", axes=[3], # e.g. 1x64x5268x1 -> 1x64x5268
    )
    # Transpose the input tensor similar to numpy.transpose.
    transpose_node_1 = onnx.helper.make_node(op_type="Transpose",
        inputs=['pfe_squeeze_1',], outputs=['pfe_transpose_1'],
        name="pfe_Transpose_1", perm=[0,2,1], # e.g. 1x64x5268 -> 1x5268x64
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
        inputs=['scatter_data', 'indices_input', 'pfe_transpose_1'],
        outputs=['scatter_1'], name="ScatterND_1", **attr)

    #### Step3: Create nodes after op "ScatterND" ####
    # Create tensor "pfe_reshape_shape" which binds to Reshape's input
    reshape_shape = np.array(rpn_input_shape, dtype=np.int64)
    reshape_tensor = onnx.numpy_helper.from_array(reshape_shape, name="pfe_reshape_shape")
    # Append tensor to model initializer
    model.graph.initializer.append(reshape_tensor)    

    # create nodes after ScatterND
    transpose_node_2 = onnx.helper.make_node(op_type="Transpose",
        inputs=['scatter_1',], outputs=['pfe_transpose_2'],
        name="pfe_Transpose_2", perm=[0,2,1],
    )
    reshape_node = onnx.helper.make_node(op_type="Reshape",
        inputs=["pfe_transpose_2", "pfe_reshape_shape"],
        outputs=['rpn_input'], name="pfe_reshape_1"
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


if __name__ == "__main__":
    # input model path
    pfe_sim_model_path = "./onnx_model/pfe_sim.onnx"
    rpn_sim_model_path = "./onnx_model/rpn.onnx"
    # output model path
    pointpillars_save_path = "./onnx_model/pointpillars.onnx"
    pointpillars_trt_save_path = "./onnx_model/pointpillars_trt.onnx"

    # load pfe && rpn model
    pfe_model = onnx.load(pfe_sim_model_path)
    rpn_model = onnx.load(rpn_sim_model_path)

    batch_size = 1
    pfe_out_maxpool_name = "46"
    rpn_input_conv_name = "Conv_15"
    rpn_input_shape = [batch_size, 64, 512, 512]
    indices_shape = [batch_size, 30000, 2]

    # update node name of pfe model
    for node in pfe_model.graph.node:
        node.name = "pfe_"+node.name
    
    # merge nodes, outputs and initializers
    pfe_model.graph.node.extend(rpn_model.graph.node)
    pfe_model.graph.output.pop()
    pfe_model.graph.output.extend(rpn_model.graph.output)
    pfe_model.graph.initializer.extend(rpn_model.graph.initializer)
    
    pfe_model_trt = copy.deepcopy(pfe_model)

    # Connect pfe and rpn with scatterND
    make_scatterND(pfe_model, rpn_input_shape, indices_shape, pfe_out_maxpool_name, batch_size)
    make_scatterND(pfe_model_trt, rpn_input_shape, indices_shape, pfe_out_maxpool_name, batch_size, save_for_trt=True)
    
    def change_input(model):
        for node in model.graph.node:
            if node.name == rpn_input_conv_name:
                node.input[0] = "rpn_input"
                break
            # update input tensor shape
            model.graph.input[0].type.tensor_type.shape.dim[2].dim_value = indices_shape[1]
    
    change_input(pfe_model)
    change_input(pfe_model_trt)

    onnx.save(pfe_model, pointpillars_save_path)
    onnx.save(pfe_model_trt, pointpillars_trt_save_path)

    print("Done")
