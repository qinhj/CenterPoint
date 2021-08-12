import argparse
import onnxruntime as rt
import numpy as np
import torch

# max index for ScatterND data shape
high = np.array([1, 512*512, 64], dtype='i8') - 1

def torch2onnx(model, save_path, dynamic=False):
    """
    @param: model   input torch model path
    @param: path    output onnx model path
    """
    model.eval()
    data = torch.rand(1, 3, 224, 224)
    input_names = ["input"]
    output_names = ["out"]
    dynamic_axes = {'input': [2, 3], 'out': [2, 3]} if dynamic else None
    torch.onnx._export(model, data, save_path, export_params=True, opset_version=11,
        input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes)
    print("torch2onnx finish.")


def model_forward_once(model, input_feed=None):
    """
    @param: model   onnx model path
    @param: input_feed
    @return:    list of <class 'numpy.ndarray'>
    """
    # create inference session
    sess = rt.InferenceSession(model)

    # check inputs and outputs
    inputs = sess.get_inputs()
    assert(len(inputs))
    outputs = sess.get_outputs()
    assert(len(outputs))

    output_names = [n.name for n in outputs]
    if not input_feed:
        input_feed = {}
        for n in inputs:
            print("input[%s] shape: %s" % (n.name, n.shape))
            shape = n.shape
            if 'tensor(int64)' == n.type:
                size = shape[:-1]
                size.append(0)  # as 1x??x0
                input_feed[n.name] = np.random.randint(0, size=size)
                size[-1] = 1    # as 1x??x1
                # create random tensor via numpy
                for idx in range(n.shape[-1]):
                    if 0 < high[idx]:
                        # Note: default dtype='i8', i.e. np.int64
                        arr = np.random.randint(0, high=high[idx], size=size)
                    else:
                        arr = np.zeros(size, dtype='i8')
                    input_feed[n.name] = np.append(input_feed[n.name], arr, axis=-1)
            else:
                # suppose float32, create random tensor via torch
                input_feed[n.name] = torch.rand(shape).cpu().numpy()

    # run session once
    outputs = sess.run(output_names, input_feed)
    return outputs, input_feed


def update_input(path):
    import onnx
    model = onnx.load(path)

    flag = False
    # check input tensor shape
    for i, n in enumerate(model.graph.input):
        if 30000 == n.type.tensor_type.shape.dim[2].dim_value:
            value_info = onnx.helper.make_tensor_value_info(n.name, n.type.tensor_type.elem_type, [1, 10, 5268, 20])
            model.graph.input.remove(n)
            model.graph.input.insert(i, value_info)
            flag = True
        elif 30000 == n.type.tensor_type.shape.dim[1].dim_value:
            value_info = onnx.helper.make_tensor_value_info(n.name, n.type.tensor_type.elem_type, [1, 5268, 2])
            model.graph.input.remove(n)
            model.graph.input.insert(i, value_info)
            flag = True
    if flag:
        # do infer onnx model's shapes
        model = onnx.shape_inference.infer_shapes(model)
        # Note: we can't check and simplify the origin official onnx model
        #import onnxsim
        #model, _ = onnxsim.simplify(model)
        print("%s is updated !!!" % path)
        onnx.save(model, path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model1', type=str, default='pfn.onnx', help='onnx model path')
    parser.add_argument('--model2', type=str, default='pfn.opt.onnx', help='onnx model path')
    args = parser.parse_args()
    #print(args)
    print("Verify model %s with %s" % (args.model1, args.model2))

    print("%s is running ..." % args.model1)
    update_input(args.model1)
    outputs1, input_feed = model_forward_once(args.model1, None)

    print("%s is running ..." % args.model2)
    update_input(args.model2)
    outputs2, _ = model_forward_once(args.model2, input_feed)

    assert(len(outputs1) == len(outputs2))
    for i, (s, d) in enumerate(zip(outputs1, outputs2)):
        print("output[%d] shape: %s %s" % (i, s.shape, (s == d).all()))

    print("Done")
