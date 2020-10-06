# Some standard imports
from onnx import optimizer
import onnx
import io
import numpy as np

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx

import yaml
from TDDFA import TDDFA

# Load pretrained model weights
batch_size = 1    # just a random number

cfg = yaml.load(open('configs/mb05_120x120.yml'), Loader=yaml.SafeLoader)

tddfa = TDDFA(gpu_mode=False, **cfg)

x = torch.randn(batch_size, 3, 120, 120, requires_grad=True)
torch_out = tddfa.model(x)

# traced_model = torch.jit.trace(tddfa.model, x)
# torch.jit.save(traced_model, "mb05_120x120.pt")

# model_dict = traced_model.state_dict()

# print(model_dict.keys())

# # Print model's state_dict
# print("Model's state_dict:")
# for param_tensor in model_dict:
#     print(param_tensor, "\t", model_dict[param_tensor].size())

pt_model = torch.jit.load("mb05_120x120.pt")
pt_model.eval()

pt_out = pt_model(x)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

np.testing.assert_allclose(to_numpy(torch_out), to_numpy(pt_out), rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")


# # Export the model
# torch.onnx.export(tddfa.model,               # model being run
#                   # model input (or a tuple for multiple inputs)
#                   x,
#                   # where to save the model (can be a file or file-like object)
#                   "mb05_120x120.onnx",
#                   export_params=True,        # store the trained parameter weights inside the model file
#                   opset_version=10,          # the ONNX version to export the model to
#                   do_constant_folding=True,  # whether to execute constant folding for optimization
#                   input_names=['input'],   # the model's input names
#                   output_names=['output'])  # the model's output names


# onnx_model = onnx.load("mb05_120x120.onnx")
# passes = ['nop', 'eliminate_identity', 'eliminate_nop_transpose', 'eliminate_nop_pad', 'eliminate_unused_initializer',
#           'fuse_consecutive_squeezes', 'fuse_consecutive_transposes', 'fuse_add_bias_into_conv', 'fuse_transpose_into_gemm']
# optimized_model = optimizer.optimize(onnx_model, passes, fixed_point=True)

# onnx.save(optimized_model, "mb05_120x120_opt.onnx")


# import onnx
# from onnx_tf.backend import prepare

# # Load ONNX model and convert to TensorFlow format
# model_onnx = onnx.load('mb05_120x120.onnx')

# tf_rep = prepare(model_onnx)

# # Export model as .pb file
# tf_rep.export_graph('mb05_120x120.pb')

# import onnxruntime

# ort_session = onnxruntime.InferenceSession("mb05_120x120_opt.onnx")

# def to_numpy(tensor):
#     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# # compute ONNX Runtime output prediction
# ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
# ort_outs = ort_session.run(None, ort_inputs)

# # compare ONNX Runtime and PyTorch results
# np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

# print("Exported model has been tested with ONNXRuntime, and the result looks good!")
