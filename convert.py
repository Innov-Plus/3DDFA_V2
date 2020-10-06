# Some standard imports
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

model_dict = tddfa.model.state_dict()
# Print model's state_dict
# print("Model's state_dict:")
# for param_tensor in model_dict:
#     print(param_tensor, "\t", model_dict[param_tensor])

import tensorflow as tf

print(tf.__version__)


# # compare ONNX Runtime and PyTorch results
# np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

# print("Exported model has been tested with ONNXRuntime, and the result looks good!")
