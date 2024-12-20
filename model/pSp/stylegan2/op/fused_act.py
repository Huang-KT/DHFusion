import os

import torch
from torch import nn
from torch.autograd import Function
from torch.utils.cpp_extension import load, _import_module_from_library


try:
    op_path = '/home/pp/.cache/torch_extensions/py38_cu113/fused'
    fused = _import_module_from_library('fused', op_path, True)
    print(f'Load fused from {op_path}')
except:
    module_path = os.path.dirname(__file__)
    fused = load(
        'fused',
        sources=[
            os.path.join(module_path, 'fused_bias_act.cpp'),
            os.path.join(module_path, 'fused_bias_act_kernel.cu'),
        ],
    )
    print(f'Build fused from cpp & cu files')

# import importlib
# def _import_module_from_library(module_name, path, is_python_module):
#     filepath = os.path.join(path, f"{module_name}.so")
#     if is_python_module:
#         # https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
#         spec = importlib.util.spec_from_file_location(module_name, filepath)
#         module = importlib.util.module_from_spec(spec)
#         assert isinstance(spec. loader, importlib.abc.Loader)
#         spec.loader.exec_module(module)
#         return module
#     else:
#         torch.ops.load_library(filepath)

# fused = _import_module_from_library('fused', '/home/pp/.cache/torch_extensions/py38_cu113/fused', True)

# module_path = os.path.dirname(__file__)
# fused = load(
#     'fused',
#     sources=[
#         os.path.join(module_path, 'fused_bias_act.cpp'),
#         os.path.join(module_path, 'fused_bias_act_kernel.cu'),
#     ],

# )


class FusedLeakyReLUFunctionBackward(Function):
    @staticmethod
    def forward(ctx, grad_output, out, negative_slope, scale):
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        empty = grad_output.new_empty(0)

        grad_input = fused.fused_bias_act(
            grad_output, empty, out, 3, 1, negative_slope, scale
        )

        dim = [0]

        if grad_input.ndim > 2:
            dim += list(range(2, grad_input.ndim))

        grad_bias = grad_input.sum(dim).detach()

        return grad_input, grad_bias

    @staticmethod
    def backward(ctx, gradgrad_input, gradgrad_bias):
        out, = ctx.saved_tensors
        gradgrad_out = fused.fused_bias_act(
            gradgrad_input, gradgrad_bias, out, 3, 1, ctx.negative_slope, ctx.scale
        )

        return gradgrad_out, None, None, None


class FusedLeakyReLUFunction(Function):
    @staticmethod
    def forward(ctx, input, bias, negative_slope, scale):
        empty = input.new_empty(0)
        out = fused.fused_bias_act(input, bias, empty, 3, 0, negative_slope, scale)
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, = ctx.saved_tensors

        grad_input, grad_bias = FusedLeakyReLUFunctionBackward.apply(
            grad_output, out, ctx.negative_slope, ctx.scale
        )

        return grad_input, grad_bias, None, None


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()

        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    return FusedLeakyReLUFunction.apply(input, bias, negative_slope, scale)
