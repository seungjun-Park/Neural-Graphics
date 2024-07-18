import torch
import torch.nn as nn
import torch.nn.functional as F


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, *args):
        ctx.run_function = run_function
        ctx.save_for_backward(*args)
        with torch.no_grad():
            outputs = ctx.run_function(*args)
        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        inputs = ctx.saved_tensors
        with torch.enable_grad():
            inputs = [inp.detach().requires_grad_(True) for inp in inputs]
            outputs = ctx.run_function(*inputs)
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs, )
        grads = torch.autograd.grad(outputs, inputs, grad_outputs, retain_graph=True, allow_unused=True)
        return (None, ) + grads


def checkpoint(func, inputs, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs)
        return CheckpointFunction.apply(func, *args)
    else:
        return func(*inputs)
