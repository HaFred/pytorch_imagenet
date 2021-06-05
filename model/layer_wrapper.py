# -*- coding: utf-8 -*-

"""
------------------------------------------------------------------------------

Copyright (C) 2020 Hong Kong Univ. of Sci. and Tech.

 "layer_wrapper.py" - Definition of hooks that allow performing FA, SFA, and DRTP training.

fixme note:
    #  Here FA works for both bias=None or bias exists, apparently the return of backward hook need to be changed
    #   adaptively. But in fact, FA only requires fixed-uniform-distributed feedback_weight.
    #   Thus, no need to assign extended autograd.Function.hook to FA, but for SFA, and DRTP, due to their
    #   correlation with the other modelssss counterpart in each component, it needs the Hook function.

    #   In fact, the FA_wrapper is just a simply version of TrainingHook, it can be absorbed by TrainingHook,
    #   and since the backward is the same as BP, so we make it as an FA_wrapper.

    # note2:
    # For ResNet-18, the cossim brought by angle_hook cannot register the Conv1 layer, since
    # there is no preceding layer for Conv1, thus the autograd won't go over the forward fn to compute the grad,
    # thus Angle_hook captures nothing.

    # fixme we can somehow get the Conv1 angle by forcing the autograd compute the first layer, but it just takes time

 Project: Signed Feedback Alignment with Fixed Point Arithmetic

 Authors:  Frederick Hong

 Cite/paper:
------------------------------------------------------------------------------
"""

import torch
import torch.nn as nn
from torch.autograd import Function


class Layer_wrapper(nn.Module):
    def __init__(self, module, layer_type, dim, train_mode, angle_mea=True, stride=None, padding=None):
        super(Layer_wrapper, self).__init__()
        self.module = module
        self.scale = 1
        self.layer_type = layer_type
        self.stride = stride
        self.padding = padding
        self.output_grad = None
        self.x_shape = None
        self.train_mode = train_mode
        self.angle_mea = angle_mea
        self.cossim = torch.tensor([0], requires_grad=False).cuda()

        # FA feedback weights definition
        # initiate a tensor for fixed fb weights,
        # nn.params is used for learnable params (comparing to nn.register_buffer),
        # but it only brings the `require_grad` attribute to the fixed_fa_w,
        # it can still be not learned during backward
        if train_mode == 'FA':
            self.fixed_fb_weights = nn.Parameter(torch.Tensor(torch.Size(dim)))
            self.reset_weights()  # kaiming_uniform as in nn.module._ConvNd.reset_parameters
        elif train_mode == 'BFA':  # take the random feedback sign bit in KAIST paper
            self.fixed_fb_weights = nn.Parameter(torch.Tensor(torch.Size(dim)))
            self.reset_weights()
        elif train_mode == 'Sign_symmetry_magnitude_normal':
            self.fixed_fb_weights = nn.Parameter(torch.Tensor(torch.Size(dim)))
            self.fixed_fb_weights.requires_grad = False
            # self.fixed_fb_weights.normal_(0, 1)  # mean, std
            torch.nn.init.kaiming_normal_(self.fixed_fb_weights)
        elif train_mode == 'Sign_symmetry_magnitude_uniform':
            self.fixed_fb_weights = nn.Parameter(torch.Tensor(torch.Size(dim)))
            # self.fixed_fb_weights.data.uniform_(0, 1)  # from, to; this uniform is not well
            self.reset_weights()
        elif train_mode == 'Sign_symmetry_only_sign':  # the actual uSF in HowWeight paper
            self.fixed_fb_weights = nn.Parameter(module.weight.sign().detach_())
            self.fixed_fb_weights.requires_grad = False
        elif train_mode == 'uniform-signsym-feedback-SN':
            # todo finish the uniform-signsym-strict-norm here as in the paper
            self.fixed_fb_weights = nn.Parameter(module.weight.sign().detach_())
        else:
            raise NameError("No {} supported for wrapper".format(train_mode))

    def forward(self, x):
        if x.requires_grad:
            x.register_hook(self.FA_hook_pre)
            self.x_shape = x.shape

            # below is for modulatory signal angle
            if self.angle_mea:
                # x = self.angle_hook(x, self.output_grad, self.fixed_fb_weights, self.module.weight, self.angle)
                x.register_hook(self.Angle_hook)
            if self.train_mode in ['Sign_symmetry_magnitude_normal', 'Sign_symmetry_magnitude_uniform']:
                # * is matrix element-wise mul, the below two lines are same
                self.fixed_fb_weights.data = torch.abs(self.fixed_fb_weights) * self.module.weight.sign().detach_()
                # self.fixed_fb_weights.data = torch.abs(self.fixed_fb_weights) * torch.sign(self.module.weight)
            elif self.train_mode == 'Sign_symmetry_only_sign':
                self.fixed_fb_weights.data = self.module.weight.sign().detach_()  # weight is from dcg, needs to detach
            elif self.train_mode == 'BFA':
                self.fixed_fb_weights.data = self.fixed_fb_weights.sign()
            x = self.module(x)
            x.register_hook(self.FA_hook_post)

            return x
        else:
            return self.module(x)

    def FA_hook_pre(self, grad):
        if self.output_grad is not None:
            if (self.layer_type == "fc"):
                return self.output_grad.mm(self.fixed_fb_weights)
            elif (self.layer_type == "conv"):
                return torch.nn.grad.conv2d_input(self.x_shape, self.fixed_fb_weights, self.output_grad,
                                                  self.stride, self.padding)
            else:
                raise NameError("=== ERROR: layer type " + str(self.layer_type) + " is not supported in FA wrapper")
        else:
            return grad

    # the output_grad used for FA
    def FA_hook_post(self, grad):
        self.output_grad = grad
        return grad

    def Angle_hook(self, grad):
        if self.output_grad is None:
            raise ValueError("ERROR: Why the output_grad before post hook is still None?")
        # below follows the definition for angle in 'Random synaptic feedback weights support error
        # backpropagation for deep learning'
        elif self.layer_type == 'fc':
            self.cossim = nn.CosineSimilarity()(torch.abs(self.output_grad.mm(self.fixed_fb_weights).reshape([1, -1])),
                                                torch.abs(self.output_grad.mm(self.module.weight).reshape([1, -1])))
        elif self.layer_type == 'conv':
            self.cossim = nn.CosineSimilarity()(torch.abs(torch.nn.grad.conv2d_input(self.x_shape, self.fixed_fb_weights,
                                                                                     self.output_grad, self.stride, self.padding).reshape([1, -1])),
                                                torch.abs(torch.nn.grad.conv2d_input(self.x_shape, self.module.weight,
                                                                                    self.output_grad, self.stride, self.padding).reshape([1, -1])))
        else:
            raise ValueError("ERROR: The {} is not supported".format(self.layer_type))

        return grad

    def reset_weights(self):
        torch.nn.init.kaiming_uniform_(self.fixed_fb_weights)
        self.fixed_fb_weights.requires_grad = False

# deprecated below
# class AngleHook(nn.Module):
#     def __init__(self):
#         super(AngleHook, self).__init__()
#
#     def forward(self, input, output_grad, fixed_fb_weights, weight, angle):
#         return angleHook(input, output_grad, fixed_fb_weights, weight, angle)
#
#
# class AngleHookFunction(Function):
#
#     @staticmethod
#     def forward(ctx, input, output_grad, fixed_fb_weights, weight, angle):
#         ctx.save_for_backward(output_grad, fixed_fb_weights, weight, angle)
#         return input
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         output_grad, fixed_fb_weights, weight, angle = ctx.saved_variables
#         angle = nn.CosineSimilarity()(torch.abs(output_grad.mm(fixed_fb_weights).reshape([1, -1])),
#                                       torch.abs(output_grad.mm(weight).reshape([1, -1])))
#         return grad_output, None, None, None, angle
#
#
# angleHook = AngleHookFunction.apply
