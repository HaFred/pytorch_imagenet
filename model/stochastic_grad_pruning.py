"""
------------------------------------------------------------------------------

Copyright (C) 2020 Hong Kong Univ. of Sci. and Tech.

 "~/col_combined_pruning/stochastic_grad_pruning.py" - The API for stochastic
    gradient pruning

 Project: Signed Feedback Alignment with Fixed Point Arithmetic

 Authors:  Frederick Hong

 Cite/paper:

------------------------------------------------------------------------------
"""

from model.layer_wrapper import *
# from model.uniform_grad_pruning import linear_prune_grad, conv2d_prune_grad

_VERBOSE = False


# pure version of FA linear and conv are for resnet & vgg
class PureFALinear(nn.Module):
    def __init__(self, in_features, out_features, train_mode, angle_mea, percent=None, bias=False):
        super(PureFALinear, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features, bias=False)
        if train_mode in ['FA', 'Sign_symmetry_magnitude_normal', 'Sign_symmetry_magnitude_uniform',
                          'Sign_symmetry_only_sign', 'BFA']:
            self.fc = Layer_wrapper(module=self.fc, layer_type='fc', dim=self.fc.weight.shape, train_mode=train_mode,
                                    angle_mea=angle_mea)
        self.error_grad = None

    def forward(self, x):
        x = self.fc(x)
        if x.requires_grad:
            x.register_hook(self.grad_hook)
        return x

    def grad_hook(self, grad):  # this hook capture error grad (modulatory signal) only
        self.error_grad = grad
        return grad


class PureFAConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, train_mode, angle_mea, stride=1,
                 percent=None, padding=0, bias=False):
        super(PureFAConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        if train_mode in ['FA', 'Sign_symmetry_magnitude_normal', 'Sign_symmetry_magnitude_uniform',
                          'Sign_symmetry_only_sign', 'BFA']:
            self.conv = Layer_wrapper(module=self.conv, layer_type='conv', dim=self.conv.weight.shape,
                                      stride=stride, padding=padding, train_mode=train_mode, angle_mea=angle_mea)
        self.error_grad = None

    def forward(self, x):
        x = self.conv(x)
        if x.requires_grad:
            x.register_hook(self.grad_hook)  # this hook capture error grad (modulatory signal) only, no cal
        return x

    def grad_hook(self, grad):
        self.error_grad = grad
        return grad


# note that all the grad prune below support FA
# and the angle_mea could only be done in the FA_wrapper
class StochasticGradPruneLinear(nn.Module):

    def __init__(self, in_features, out_features, train_mode, angle_mea, prune_rate=None, bias=False):
        super(StochasticGradPruneLinear, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features, bias=False)
        if train_mode in ['FA', 'Sign_symmetry_magnitude_normal', 'Sign_symmetry_magnitude_uniform',
                          'Sign_symmetry_only_sign', 'BFA']:
            self.fc = Layer_wrapper(module=self.fc, layer_type='fc', dim=self.fc.weight.shape,
                                    train_mode=train_mode, angle_mea=angle_mea)
        self.error_grad = None
        self.final_tau = torch.tensor([0], requires_grad=False).cuda()
        self.percent = prune_rate

    def forward(self, x):
        x = self.fc(x)
        if x.requires_grad:
            x.register_hook(self.grad_hook)
            x = prune_grad(x, final_tau=self.final_tau, prune_rate=self.percent)
        return x

    def grad_hook(self, grad):
        self.error_grad = grad
        return grad


class StochasticGradPruneConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, train_mode, angle_mea, stride=1,
                 prune_rate=None, padding=0, bias=False):
        super(StochasticGradPruneConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        if train_mode in ['FA', 'Sign_symmetry_magnitude_normal', 'Sign_symmetry_magnitude_uniform',
                          'Sign_symmetry_only_sign', 'BFA']:
            self.conv = Layer_wrapper(module=self.conv, layer_type='conv', dim=self.conv.weight.shape,
                                      stride=stride, padding=padding, train_mode=train_mode, angle_mea=angle_mea)
        self.error_grad = None
        self.final_tau = torch.tensor([0], requires_grad=False).cuda()
        self.percent = prune_rate

    def forward(self, x):
        x = self.conv(x)
        if x.requires_grad:
            x.register_hook(self.grad_hook)
            x = prune_grad(x, final_tau=self.final_tau, prune_rate=self.percent)
        return x

    def grad_hook(self, grad):
        self.error_grad = grad
        return grad


class StochasticZeroGrad(Function):

    @staticmethod
    def forward(ctx, x, final_tau, prune_rate):
        ctx.save_for_backward(final_tau)
        ctx.prune_rate = prune_rate
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # final_tau = ctx.saved_tensors
        final_tau = calculate_tau(grad_output, ctx.prune_rate)

        if final_tau > 0:
            if _VERBOSE:
                print('the final tau is positive as expected')
            rand = final_tau * torch.rand(grad_output.shape, device="cuda", dtype=torch.float32)
            grad_abs = grad_output.abs()
            # print('rand shape is {}'.format(rand.shape))
            # print('grad_abs shape is {}'.format(grad_abs.shape))
            # print('rand one entry is {}'.format(rand[0,:]))
            # print('grad_abs one entry is {}'.format(grad_abs[0,:]))
            grad_input = torch.where(grad_abs < final_tau, final_tau * torch.sign(grad_output), grad_output)
            grad_input = torch.where(grad_abs < rand, torch.tensor([0], device="cuda", dtype=torch.float32), grad_input)
        else:
            if _VERBOSE:
                print('[Warning] the final tau is negative, not expected')
            grad_input = grad_output

        return grad_input, None, None


def prune_grad(x, final_tau, prune_rate):
    return StochasticZeroGrad().apply(x, final_tau, prune_rate)


"""
Results of different tau cal scheme, resnet-18, cifar-10:

                Accu    ZeroGrad
Direct std:     56.52   13805365 - first epoch (this needs the exact sigma for every normal instantiation)
sigma(grad):    95.29   21163190 - epoch 206 lr=0.1
esti sigma:     95.47   14316510 - epoch 199 lr=0.1
(with pi)
esti sigma:     95.62   15152338 - epoch 203 lr=0.1
(w/o pi)     
"""


# the standard deviation phi is estimated in a unbiased way, based on the grad_to_prune x
def calculate_tau(grad_to_prune, prune_rate):
    normal = torch.distributions.normal.Normal(0, 1)
    theta = normal.icdf(torch.tensor((1 + prune_rate / 100) / 2, requires_grad=False).cuda())
    # print('the theta is {}'.format(theta))
    mean = torch.mean(torch.abs(grad_to_prune).view(-1).detach())
    # remove pi for estimated sigma
    estimated_sigma = torch.mean(torch.abs(grad_to_prune).view(-1).detach())  # est is good enough to get 95% on cifar10
    sigma = torch.std(grad_to_prune)
    # tau = theta * math.sqrt(2 / math.pi) * mean
    tau = theta * estimated_sigma
    if prune_rate == 0:
        tau = 0
    if _VERBOSE:
        # print('pass in prune_rate {}'.format(prune_rate))
        # print('normal is {}'.format(normal))
        # print('theta is {}'.format(theta))
        # print('mean is {}'.format(mean))
        # print('tau is {}'.format(tau))
        print('sigma is {}'.format(sigma))
        print('estimated sigma is {}'.format(estimated_sigma))

    return tau

# def calculate_tau(grad_to_prune, prune_rate):  # reference: directly using grad std
#     sigma = torch.std(grad_to_prune)
#     normal = torch.distributions.normal.Normal(0, sigma)
#     tau = normal.icdf(torch.tensor((1 + prune_rate / 100) / 2, requires_grad=False).cuda())
#     if prune_rate == 0:
#         tau = 0
#     return tau
