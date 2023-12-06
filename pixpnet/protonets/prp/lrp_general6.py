"""
Copyright (c) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
Copyright (c) 2022 Srishti Gautam, Marina Hohne, Robert Jenssen, Michael Kampffmeyer

SPDX-License-Identifier: AGPL-3.0-or-later
SPDX-License-Identifier: MIT
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

#######################################################
#######################################################
# wrappers for autograd type modules
#######################################################
#######################################################


class ZeroparamWrapperClass(nn.Module):
    def __init__(self, module, autogradfunction):
        super(ZeroparamWrapperClass, self).__init__()
        self.module = module
        self.wrapper = autogradfunction

    def forward(self, x):
        y = self.wrapper.apply(x, self.module)
        return y


class OneParamWrapperClass(nn.Module):
    def __init__(self, module, autogradfunction, parameter1):
        super(OneParamWrapperClass, self).__init__()
        self.module = module
        self.wrapper = autogradfunction
        self.parameter1 = parameter1

    def forward(self, x):
        y = self.wrapper.apply(x, self.module, self.parameter1)
        return y


class Conv2DZBetaWrapperClass(nn.Module):
    def __init__(self, module, lrpignorebias, lowest=None, highest=None):
        super(Conv2DZBetaWrapperClass, self).__init__()

        if lowest is None:
            lowest = torch.min(torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]))
        if highest is None:
            highest = torch.max(torch.tensor([(1 - 0.485) / 0.229, (1 - 0.456) / 0.224, (1 - 0.406) / 0.225]))
        assert isinstance(module, nn.Conv2d)

        self.module = module
        self.wrapper = Conv2DZBetaWrapperFct()

        self.lrpignorebias = lrpignorebias

        self.lowest = lowest
        self.highest = highest

    def forward(self, x):
        y = self.wrapper.apply(x, self.module, self.lrpignorebias, self.lowest, self.highest)
        return y


class LRLookupNotFoundError(Exception):
    pass


VERBOSE = False


def get_lrpwrapperformodule(module, lrp_params, lrp_layer2method, thisis_inputconv_andiwant_zbeta=False):
    if isinstance(module, nn.ReLU):
        key = "nn.ReLU"
        if key not in lrp_layer2method:
            print("found no dictionary entry in " "lrp_layer2method for this module name:", key)
            raise LRLookupNotFoundError("found no dictionary entry in " "lrp_layer2method for this module name:", key)

        # default relu_wrapper_fct()
        autogradfunction = lrp_layer2method[key]()
        return ZeroparamWrapperClass(module, autogradfunction=autogradfunction)

    elif isinstance(module, nn.Sigmoid):
        key = "nn.Sigmoid"
        if key not in lrp_layer2method:
            print("found no dictionary entry in " "lrp_layer2method for this module name:", key)
            raise LRLookupNotFoundError("found no dictionary entry in " "lrp_layer2method for this module name:", key)

        # default relu_wrapper_fct()
        autogradfunction = lrp_layer2method[key]()
        return ZeroparamWrapperClass(module, autogradfunction=autogradfunction)

    elif isinstance(module, nn.BatchNorm2d):

        key = "nn.BatchNorm2d"
        if key not in lrp_layer2method:
            print("found no dictionary entry in " "lrp_layer2method for this module name:", key)
            raise LRLookupNotFoundError("found no dictionary entry in " "lrp_layer2method for this module name:", key)

        # default relu_wrapper_fct()
        autogradfunction = lrp_layer2method[key]()
        return ZeroparamWrapperClass(module, autogradfunction=autogradfunction)

    elif isinstance(module, nn.Linear):

        key = "nn.Linear"
        if key not in lrp_layer2method:
            print("found no dictionary entry in " "lrp_layer2method for this module name:", key)
            raise LRLookupNotFoundError("found no dictionary entry in " "lrp_layer2method for this module name:", key)

        # default linearlayer_eps_wrapper_fct()
        autogradfunction = lrp_layer2method[key]()
        return OneParamWrapperClass(module, autogradfunction=autogradfunction, parameter1=lrp_params["linear_eps"])

    elif isinstance(module, nn.Conv2d):
        if thisis_inputconv_andiwant_zbeta:
            return Conv2DZBetaWrapperClass(module, lrp_params["conv2d_ignorebias"])
        else:
            key = "nn.Conv2d"
            if key not in lrp_layer2method:
                print("found no dictionary entry in " "lrp_layer2method for this module name:", key)
                raise LRLookupNotFoundError(
                    "found no dictionary entry in " "lrp_layer2method for this module name:", key
                )

            # default conv2d_beta0_wrapper_fct()
            autogradfunction = lrp_layer2method[key]()
            return OneParamWrapperClass(
                module, autogradfunction=autogradfunction, parameter1=lrp_params["conv2d_ignorebias"]
            )

    elif isinstance(module, nn.AdaptiveAvgPool2d):

        key = "nn.AdaptiveAvgPool2d"
        if key not in lrp_layer2method:
            print("found no dictionary entry in " "lrp_layer2method for this module name:", key)
            raise LRLookupNotFoundError("found no dictionary entry in " "lrp_layer2method for this module name:", key)

        # default adaptiveavgpool2d_wrapper_fct()
        autogradfunction = lrp_layer2method[key]()
        return OneParamWrapperClass(module, autogradfunction=autogradfunction, parameter1=lrp_params["pooling_eps"])

    elif isinstance(module, nn.AvgPool2d):

        key = "nn.AvgPool2d"
        if key not in lrp_layer2method:
            print("found no dictionary entry in " "lrp_layer2method for this module name:", key)
            raise LRLookupNotFoundError("found no dictionary entry in " "lrp_layer2method for this module name:", key)

        # default adaptiveavgpool2d_wrapper_fct()
        autogradfunction = lrp_layer2method[key]()
        return OneParamWrapperClass(module, autogradfunction=autogradfunction, parameter1=lrp_params["pooling_eps"])

    elif isinstance(module, nn.MaxPool2d):

        key = "nn.MaxPool2d"
        if key not in lrp_layer2method:
            print("found no dictionary entry in " "lrp_layer2method for this module name:", key)
            raise LRLookupNotFoundError("found no dictionary entry in " "lrp_layer2method for this module name:", key)

        # default maxpool2d_wrapper_fct()
        autogradfunction = lrp_layer2method[key]()
        return ZeroparamWrapperClass(module, autogradfunction=autogradfunction)

    elif isinstance(module, SumStacked2):  # resnet specific

        key = "sum_stacked2"
        if key not in lrp_layer2method:
            print("found no dictionary entry in " "lrp_layer2method for this module name:", key)
            raise LRLookupNotFoundError("found no dictionary entry in " "lrp_layer2method for this module name:", key)

        # default eltwisesum_stacked2_eps_wrapper_fct()
        autogradfunction = lrp_layer2method[key]()
        return OneParamWrapperClass(module, autogradfunction=autogradfunction, parameter1=lrp_params["eltwise_eps"])

    elif isinstance(module, ClampLayer):  # densenet specific

        key = "clamplayer"
        if key not in lrp_layer2method:
            print("found no dictionary entry in " "lrp_layer2method for this module name:", key)
            raise LRLookupNotFoundError("found no dictionary entry in " "lrp_layer2method for this module name:", key)

        # default relu_wrapper_fct()
        autogradfunction = lrp_layer2method[key]()
        return ZeroparamWrapperClass(module, autogradfunction=autogradfunction)

    elif isinstance(module, TensorBiasedLinearLayer):  # densenet specific

        key = "tensorbiased_linearlayer"
        if key not in lrp_layer2method:
            print("found no dictionary entry in " "lrp_layer2method for this module name:", key)
            raise LRLookupNotFoundError("found no dictionary entry in " "lrp_layer2method for this module name:", key)

        # default relu_wrapper_fct()
        autogradfunction = lrp_layer2method[key]()
        return OneParamWrapperClass(module, autogradfunction=autogradfunction, parameter1=lrp_params["linear_eps"])

    elif isinstance(module, TensorBiasedConvLayer):  # densenet specific

        key = "tensorbiased_convlayer"
        if key not in lrp_layer2method:
            print("found no dictionary entry in " "lrp_layer2method for this module name:", key)
            raise LRLookupNotFoundError("found no dictionary entry in " "lrp_layer2method for this module name:", key)

        # default relu_wrapper_fct()
        autogradfunction = lrp_layer2method[key]()
        return OneParamWrapperClass(
            module, autogradfunction=autogradfunction, parameter1=lrp_params["conv2d_ignorebias"]
        )

    else:
        key = "nn.MaxPool2d"
        if key not in lrp_layer2method:
            print("found no dictionary entry in " "lrp_layer2method for this module name:", key)
            raise LRLookupNotFoundError("found no dictionary entry in " "lrp_layer2method for this module name:", key)

        # default maxpool2d_wrapper_fct()
        autogradfunction = lrp_layer2method[key]()
        return ZeroparamWrapperClass(module, autogradfunction=autogradfunction)
        print("found no lookup for this module:", module)
        raise LRLookupNotFoundError("found no lookup for this module:", module)


#######################################################
#######################################################
# canonization functions
#######################################################
#######################################################


def resetbn(bn):
    assert isinstance(bn, nn.BatchNorm2d)

    bnc = copy.deepcopy(bn)
    bnc.reset_parameters()

    return bnc


# vanilla fusion conv-bn --> conv(updatedparams)
def bnafterconv_overwrite_intoconv(conv, bn):  # after visatt

    if VERBOSE:
        print(conv, bn)

    assert isinstance(bn, nn.BatchNorm2d)
    assert isinstance(conv, nn.Conv2d)

    s = (bn.running_var + bn.eps) ** 0.5
    w = bn.weight
    b = bn.bias
    m = bn.running_mean
    conv.weight = torch.nn.Parameter(conv.weight * (w / s).reshape(-1, 1, 1, 1))

    if conv.bias is None:
        conv.bias = torch.nn.Parameter((0 - m) * (w / s) + b)
    else:
        conv.bias = torch.nn.Parameter((conv.bias - m) * (w / s) + b)
    return conv


# resnet stuff

###########################################################
#########################################################
###########################################################

# for resnet shortcut / residual connections
class EltwiseSum2(nn.Module):  # see torchray excitation backprop, using *inputs
    def __init__(self):
        super(EltwiseSum2, self).__init__()

    def forward(self, x1, x2):
        return x1 + x2


# densenet stuff

###########################################################
#########################################################
###########################################################

# bad name actually, threshrelu would be better
class ClampLayer(nn.Module):
    def __init__(self, thresh, w_bn_sign, forconv):
        super(ClampLayer, self).__init__()

        # thresh will be -b_bn*vareps / w_bn + mu_bn
        if forconv:
            self.thresh = thresh.reshape((-1, 1, 1))
            self.w_bn_sign = w_bn_sign.reshape((-1, 1, 1))
        else:
            # for linear that should be ok
            self.thresh = thresh
            self.w_bn_sign = w_bn_sign

    def forward(self, x):
        return (x - self.thresh) * (
            (x > self.thresh) * (self.w_bn_sign > 0) + (x < self.thresh) * (self.w_bn_sign < 0)
        ) + self.thresh


class TensorBiasedLinearLayer(nn.Module):
    def __init__(self, in_features, out_features, newweight, newbias):
        super(TensorBiasedLinearLayer, self).__init__()
        assert newbias.numel() == out_features

        self.linearbase = nn.Linear(in_features, out_features, bias=False)
        self.linearbase.weight = torch.nn.Parameter(newweight)
        self.biastensor = torch.nn.Parameter(newbias)

        # this is pure convenience
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        y = self.linearbase.forward(x) + self.biastensor
        return y


class TensorBiasedConvLayer(nn.Module):
    def _clone_module(self, module):
        clone = nn.Conv2d(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            **{attr: getattr(module, attr) for attr in ["stride", "padding", "dilation", "groups"]}
        )
        return clone.to(module.weight.device)

    def __init__(self, newweight, baseconv, inputfornewbias):
        super(TensorBiasedConvLayer, self).__init__()

        self.baseconv = baseconv  # evtl store weights, clone mod

        self.inputfornewbias = inputfornewbias

        self.conv = self._clone_module(baseconv)
        self.conv.weight = torch.nn.Parameter(newweight)
        self.conv.bias = None

        self.biasmode = "neutr"

    def gettensorbias(self, x):

        with torch.no_grad():
            tensorbias = self.baseconv(
                self.inputfornewbias.unsqueeze(1).unsqueeze(2).repeat((1, x.shape[2], x.shape[3])).unsqueeze(0)
            )
        if VERBOSE:
            print("done tensorbias", tensorbias.shape)
        return tensorbias

    def forward(self, x):
        if VERBOSE:
            print("tensorbias fwd", x.shape)
        if len(x.shape) != 4:
            print("bad tensor length")
            exit()
        if VERBOSE:
            if self.inputfornewbias is not None:
                print("self.inputfornewbias.shape", self.inputfornewbias.shape)
            else:
                print("self.inputfornewbias", self.inputfornewbias)
        if self.inputfornewbias is None:
            return self.conv.forward(x)  # z
        else:
            b = self.gettensorbias(x)
            if self.biasmode == "neutr":
                return self.conv.forward(x) + b
            elif self.biasmode == "pos":
                return self.conv.forward(x) + torch.clamp(b, min=0)  # z
            elif self.biasmode == "neg":
                return self.conv.forward(x) + torch.clamp(b, max=0)  # z


#######################################################
#######################################################
# autograd type modules
#######################################################
#######################################################


class Conv2DBeta0WrapperFct(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, x, module, lrpignorebias):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """

        def configvalues_totensorlist(module):
            propertynames = ["in_channels", "out_channels", "kernel_size", "stride", "padding", "dilation", "groups"]
            values = []
            for attr in propertynames:
                v = getattr(module, attr)
                # convert it into tensor
                # has no treatment for booleans yet
                if isinstance(v, int):
                    v = torch.tensor([v], dtype=torch.int32, device=module.weight.device)
                elif isinstance(v, tuple):
                    ################
                    ################
                    # FAILMODE: if it is not a tuple of ints but e.g. a tuple of floats, or a tuple of a tuple

                    v = torch.tensor(v, dtype=torch.int32, device=module.weight.device)
                else:
                    print("v is neither int nor tuple. unexpected")
                    exit()
                values.append(v)
            return propertynames, values

        # stash module config params and trainable params
        propertynames, values = configvalues_totensorlist(module)

        if module.bias is None:
            bias = None
        else:
            bias = module.bias.data.clone()
        lrpignorebiastensor = torch.tensor([lrpignorebias], dtype=torch.bool, device=module.weight.device)
        ctx.save_for_backward(
            x, module.weight.data.clone(), bias, lrpignorebiastensor, *values
        )  # *values unpacks the list

        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """

        input_, conv2dweight, conv2dbias, lrpignorebiastensor, *values = ctx.saved_tensors

        #######################################################################
        # reconstruct dictionary of config parameters
        def tensorlist_todict(values):
            propertynames = ["in_channels", "out_channels", "kernel_size", "stride", "padding", "dilation", "groups"]
            # but needs to turn tensors to ints or tuples!
            paramsdict = {}
            for i, n in enumerate(propertynames):
                v = values[i]
                if v.numel == 1:
                    paramsdict[n] = v.item()  # to cpu?
                else:
                    alist = v.tolist()
                    if len(alist) == 1:
                        paramsdict[n] = alist[0]
                    else:
                        paramsdict[n] = tuple(alist)
            return paramsdict

        #######################################################################
        paramsdict = tensorlist_todict(values)

        if conv2dbias is None:
            module = nn.Conv2d(**paramsdict, bias=False)
        else:
            module = nn.Conv2d(**paramsdict, bias=True)
            module.bias = torch.nn.Parameter(conv2dbias)

        module.weight = torch.nn.Parameter(conv2dweight)

        pnconv = PosNegConv(module, ignorebias=lrpignorebiastensor.item())

        X = input_.clone().detach().requires_grad_(True)
        R = lrp_backward(_input=X, layer=pnconv, relevance_output=grad_output[0], eps0=1e-12, eps=0)

        return R, None, None


class Conv2DZBetaWrapperFct(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, x, module, lrpignorebias, lowest, highest):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward
        method.
        """

        def configvalues_totensorlist(module):
            propertynames = ["in_channels", "out_channels", "kernel_size", "stride", "padding", "dilation", "groups"]
            values = []
            for attr in propertynames:
                v = getattr(module, attr)
                # convert it into tensor
                # has no treatment for booleans yet
                if isinstance(v, int):
                    v = torch.tensor([v], dtype=torch.int32, device=module.weight.device)
                elif isinstance(v, tuple):
                    ################
                    ################
                    # FAILMODE: if it is not a tuple of ints but e.g. a tuple
                    # of floats, or a tuple of a tuple

                    v = torch.tensor(v, dtype=torch.int32, device=module.weight.device)
                else:
                    print("v is neither int nor tuple. unexpected")
                    exit()
                values.append(v)
            return propertynames, values

        # stash module config params and trainable params
        propertynames, values = configvalues_totensorlist(module)

        if module.bias is None:
            bias = None
        else:
            bias = module.bias.data.clone()
        lrpignorebiastensor = torch.tensor([lrpignorebias], dtype=torch.bool, device=module.weight.device)
        ctx.save_for_backward(
            x,
            module.weight.data.clone(),
            bias,
            lrpignorebiastensor,
            lowest.to(module.weight.device),
            highest.to(module.weight.device),
            *values
        )  # *values unpacks the list

        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        (input_, conv2dweight, conv2dbias, lrpignorebiastensor, lowest_, highest_, *values) = ctx.saved_tensors

        #######################################################################
        # reconstruct dictionary of config parameters
        def tensorlist_todict(values):
            propertynames = ["in_channels", "out_channels", "kernel_size", "stride", "padding", "dilation", "groups"]
            # but needs to turn tensors to ints or tuples!
            paramsdict = {}
            for i, n in enumerate(propertynames):
                v = values[i]
                if v.numel == 1:
                    paramsdict[n] = v.item()  # to cpu?
                else:
                    alist = v.tolist()
                    if len(alist) == 1:
                        paramsdict[n] = alist[0]
                    else:
                        paramsdict[n] = tuple(alist)
            return paramsdict

        #######################################################################
        paramsdict = tensorlist_todict(values)

        if conv2dbias is None:
            module = nn.Conv2d(**paramsdict, bias=False)
        else:
            module = nn.Conv2d(**paramsdict, bias=True)
            module.bias = torch.nn.Parameter(conv2dbias)

        module.weight = torch.nn.Parameter(conv2dweight)

        any_conv = AnySignConv(module, ignorebias=lrpignorebiastensor.item())

        X = input_.clone().detach().requires_grad_(True)
        L = (lowest_ * torch.ones_like(X)).requires_grad_(True)
        H = (highest_ * torch.ones_like(X)).requires_grad_(True)

        with torch.enable_grad():
            Z = (
                any_conv.forward(mode="justasitis", x=X)
                - any_conv.forward(mode="pos", x=L)
                - any_conv.forward(mode="neg", x=H)
            )
            S = safe_divide(grad_output[0].clone().detach(), Z.clone().detach(), eps0=1e-6, eps=1e-6)
            Z.backward(S)
            R = (X * X.grad + L * L.grad + H * H.grad).detach()

        # for (x, conv2dclass,lrpignorebias, lowest, highest)
        return R, None, None, None, None


class AdaptiveAvgPool2DWrapperFct(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, x, module, eps):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """

        def configvalues_totensorlist(module, device):

            propertynames = ["output_size"]
            values = []
            for attr in propertynames:
                v = getattr(module, attr)
                # convert it into tensor
                # has no treatment for booleans yet
                if isinstance(v, int):
                    v = torch.tensor([v], dtype=torch.int32, device=device)
                elif isinstance(v, tuple):
                    v = torch.tensor(v, dtype=torch.int32, device=device)
                else:
                    print("v is neither int nor tuple. unexpected")
                    exit()
                values.append(v)
            return propertynames, values

        # stash module config params and trainable params
        propertynames, values = configvalues_totensorlist(module, x.device)
        epstensor = torch.tensor([eps], dtype=torch.float32, device=x.device)
        ctx.save_for_backward(x, epstensor, *values)  # *values unpacks the list

        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """

        input_, epstensor, *values = ctx.saved_tensors

        #######################################################################
        # reconstruct dictionary of config parameters
        def tensorlist_todict(values):
            propertynames = ["output_size"]
            # idea: paramsdict={ n: values[i]
            #  for i,n in enumerate(propertynames)  }
            # but needs to turn tensors to ints or tuples!
            paramsdict = {}
            for i, n in enumerate(propertynames):
                v = values[i]
                if v.numel == 1:
                    paramsdict[n] = v.item()  # to cpu?
                else:
                    alist = v.tolist()
                    if len(alist) == 1:
                        paramsdict[n] = alist[0]
                    else:
                        paramsdict[n] = tuple(alist)
            return paramsdict

        #######################################################################
        paramsdict = tensorlist_todict(values)
        eps = epstensor.item()

        # class instantiation
        layerclass = torch.nn.AdaptiveAvgPool2d(**paramsdict)

        X = input_.clone().detach().requires_grad_(True)
        R = lrp_backward(_input=X, layer=layerclass, relevance_output=grad_output[0], eps0=eps, eps=eps)

        return R, None, None


class MaxPool2DWrapperFct(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, x, module):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward
        method.
        """

        def configvalues_totensorlist(module, device):

            propertynames = ["kernel_size", "stride", "padding", "dilation", "return_indices", "ceil_mode"]
            values = []
            for attr in propertynames:
                v = getattr(module, attr)
                # convert it into tensor
                # has no treatment for booleans yet
                if isinstance(v, bool):
                    v = torch.tensor([v], dtype=torch.bool, device=device)
                elif isinstance(v, int):
                    v = torch.tensor([v], dtype=torch.int32, device=device)
                elif isinstance(v, bool):

                    v = torch.tensor([v], dtype=torch.int32, device=device)
                elif isinstance(v, tuple):
                    ################
                    ################
                    # FAILMODE: if it is not a tuple of ints but e.g. a tuple
                    # of floats, or a tuple of a tuple

                    v = torch.tensor(v, dtype=torch.int32, device=device)
                else:
                    print("v is neither int nor tuple. unexpected")
                    exit()
                values.append(v)
            return propertynames, values

        # stash module config params and trainable params
        propertynames, values = configvalues_totensorlist(module, x.device)
        ctx.save_for_backward(x, *values)  # *values unpacks the list

        if VERBOSE:
            print("maxpool2d custom forward")
        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """

        input_, *values = ctx.saved_tensors

        #######################################################################
        # reconstruct dictionary of config parameters
        def tensorlist_todict(values):
            propertynames = ["kernel_size", "stride", "padding", "dilation", "return_indices", "ceil_mode"]
            # idea: paramsdict={ n: values[i]
            #  for i,n in enumerate(propertynames)  }
            # but needs to turn tensors to ints or tuples!
            paramsdict = {}
            for i, n in enumerate(propertynames):
                v = values[i]
                if v.numel == 1:
                    paramsdict[n] = v.item()  # to cpu?
                else:
                    alist = v.tolist()
                    if len(alist) == 1:
                        paramsdict[n] = alist[0]
                    else:
                        paramsdict[n] = tuple(alist)
            return paramsdict

        paramsdict = tensorlist_todict(values)

        layerclass = torch.nn.MaxPool2d(**paramsdict)

        X = input_.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            Z = layerclass.forward(X)
        relevance_output_data = grad_output[0].clone().detach().unsqueeze(0)
        Z.backward(relevance_output_data)
        R = X.grad

        return R, None


# to be used with generic_activation_pool_wrapper_class(module,this)
class ReluWrapperFct(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, x, module):
        # stash module config params and trainable params
        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


# to be used with generic_activation_pool_wrapper_class(module,this)
class SigmoidWrapperFct(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, x, module):
        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


# lineareps_wrapper_fct
class LinearLayerEpsWrapperFct(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, x, module, eps):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward
        method.
        """

        def configvalues_totensorlist(module):

            propertynames = ["in_features", "out_features"]
            values = []
            for attr in propertynames:
                v = getattr(module, attr)
                # convert it into tensor
                # has no treatment for booleans yet
                if isinstance(v, int):
                    v = torch.tensor([v], dtype=torch.int32, device=module.weight.device)
                elif isinstance(v, tuple):
                    ################
                    ################
                    # FAILMODE: if it is not a tuple of ints but e.g. a tuple
                    # of floats, or a tuple of a tuple

                    v = torch.tensor(v, dtype=torch.int32, device=module.weight.device)
                else:
                    print("v is neither int nor tuple. unexpected")
                    exit()
                values.append(v)
            return propertynames, values

        # stash module config params and trainable params
        propertynames, values = configvalues_totensorlist(module)
        epstensor = torch.tensor([eps], dtype=torch.float32, device=x.device)

        if module.bias is None:
            bias = None
        else:
            bias = module.bias.data.clone()
        ctx.save_for_backward(x, module.weight.data.clone(), bias, epstensor, *values)  # *values unpacks the list

        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """

        input_, weight, bias, epstensor, *values = ctx.saved_tensors

        #######################################################################
        # reconstruct dictionary of config parameters
        def tensorlist_todict(values):
            propertynames = ["in_features", "out_features"]
            # but needs to turn tensors to ints or tuples!
            paramsdict = {}
            for i, n in enumerate(propertynames):
                v = values[i]
                if v.numel == 1:
                    paramsdict[n] = v.item()  # to cpu?
                else:
                    alist = v.tolist()
                    if len(alist) == 1:
                        paramsdict[n] = alist[0]
                    else:
                        paramsdict[n] = tuple(alist)
            return paramsdict

        #######################################################################
        paramsdict = tensorlist_todict(values)

        if bias is None:
            module = nn.Linear(**paramsdict, bias=False)
        else:
            module = nn.Linear(**paramsdict, bias=True)
            module.bias = torch.nn.Parameter(bias)

        module.weight = torch.nn.Parameter(weight)

        eps = epstensor.item()
        X = input_.clone().detach().requires_grad_(True)
        R = lrp_backward(_input=X, layer=module, relevance_output=grad_output[0], eps0=eps, eps=eps)

        return R, None, None


class SumStacked2(nn.Module):
    def __init__(self):
        super(SumStacked2, self).__init__()

    @staticmethod
    def forward(x):  # from X=torch.stack([X0, X1], dim=0)
        assert x.shape[0] == 2
        return torch.sum(x, dim=0)


# to be used with generic_activation_pool_wrapper_class(module,this)
class EltwiseSumStacked2EpsWrapperFct(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, stackedx, module, eps):
        epstensor = torch.tensor([eps], dtype=torch.float32, device=stackedx.device)
        ctx.save_for_backward(stackedx, epstensor)
        return module.forward(stackedx)

    @staticmethod
    def backward(ctx, grad_output):
        stackedx, epstensor = ctx.saved_tensors

        X = stackedx.clone().detach().requires_grad_(True)

        eps = epstensor.item()

        s2 = SumStacked2().to(X.device)
        Rtmp = lrp_backward(_input=X, layer=s2, relevance_output=grad_output[0], eps0=eps, eps=eps)

        return Rtmp, None, None


#######################################################
#######################################################
# aux input classes
#######################################################
#######################################################


class PosNegConv(nn.Module):
    def _clone_module(self, module):
        clone = nn.Conv2d(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            **{attr: getattr(module, attr) for attr in ["stride", "padding", "dilation", "groups"]}
        )
        return clone.to(module.weight.device)

    def __init__(self, conv, ignorebias):
        super(PosNegConv, self).__init__()

        self.posconv = self._clone_module(conv)
        self.posconv.weight = torch.nn.Parameter(conv.weight.data.clone().clamp(min=0)).to(conv.weight.device)

        self.negconv = self._clone_module(conv)
        self.negconv.weight = torch.nn.Parameter(conv.weight.data.clone().clamp(max=0)).to(conv.weight.device)

        self.anyconv = self._clone_module(conv)
        self.anyconv.weight = torch.nn.Parameter(conv.weight.data.clone()).to(conv.weight.device)

        if ignorebias:
            self.posconv.bias = None
            self.negconv.bias = None
        else:
            if conv.bias is not None:
                self.posconv.bias = torch.nn.Parameter(conv.bias.data.clone().clamp(min=0))
                self.negconv.bias = torch.nn.Parameter(conv.bias.data.clone().clamp(max=0))

    def forward(self, x):
        vp = self.posconv(torch.clamp(x, min=0))
        vn = self.negconv(torch.clamp(x, max=0))
        return vp + vn


class AnySignConv(nn.Module):
    def _clone_module(self, module):
        clone = nn.Conv2d(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            **{attr: getattr(module, attr) for attr in ["stride", "padding", "dilation", "groups"]}
        )
        return clone.to(module.weight.device)

    def __init__(self, conv, ignorebias):
        super(AnySignConv, self).__init__()

        self.posconv = self._clone_module(conv)
        self.posconv.weight = torch.nn.Parameter(conv.weight.data.clone().clamp(min=0)).to(conv.weight.device)

        self.negconv = self._clone_module(conv)
        self.negconv.weight = torch.nn.Parameter(conv.weight.data.clone().clamp(max=0)).to(conv.weight.device)

        self.jusconv = self._clone_module(conv)
        self.jusconv.weight = torch.nn.Parameter(conv.weight.data.clone()).to(conv.weight.device)

        if ignorebias:
            self.posconv.bias = None
            self.negconv.bias = None
            self.jusconv.bias = None
        else:
            if conv.bias is not None:
                self.posconv.bias = torch.nn.Parameter(conv.bias.data.clone().clamp(min=0)).to(conv.weight.device)
                self.negconv.bias = torch.nn.Parameter(conv.bias.data.clone().clamp(max=0)).to(conv.weight.device)
                self.jusconv.bias = torch.nn.Parameter(conv.bias.data.clone()).to(conv.weight.device)

    def forward(self, mode, x):
        if mode == "pos":
            return self.posconv.forward(x)
        elif mode == "neg":
            return self.negconv.forward(x)
        elif mode == "justasitis":
            return self.jusconv.forward(x)
        else:
            raise NotImplementedError("anysign_conv notimpl mode: " + str(mode))


class PosNegConvTensorBiased(nn.Module):
    def _clone_module(self, module):
        clone = nn.Conv2d(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            **{attr: getattr(module, attr) for attr in ["stride", "padding", "dilation", "groups"]}
        )
        return clone.to(module.weight.device)

    def __init__(self, tensorbiasedconv, ignorebias):
        super(PosNegConvTensorBiased, self).__init__()

        self.posconv = TensorBiasedConvLayer(
            tensorbiasedconv.conv.weight, tensorbiasedconv.baseconv, tensorbiasedconv.inputfornewbias
        )
        self.negconv = TensorBiasedConvLayer(
            tensorbiasedconv.conv.weight, tensorbiasedconv.baseconv, tensorbiasedconv.inputfornewbias
        )

        self.posconv.conv.weight = torch.nn.Parameter(tensorbiasedconv.conv.weight.data.clone().clamp(min=0)).to(
            tensorbiasedconv.conv.weight.device
        )

        self.negconv.conv.weight = torch.nn.Parameter(tensorbiasedconv.conv.weight.data.clone().clamp(max=0)).to(
            tensorbiasedconv.conv.weight.device
        )

        if ignorebias:
            self.posconv.inputfornewbias = None
            self.negconv.inputfornewbias = None
        else:
            self.posconv.biasmode = "pos"
            self.negconv.biasmode = "neg"

        if VERBOSE:
            print("posnegconv_tensorbiased done init")

    def forward(self, x):
        vp = self.posconv(torch.clamp(x, min=0))
        vn = self.negconv(torch.clamp(x, max=0))
        return vp + vn


#######################################################
#######################################################
# #base routines
#######################################################
#######################################################


def safe_divide(numerator, divisor, eps0, eps):
    return numerator / (divisor + eps0 * (divisor == 0).to(divisor) + eps * divisor.sign())


def lrp_backward(_input, layer, relevance_output, eps0, eps):
    """
    Performs the LRP backward pass, implemented as standard forward and backward
    passes.
    """
    relevance_output_data = relevance_output.clone().detach()
    with torch.enable_grad():
        Z = layer(_input)
    S = safe_divide(relevance_output_data, Z.clone().detach(), eps0, eps)

    Z.backward(S)
    relevance_input = _input.data * _input.grad.data
    return relevance_input


class L2LRPClass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, conv_features, model):
        # *values unpacks the list
        ctx.save_for_backward(conv_features, model.prototype_vectors)
        if VERBOSE:
            print("l2 custom forward")
        x2 = conv_features**2
        x2_patch_sum = F.conv2d(input=x2, weight=model.ones)

        p2 = model.prototype_vectors**2
        p2 = torch.sum(p2, dim=(1, 2, 3))
        # p2 is a vector of shape (num_prototypes,)
        # then we reshape it to (num_prototypes, 1, 1)
        p2_reshape = p2.view(-1, 1, 1)

        xp = F.conv2d(input=conv_features, weight=model.prototype_vectors)
        intermediate_result = -2 * xp + p2_reshape  # use broadcast
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(x2_patch_sum + intermediate_result)

        similarities = torch.log((distances + 1) / (distances + model.epsilon))

        return similarities

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the
        loss with respect to the output, and we need to compute the gradient of
        the loss with respect to the input.
        """
        if VERBOSE:
            print("l2 custom backward")
        conv, prototypes = ctx.saved_tensors
        i = conv.shape[2]
        j = conv.shape[3]
        c = conv.shape[1]
        p = prototypes.shape[0]

        # Broadcast conv to Nxsize(conv) (No. of prototypes)
        conv = conv.repeat(p, 1, 1, 1)
        prototype = prototypes.repeat(1, 1, i, j)

        conv = conv.squeeze()

        l2 = (conv - prototype) ** 2
        d = 1 / (l2**2 + 1e-12)

        denom = torch.sum(d, dim=1, keepdim=True) + 1e-12
        denom = denom.repeat(1, c, 1, 1) + 1e-12
        R = torch.div(d, denom)

        grad_output = grad_output.repeat(c, 1, 1, 1)
        grad_output = grad_output.permute(1, 0, 2, 3)

        R = R * grad_output

        R = torch.sum(R, dim=0)

        R = torch.unsqueeze(R, dim=0)

        return R, None, None


class CosineDistLRPClass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, conv_features, model):
        ctx.save_for_backward(conv_features, model.prototype_vectors)
        if VERBOSE:
            print("cosine custom forward")

        # An alternative distance metric used in TesNet. Alternative to
        #  l2_convolution
        x = F.normalize(conv_features, p=2, dim=1)
        prototype_vectors = F.normalize(model.prototype_vectors, p=2, dim=1)
        similarities = F.conv2d(input=x, weight=prototype_vectors)
        # clip similarities in the range [-1, +1] (numerical error can
        #  cause similarities to be outside this range)
        similarities = torch.clamp(similarities, -1, 1)
        distances = 1 - similarities  # bounded [0, 2]

        similarities = torch.log((distances + 1) / (distances + model.epsilon))

        return similarities

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the
        loss with respect to the output, and we need to compute the gradient of
        the loss with respect to the input.
        """
        if VERBOSE:
            print("cosine custom backward")
        conv, prototypes = ctx.saved_tensors
        i = conv.shape[2]
        j = conv.shape[3]
        c = conv.shape[1]
        p = prototypes.shape[0]

        # Broadcast conv to Nxsize(conv) (No. of prototypes)
        conv = conv.repeat(p, 1, 1, 1)  # NP x D x Hz x Wz
        prototype = prototypes.repeat(1, 1, i, j)  # P x D x Hz x Wz

        conv = conv.squeeze()  # think this does nothing

        cosine_dists = 1 - F.normalize(prototype, p=2, dim=1) * F.normalize(conv, p=2, dim=1)
        d = 1 / (cosine_dists**2 + 1e-12)

        denom = torch.sum(d, dim=1, keepdim=True) + 1e-12
        denom = denom.repeat(1, c, 1, 1) + 1e-12
        R = torch.div(d, denom)

        grad_output = grad_output.repeat(c, 1, 1, 1)
        grad_output = grad_output.permute(1, 0, 2, 3)

        R = R * grad_output

        R = torch.sum(R, dim=0)

        R = torch.unsqueeze(R, dim=0)

        return R, None, None
