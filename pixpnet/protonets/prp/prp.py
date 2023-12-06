"""
Copyright (c) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)
Copyright (c) 2022 Srishti Gautam, Marina Hohne, Robert Jenssen, Michael Kampffmeyer

SPDX-License-Identifier: AGPL-3.0-or-later
SPDX-License-Identifier: MIT
"""
import copy
from collections import OrderedDict

import torch
from torch import nn
from torchvision import datasets

from pixpnet.protonets.prp.lrp_general6 import (
    AdaptiveAvgPool2DWrapperFct,
    Conv2DBeta0WrapperFct,
    CosineDistLRPClass,
    EltwiseSumStacked2EpsWrapperFct,
    L2LRPClass,
    LinearLayerEpsWrapperFct,
    MaxPool2DWrapperFct,
    ReluWrapperFct,
    SigmoidWrapperFct,
    SumStacked2,
    bnafterconv_overwrite_intoconv,
    get_lrpwrapperformodule,
    resetbn,
)
from pixpnet.protonets.prp.resnet_features import BasicBlock, Bottleneck, ResNetFeatures


def imshow_im(hm, q=100):
    hm = hm.squeeze().sum(dim=0).detach()
    return hm


# partial replacement of BN, use own classes, no pretrained loading


class TorchModuleNotFoundError(Exception):
    pass


class BasicBlockFused(BasicBlock):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockFused, self).__init__(inplanes, planes, stride, downsample)

        # own
        self.elt = SumStacked2()  # eltwisesum2()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.elt(torch.stack([out, identity], dim=0))  # self.elt(out,identity)
        out = self.relu(out)

        return out


class BottleneckFused(Bottleneck):
    # Bottleneck in torchvision places the stride for downsampling at 3x3
    # convolution(self.conv2) while original implementation places the stride
    # at the first 1x1 convolution(self.conv1) according to "Deep residual
    # learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according
    # to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckFused, self).__init__(inplanes, planes, stride, downsample)

        # own
        self.elt = SumStacked2()  # eltwisesum2()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.elt(torch.stack([out, identity], dim=0))  # self.elt(out,identity)
        out = self.relu(out)

        return out


VERBOSE = False


class ResNetCanonized(ResNetFeatures):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNetCanonized, self).__init__(block, layers, num_classes=1000, zero_init_residual=False)

    # runs in your current module to find the object layer3.1.conv2, and
    # replaces it by the object stored in value
    # (see         success=iteratset(self,components,value) as initializer,
    #  can be modified to run in another class when replacing that self)
    def setbyname(self, name, value):
        def iteratset(obj, components, value):

            if not hasattr(obj, components[0]):
                return False
            elif len(components) == 1:
                setattr(obj, components[0], value)
                return True
            else:
                nextobj = getattr(obj, components[0])
                return iteratset(nextobj, components[1:], value)

        components = name.split(".")
        success = iteratset(self, components, value)
        if VERBOSE:
            print("success =", success, "name =", name, "obj = resnet", "value =", str(value)[:20])
        return success

    def copyfromresnet(self, net, lrp_params, lrp_layer2method):
        # --copy linear
        # --copy conv2, while fusing bns
        # --reset bn

        # first conv, then bn,
        # means: when encounter bn, find the conv before -- implementation
        #  dependent

        updated_layers_names = []

        last_src_module_name = None
        last_src_module = None

        for src_module_name, src_module in net.named_modules():
            if VERBOSE:
                print("at src_module_name", src_module_name)

            if src_module_name.startswith("module_dict."):
                src_module_name = src_module_name.split(".", 1)[1]

            if isinstance(src_module, nn.Linear):
                # copy linear layers
                if VERBOSE:
                    print("is Linear")
                wrapped = get_lrpwrapperformodule(copy.deepcopy(src_module), lrp_params, lrp_layer2method)
                if VERBOSE:
                    print(wrapped)
                if not self.setbyname(src_module_name, wrapped):
                    raise TorchModuleNotFoundError(
                        "could not find module " + src_module_name + " in target net to copy"
                    )
                updated_layers_names.append(src_module_name)
            # end of if

            if isinstance(src_module, nn.Conv2d):
                # store conv2d layers
                if VERBOSE:
                    print("is Conv2d")
                last_src_module_name = src_module_name
                last_src_module = src_module
            # end of if

            if isinstance(src_module, nn.BatchNorm2d):
                # conv-bn chain
                if VERBOSE:
                    print("is BatchNorm2d")

                if lrp_params["use_zbeta"] and (last_src_module_name == "conv1"):
                    thisis_inputconv_andiwant_zbeta = True
                else:
                    thisis_inputconv_andiwant_zbeta = False

                m = copy.deepcopy(last_src_module)
                m = bnafterconv_overwrite_intoconv(m, bn=src_module)
                # wrap conv
                wrapped = get_lrpwrapperformodule(
                    m, lrp_params, lrp_layer2method, thisis_inputconv_andiwant_zbeta=(thisis_inputconv_andiwant_zbeta)
                )
                if VERBOSE:
                    print(wrapped)

                if not self.setbyname(last_src_module_name, wrapped):
                    raise TorchModuleNotFoundError(
                        "could not find module " + last_src_module_name + " in target net to copy"
                    )

                updated_layers_names.append(last_src_module_name)

                # wrap batchnorm
                wrapped = get_lrpwrapperformodule(resetbn(src_module), lrp_params, lrp_layer2method)
                if VERBOSE:
                    print(wrapped)
                if not self.setbyname(src_module_name, wrapped):
                    raise TorchModuleNotFoundError(
                        "could not find module " + src_module_name + " in target net to copy"
                    )
                updated_layers_names.append(src_module_name)
            # end of if

            if VERBOSE:
                print("\n")

        # sum_stacked2 is present only in the targetclass, so must iterate here
        for target_module_name, target_module in self.named_modules():

            if isinstance(target_module, (nn.ReLU, nn.AdaptiveAvgPool2d, nn.MaxPool2d)):
                wrapped = get_lrpwrapperformodule(target_module, lrp_params, lrp_layer2method)
                if VERBOSE:
                    print(wrapped)
                if not self.setbyname(target_module_name, wrapped):
                    raise TorchModuleNotFoundError(
                        "could not find module " + src_module_name + " in target net to copy"
                    )
                updated_layers_names.append(target_module_name)

            if isinstance(target_module, SumStacked2):

                wrapped = get_lrpwrapperformodule(target_module, lrp_params, lrp_layer2method)
                if VERBOSE:
                    print(wrapped)
                if not self.setbyname(target_module_name, wrapped):
                    raise TorchModuleNotFoundError(
                        "could not find module " + target_module_name + " in target net , impossible!"
                    )
                updated_layers_names.append(target_module_name)

        to_delete = []
        for target_module_name, target_module in self.named_modules():
            if target_module_name not in updated_layers_names:
                if not (target_module_name.endswith(".module") or target_module_name.endswith(".downsample")):
                    if (
                        target_module_name
                        and "." not in target_module_name
                        and hasattr(net, "module_dict")
                        and not hasattr(net.module_dict, target_module_name)
                    ):
                        print("Replacing", target_module_name, "with identity")
                        to_delete.append(target_module_name)
                        setattr(self, target_module_name, nn.Identity())

                    elif target_module_name.split(".", 1)[0] in to_delete:
                        if VERBOSE:
                            print(target_module_name, "part of to_delete")
                    else:
                        print("not updated:", target_module_name)
            else:
                if VERBOSE:
                    print("updated:", target_module_name)


class AddonCanonized(nn.Module):
    def __init__(self, in_channels=512, out_channels=128):
        super(AddonCanonized, self).__init__()
        self.addon = nn.Sequential(
            OrderedDict(
                (
                    ("conv1", nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)),
                    ("relu1", nn.ReLU()),
                    ("conv_last", nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1)),
                    ("sigmoid", nn.Sigmoid()),
                )
            )
        )


def _addon_canonized(in_channels=512, out_channels=128, pretrained=False, progress=True, **kwargs):
    model = AddonCanonized(in_channels=in_channels, out_channels=out_channels)
    return model


def _resnet_canonized(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNetCanonized(block, layers, **kwargs)
    return model


def resnet18_canonized(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition"
    <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pretrained on ImageNet
        progress (bool): If True, displays a progress bar of the download to
            stderr
    """
    return _resnet_canonized("resnet18", BasicBlockFused, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet50_canonized(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition"
    <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pretrained on ImageNet
        progress (bool): If True, displays a progress bar of the download to
            stderr
    """
    return _resnet_canonized("resnet50", BottleneckFused, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet34_canonized(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition"
    <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pretrained on ImageNet
        progress (bool): If True, displays a progress bar of the download to
            stderr
    """
    return _resnet_canonized("resnet34", BasicBlockFused, [3, 4, 6, 3], **kwargs)


def resnet152_canonized(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition"
    <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pretrained on ImageNet
        progress (bool): If True, displays a progress bar of the download to
            stderr
    """
    return _resnet_canonized("resnet152", BottleneckFused, [3, 8, 36, 3], **kwargs)


def resnet101_canonized(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition"
    <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pretrained on ImageNet
        progress (bool): If True, displays a progress bar of the download to
            stderr
    """
    return _resnet_canonized("resnet101", BottleneckFused, [3, 4, 23, 3], **kwargs)


class SumLRP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)  # *values unpacks the list

        if VERBOSE:
            print("ctx.needs_input_grad", ctx.needs_input_grad)
            print("sum custom forward")
        return torch.sum(x, dim=(1, 2, 3))

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the
        loss with respect to the output, and we need to compute the gradient of
        the loss with respect to the input.
        """

        input_ = ctx.saved_tensors
        X = input_.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            Z = torch.sum(X, dim=(1, 2, 3))
        relevance_output_data = grad_output[0].clone().detach().unsqueeze(0)
        R = relevance_output_data * X / Z
        return R, None


def generate_prp_image(inputs, pno, model, config):
    model.train(False)
    inputs.requires_grad = True

    with torch.enable_grad():
        conv_features = model.conv_features(inputs)

        if config.model.distance == "cosine":
            new_dist = CosineDistLRPClass.apply
        else:
            new_dist = L2LRPClass.apply
        similarities = new_dist(conv_features, model)

        # global max pooling
        min_distances = model.max_layer(similarities)

        min_distances = min_distances.view(-1, model.num_prototypes)

        # For individual prototype
        (min_distances[:, pno]).backward()

    rel = inputs.grad.data
    prp = imshow_im(rel.to("cpu"))

    return prp


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = original_tuple + (path,)
        return tuple_with_path


def setbyname(obj, name, value):
    def iteratset(obj, components, value):

        if not hasattr(obj, components[0]):
            if VERBOSE:
                print(components[0])
            return False
        elif len(components) == 1:
            setattr(obj, components[0], value)
            return True
        else:
            nextobj = getattr(obj, components[0])
            return iteratset(nextobj, components[1:], value)

    components = name.split(".")
    success = iteratset(obj, components, value)
    print("success =", success, "name =", name, "obj =", str(obj)[:20], "value =", str(value)[:20])
    return success


base_architecture_to_features = {
    "resnet18": resnet18_canonized,
    "resnet34": resnet34_canonized,
    "resnet50": resnet50_canonized,
    "resnet101": resnet101_canonized,
    "resnet152": resnet152_canonized,
}


def prp_canonized_model(ppnet, config):
    device = ppnet.prototype_vectors.device
    base_arch = config.model.feature_extractor
    distance = config.model.distance

    model = base_architecture_to_features[base_arch](pretrained=False)
    model = model.to(device)

    lrp_params_def1 = {
        "conv2d_ignorebias": True,
        "eltwise_eps": 1e-6,
        "linear_eps": 1e-6,
        "pooling_eps": 1e-6,
        "use_zbeta": True,
    }

    lrp_layer2method = {
        "nn.ReLU": ReluWrapperFct,
        "nn.Sigmoid": SigmoidWrapperFct,
        "nn.BatchNorm2d": ReluWrapperFct,
        "nn.Conv2d": Conv2DBeta0WrapperFct,
        "nn.Linear": LinearLayerEpsWrapperFct,
        "nn.AdaptiveAvgPool2d": AdaptiveAvgPool2DWrapperFct,
        "nn.MaxPool2d": MaxPool2DWrapperFct,
        "sum_stacked2": EltwiseSumStacked2EpsWrapperFct,
    }

    model.copyfromresnet(ppnet.features, lrp_params=lrp_params_def1, lrp_layer2method=lrp_layer2method)
    model = model.to(device)
    ppnet.features = model

    if distance == "cosine":
        conv_layer1 = nn.Conv2d(ppnet.prototype_shape[1], ppnet.prototype_shape[0], kernel_size=1, bias=False).to(
            device
        )
        conv_layer1.weight.data = ppnet.prototype_vectors
    else:
        conv_layer1 = nn.Conv2d(ppnet.prototype_shape[1], ppnet.prototype_shape[0], kernel_size=1, bias=False).to(
            device
        )
        conv_layer1.weight.data = ppnet.ones

        wrapped = get_lrpwrapperformodule(copy.deepcopy(conv_layer1), lrp_params_def1, lrp_layer2method)
        conv_layer1 = wrapped

        conv_layer2 = nn.Conv2d(ppnet.prototype_shape[1], ppnet.prototype_shape[0], kernel_size=1, bias=False).to(
            device
        )
        conv_layer2.weight.data = ppnet.prototype_vectors

        wrapped = get_lrpwrapperformodule(copy.deepcopy(conv_layer2), lrp_params_def1, lrp_layer2method)
        conv_layer2 = wrapped

        relu_layer = nn.ReLU().to(device)
        wrapped = get_lrpwrapperformodule(copy.deepcopy(relu_layer), lrp_params_def1, lrp_layer2method)
        relu_layer = wrapped

    add_on_layers = _addon_canonized(
        in_channels=ppnet.add_on_layers.conv1.in_channels,
        out_channels=ppnet.prototype_dim,
    )

    for src_module_name, src_module in ppnet.add_on_layers.named_modules():
        if isinstance(src_module, nn.Conv2d):
            if VERBOSE:
                print(hasattr(add_on_layers.addon, src_module_name))
            wrapped = get_lrpwrapperformodule(copy.deepcopy(src_module), lrp_params_def1, lrp_layer2method)
            setbyname(add_on_layers.addon, src_module_name, wrapped)

        if isinstance(src_module, nn.ReLU):
            if VERBOSE:
                print(hasattr(add_on_layers.addon, src_module_name))
            wrapped = get_lrpwrapperformodule(copy.deepcopy(src_module), lrp_params_def1, lrp_layer2method)
            setbyname(add_on_layers.addon, src_module_name, wrapped)

        if isinstance(src_module, nn.Sigmoid):
            if VERBOSE:
                print(hasattr(add_on_layers.addon, src_module_name))
            wrapped = get_lrpwrapperformodule(copy.deepcopy(src_module), lrp_params_def1, lrp_layer2method)
            setbyname(add_on_layers.addon, src_module_name, wrapped)

    Hz = len(ppnet.rf_slices)
    Wz = len(ppnet.rf_slices[0])
    ppnet.max_layer = nn.MaxPool2d((Hz, Wz), return_indices=False)

    # Maxpool
    ppnet.max_layer = get_lrpwrapperformodule(copy.deepcopy(ppnet.max_layer), lrp_params_def1, lrp_layer2method)

    add_on_layers = add_on_layers.to(device)
    ppnet.add_on_layers = add_on_layers.addon

    if distance == "cosine":
        ppnet.conv_layer1 = conv_layer1
    else:
        ppnet.conv_layer1 = conv_layer1
        ppnet.conv_layer2 = conv_layer2
        ppnet.relu_layer = relu_layer

    return ppnet
