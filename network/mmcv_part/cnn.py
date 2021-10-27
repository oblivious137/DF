import torch.nn as nn

CONV_LAYERS=dict(
    Conv1d = nn.Conv1d,
    Conv2d = nn.Conv2d,
    Conv3d = nn.Conv3d,
    Conv = nn.Conv2d,
)

def build_conv_layer(cfg, *args, **kwargs):
    """Build convolution layer.
    Args:
        cfg (None or dict): The conv layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an conv layer.
        args (argument list): Arguments passed to the `__init__`
            method of the corresponding conv layer.
        kwargs (keyword arguments): Keyword arguments passed to the `__init__`
            method of the corresponding conv layer.
    Returns:
        nn.Module: Created conv layer.
    """
    if cfg is None:
        cfg_ = dict(type='Conv2d')
    else:
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        if 'type' not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in CONV_LAYERS:
        raise KeyError(f'Unrecognized norm type {layer_type}')
    else:
        conv_layer = CONV_LAYERS.get(layer_type)

    layer = conv_layer(*args, **kwargs, **cfg_)

    return layer



# Copyright (c) OpenMMLab. All rights reserved.
import inspect

def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.
    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.
    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True

def is_tuple_of(seq, expected_type):
    """Check whether it is a tuple of some type.
    A partial method of :func:`is_seq_of`.
    """
    return is_seq_of(seq, expected_type, seq_type=tuple)

    
from functools import partial

import torch

TORCH_VERSION = torch.__version__


def is_rocm_pytorch() -> bool:
    is_rocm = False
    if TORCH_VERSION != 'parrots':
        try:
            from torch.utils.cpp_extension import ROCM_HOME
            is_rocm = True if ((torch.version.hip is not None) and
                               (ROCM_HOME is not None)) else False
        except ImportError:
            pass
    return is_rocm


def _get_cuda_home():
    if TORCH_VERSION == 'parrots':
        from parrots.utils.build_extension import CUDA_HOME
    else:
        if is_rocm_pytorch():
            from torch.utils.cpp_extension import ROCM_HOME
            CUDA_HOME = ROCM_HOME
        else:
            from torch.utils.cpp_extension import CUDA_HOME
    return CUDA_HOME


def get_build_config():
    if TORCH_VERSION == 'parrots':
        from parrots.config import get_build_info
        return get_build_info()
    else:
        return torch.__config__.show()


def _get_conv():
    if TORCH_VERSION == 'parrots':
        from parrots.nn.modules.conv import _ConvNd, _ConvTransposeMixin
    else:
        from torch.nn.modules.conv import _ConvNd, _ConvTransposeMixin
    return _ConvNd, _ConvTransposeMixin


def _get_dataloader():
    if TORCH_VERSION == 'parrots':
        from torch.utils.data import DataLoader, PoolDataLoader
    else:
        from torch.utils.data import DataLoader
        PoolDataLoader = DataLoader
    return DataLoader, PoolDataLoader


def _get_extension():
    if TORCH_VERSION == 'parrots':
        from parrots.utils.build_extension import BuildExtension, Extension
        CppExtension = partial(Extension, cuda=False)
        CUDAExtension = partial(Extension, cuda=True)
    else:
        from torch.utils.cpp_extension import (BuildExtension, CppExtension,
                                               CUDAExtension)
    return BuildExtension, CppExtension, CUDAExtension


def _get_pool():
    if TORCH_VERSION == 'parrots':
        from parrots.nn.modules.pool import (_AdaptiveAvgPoolNd,
                                             _AdaptiveMaxPoolNd, _AvgPoolNd,
                                             _MaxPoolNd)
    else:
        from torch.nn.modules.pooling import (_AdaptiveAvgPoolNd,
                                              _AdaptiveMaxPoolNd, _AvgPoolNd,
                                              _MaxPoolNd)
    return _AdaptiveAvgPoolNd, _AdaptiveMaxPoolNd, _AvgPoolNd, _MaxPoolNd


def _get_norm():
    if TORCH_VERSION == 'parrots':
        from parrots.nn.modules.batchnorm import _BatchNorm, _InstanceNorm
        SyncBatchNorm_ = torch.nn.SyncBatchNorm2d
    else:
        from torch.nn.modules.instancenorm import _InstanceNorm
        from torch.nn.modules.batchnorm import _BatchNorm
        SyncBatchNorm_ = torch.nn.SyncBatchNorm
    return _BatchNorm, _InstanceNorm, SyncBatchNorm_


_ConvNd, _ConvTransposeMixin = _get_conv()
DataLoader, PoolDataLoader = _get_dataloader()
BuildExtension, CppExtension, CUDAExtension = _get_extension()
_BatchNorm, _InstanceNorm, SyncBatchNorm_ = _get_norm()
_AdaptiveAvgPoolNd, _AdaptiveMaxPoolNd, _AvgPoolNd, _MaxPoolNd = _get_pool()


class SyncBatchNorm(SyncBatchNorm_):

    def _check_input_dim(self, input):
        if TORCH_VERSION == 'parrots':
            if input.dim() < 2:
                raise ValueError(
                    f'expected at least 2D input (got {input.dim()}D input)')
        else:
            super()._check_input_dim(input)


NORM_LAYERS = {
    'BN': nn.BatchNorm2d,
    'BN1d': nn.BatchNorm1d,
    'BN2d': nn.BatchNorm2d,
    'BN3d': nn.BatchNorm3d,
    'SyncBN': SyncBatchNorm,
    'GN': nn.GroupNorm,
    'IN': nn.InstanceNorm2d,
    'IN1d': nn.InstanceNorm1d,
    'IN2d': nn.InstanceNorm2d,
    'IN3d': nn.InstanceNorm3d,
}


def infer_abbr(class_type):
    """Infer abbreviation from the class name.
    When we build a norm layer with `build_norm_layer()`, we want to preserve
    the norm type in variable names, e.g, self.bn1, self.gn. This method will
    infer the abbreviation to map class types to abbreviations.
    Rule 1: If the class has the property "_abbr_", return the property.
    Rule 2: If the parent class is _BatchNorm, GroupNorm, LayerNorm or
    InstanceNorm, the abbreviation of this layer will be "bn", "gn", "ln" and
    "in" respectively.
    Rule 3: If the class name contains "batch", "group", "layer" or "instance",
    the abbreviation of this layer will be "bn", "gn", "ln" and "in"
    respectively.
    Rule 4: Otherwise, the abbreviation falls back to "norm".
    Args:
        class_type (type): The norm layer type.
    Returns:
        str: The inferred abbreviation.
    """
    if not inspect.isclass(class_type):
        raise TypeError(
            f'class_type must be a type, but got {type(class_type)}')
    if hasattr(class_type, '_abbr_'):
        return class_type._abbr_
    if issubclass(class_type, _InstanceNorm):  # IN is a subclass of BN
        return 'in'
    elif issubclass(class_type, _BatchNorm):
        return 'bn'
    elif issubclass(class_type, nn.GroupNorm):
        return 'gn'
    elif issubclass(class_type, nn.LayerNorm):
        return 'ln'
    else:
        class_name = class_type.__name__.lower()
        if 'batch' in class_name:
            return 'bn'
        elif 'group' in class_name:
            return 'gn'
        elif 'layer' in class_name:
            return 'ln'
        elif 'instance' in class_name:
            return 'in'
        else:
            return 'norm_layer'


def build_norm_layer(cfg, num_features, postfix=''):
    """Build normalization layer.
    Args:
        cfg (dict): The norm layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate a norm layer.
            - requires_grad (bool, optional): Whether stop gradient updates.
        num_features (int): Number of input channels.
        postfix (int | str): The postfix to be appended into norm abbreviation
            to create named layer.
    Returns:
        (str, nn.Module): The first element is the layer name consisting of
            abbreviation and postfix, e.g., bn1, gn. The second element is the
            created norm layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in NORM_LAYERS:
        raise KeyError(f'Unrecognized norm type {layer_type}')

    norm_layer = NORM_LAYERS.get(layer_type)
    abbr = infer_abbr(norm_layer)

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)
    if layer_type != 'GN':
        layer = norm_layer(num_features, **cfg_)
        if layer_type == 'SyncBN' and hasattr(layer, '_specify_ddp_gpu_num'):
            layer._specify_ddp_gpu_num(1)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer


def is_norm(layer, exclude=None):
    """Check if a layer is a normalization layer.
    Args:
        layer (nn.Module): The layer to be checked.
        exclude (type | tuple[type]): Types to be excluded.
    Returns:
        bool: Whether the layer is a norm layer.
    """
    if exclude is not None:
        if not isinstance(exclude, tuple):
            exclude = (exclude, )
        if not is_tuple_of(exclude, type):
            raise TypeError(
                f'"exclude" must be either None or type or a tuple of types, '
                f'but got {type(exclude)}: {exclude}')

    if exclude and isinstance(layer, exclude):
        return False

    all_norm_bases = (_BatchNorm, _InstanceNorm, nn.GroupNorm, nn.LayerNorm)
    return isinstance(layer, all_norm_bases)