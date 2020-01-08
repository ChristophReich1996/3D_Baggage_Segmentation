from typing import List, Union, Tuple

import torch.nn as nn


def get_activation(activation: str) -> nn.Sequential:
    """
    Method to return different types of activation functions
    :param activation: (str) Type of activation ('relu', 'leaky relu', 'elu', 'prlu', 'selu', 'sigmoid', 'identity')
    :return: (nn.Sequential) Activation function
    """
    assert activation in ['relu', 'leaky relu', 'elu', 'prelu', 'selu', 'sigmoid', 'identity'], \
        'Activation {} is not available!'.format(activation)
    if activation == 'relu':
        return nn.Sequential(nn.ReLU())
    elif activation == 'leaky relu':
        return nn.Sequential(nn.LeakyReLU())
    elif activation == 'elu':
        return nn.Sequential(nn.ELU())
    elif activation == 'prelu':
        return nn.Sequential(nn.PReLU())
    elif activation == 'selu':
        return nn.Sequential(nn.SELU())
    elif activation == 'sigmoid':
        return nn.Sequential(nn.Sigmoid())
    elif activation == 'identity':
        return nn.Sequential()
    else:
        raise RuntimeError('Activation {} is not available!'.format(activation))


def get_normalization_3d(normalization: str, channels: int) -> nn.Sequential():
    """
    Method to return different types of 3D normalization operations
    :param normalization: (str) Type of normalization ('batchnorm', 'instancenorm')
    :param channels: (int) Number of channels to use
    :return: (nn.Sequential) Normalization operation
    """
    assert normalization in ['batchnorm', 'instancenorm'], \
        'Normalization {} is not available!'.format(normalization)
    if normalization == 'batchnorm':
        return nn.Sequential(nn.BatchNorm3d(channels))
    elif normalization == 'instancenorm':
        return nn.Sequential(nn.InstanceNorm3d(channels))
    else:
        raise RuntimeError('Normalization {} is not available!'.format(normalization))


def get_normalization_1d(normalization: str, channels: int) -> nn.Sequential():
    """
    Method to return different types of 1D normalization operations
    :param normalization: (str) Type of normalization ('batchnorm', 'instancenorm')
    :param channels: (int) Number of channels to use
    :return: (nn.Sequential) Normalization operation
    """
    assert normalization in ['batchnorm', 'instancenorm'], \
        'Normalization {} is not available!'.format(normalization)
    if normalization == 'batchnorm':
        return nn.Sequential(nn.BatchNorm1d(channels))
    elif normalization == 'instancenorm':
        return nn.Sequential(nn.InstanceNorm1d(channels))
    else:
        raise RuntimeError('Normalization {} is not available!'.format(normalization))


def get_downsampling_3d(downsampling: str, factor: int = 2, channels: int = 0) -> nn.Sequential:
    """
    Method to return different types of 3D downsampling operations
    :param downsampling: (str) Type of donwsnapling ('maxpool', 'averagepool', 'convolution', 'none')
    :param factor: (int) Factor of downsampling
    :param channels: (int) Number of channels (only for convolution)
    :return: (nn.Sequential) Downsampling operation
    """
    assert downsampling in ['maxpool', 'averagepool', 'convolution', 'none'], \
        'Downsampling {} is not available'.format(downsampling)
    if downsampling == 'maxpool':
        return nn.Sequential(nn.MaxPool3d(kernel_size=factor, stride=factor))
    elif downsampling == 'averagepool':
        return nn.Sequential(nn.AvgPool3d(kernel_size=factor, stride=factor))
    elif downsampling == 'convolution':
        return nn.Sequential(
            nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=factor, stride=factor, padding=0,
                      bias=True))
    elif downsampling == 'none':
        return nn.Sequential()
    else:
        raise RuntimeError('Downsampling {} is not available'.format(downsampling))


def parse_to_list(
        possible_list: Union[int, float, bool, str, Tuple[int], List[Union[int, float, bool, str, Tuple[int]]]],
        length: int, name: str = '') -> List[Union[int, float, bool, str]]:
    """
    Function checks if parameter possible list is as list or a primitive data type.
    If a primitive data type is present a list with the desired length including the primitive variable in each
    element is returned.
    Examples:   possible_list=True, length=3            ->  [True, True, True]
                possible_list=[3, 4, 5, 6], length=4    ->  [3, 4, 5, 6]
    :param possible_list:
    :param length:
    :param name:
    :return:
    """
    if isinstance(possible_list, list):
        assert len(possible_list) == length, \
            'Length of {} list has to match with the number of blocks'.format(name)
        return possible_list
    else:
        return [possible_list] * length
