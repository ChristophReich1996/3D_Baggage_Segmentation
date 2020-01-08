import torch.nn as nn


def get_activation(activation: str) -> nn.Sequential:
    """
    Method to return different types of activation functions
    :param activation: (str) Type of activation ('relu', 'leaky relu', 'elu', 'prlu', 'selu', 'sigmoid', 'identity')
    :return: (nn.Sequential) Activation function
    """
    assert activation in ['relu', 'leaky relu', 'elu', 'prlu', 'selu', 'sigmoid', 'identity'], \
        'Activation {} is not available!'.format(activation)
    if activation == 'relu':
        return nn.Sequential(nn.ReLU())
    elif activation == 'leaky relu':
        return nn.Sequential(nn.LeakyReLU())
    elif activation == 'elu':
        return nn.Sequential(nn.ELU())
    elif activation == 'prlu':
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
    Method to return different types of normalization operations
    :param normalization: (str) Type of normalization
    :param channels: (int) Number of channels to use
    :return: (nn.Sequential) Normalization operation
    """
    assert normalization in ['batchnorm', 'instancenorm'], \
        'Normalization {} is not avalible!'.format(normalization)
    if normalization == 'batchnorm':
        return nn.Sequential(nn.BatchNorm3d(channels))
    elif normalization == 'instancenorm':
        return nn.Sequential(nn.InstanceNorm3d(channels))
    else:
        raise RuntimeError('Normalization {} is not avalible!'.format(normalization))
