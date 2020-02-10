import torch
import torch.nn as nn
import torch.nn.functional as F

import Misc


class VolumeEncoderBlock(nn.Module):
    """
    Basic Volume Encoder Block
    (convolution + normalization + activation) -> (convolution + normalization + activation) -> (downsampling)
    """

    def __init__(self, input_channels: int, output_channels: int, kernel_size: int = 3, stride: int = 1,
                 padding: int = 1, activation: str = 'prelu', downsampling: str = 'averagepool',
                 downsampling_factor: int = 2, normalization: str = 'batchnorm', dropout_rate: float = 0.0,
                 bias: bool = True) -> None:
        """
        Constructor method
        :param input_channels: (int) Number of input channels
        :param output_channels: (int) Number of output channels
        :param kernel_size: (int) Filter size of convolution
        :param stride: (int) Stride factor of convolution
        :param padding: (int) Padding used in every convolution
        :param activation: (str) Type of activation function
        :param downsampling: (str) Type of downsampling operation
        :param downsampling_factor: (int) Downsampling factor to use
        :param normalization: (str) Type of normalization operation used
        :param dropout_rate: (float) Dropout rate to perform after every stage
        :param bias: (bool) True to use bias in convolution operations
        """
        # Call super constructor
        super(VolumeEncoderBlock, self).__init__()
        # Save dropout rate
        self.dropout_rate = dropout_rate
        # Init activations
        self.activation_1 = Misc.get_activation(activation=activation)
        self.activation_2 = Misc.get_activation(activation=activation)
        # Init normalizations
        self.normalization_1 = Misc.get_normalization_3d(normalization=normalization, channels=output_channels)
        self.normalization_2 = Misc.get_normalization_3d(normalization=normalization, channels=output_channels)
        # Init convolutions
        self.convolution_1 = nn.Conv3d(in_channels=input_channels, out_channels=output_channels,
                                       kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.convolution_2 = nn.Conv3d(in_channels=output_channels, out_channels=output_channels,
                                       kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        # Init downsampling operation
        self.downsampling = Misc.get_downsampling_3d(downsampling=downsampling, factor=downsampling_factor,
                                                     channels=output_channels)

    def forward(self, input: torch.tensor) -> torch.tensor:
        """
        Forward pass of the basic volume decoder block
        :param input: (torch.tensor) Input volume with shape (batch size, channels_in, x_in, y_in, z_in)
        :return: (torch.tensor) Output tensor with shape (batch size, channels_out, x_out, y_out, z_out)
        """
        # First stage
        output_convolution_1 = self.convolution_1(input)
        output_normalization_1 = self.normalization_1(output_convolution_1)
        output_activation_1 = self.activation_1(output_normalization_1)
        if self.dropout_rate > 0.0:  # Perform dropout
            output_activation_1 = F.dropout(output_activation_1, p=self.dropout_rate)
        # Second stage
        output_convolution_2 = self.convolution_2(output_activation_1)
        output_normalization_2 = self.normalization_2(output_convolution_2)
        output_activation_2 = self.activation_2(output_normalization_2)
        if self.dropout_rate > 0.0:  # Perform dropout
            output_activation_2 = F.dropout(output_activation_2, p=self.dropout_rate)
        # Downsampling stage
        output_downsampling = self.downsampling(output_activation_2)
        return output_downsampling


class CoordinatesFullyConnectedBlock(nn.Module):
    """
    Implementation of a fully connected residual block for occupancy coordinates
    (linear + normalization + activation) -> (linear + normalization + activation) -> (residual mapping of input)
    """

    def __init__(self, input_channels: int, output_channels: int, activation: str = 'selu',
                 normalization: str = 'batchnorm', dropout_rate: float = 0.0, bias: bool = True) -> None:
        """
        Constructor method
        :param input_channels: (int) Number of input channels
        :param output_channels: (int) Number of output channels
        :param activation: (str) Type of activation function to use
        :param normalization: (str) Type of normalization operation to use
        :param dropout_rate: (float) Dropout rate to perform
        """
        # Call super constructor
        super(CoordinatesFullyConnectedBlock, self).__init__()
        # Save dropout rate
        self.dropout_rate = dropout_rate
        # Init activations
        self.activation_1 = Misc.get_activation(activation=activation)
        self.activation_2 = Misc.get_activation(activation=activation)
        # Init normalizations
        self.normalization_1 = Misc.get_normalization_1d(normalization=normalization, channels=output_channels,
                                                         channels_latent=183)
        self.normalization_2 = Misc.get_normalization_1d(normalization=normalization, channels=output_channels,
                                                         channels_latent=183)
        # Init linear operations
        self.linear_1 = nn.Linear(in_features=input_channels, out_features=output_channels, bias=bias)
        self.linear_2 = nn.Linear(in_features=output_channels, out_features=output_channels, bias=bias)

    def forward(self, input: torch.tensor, latent_tensor: torch.tensor = None) -> torch.tensor:
        """
        Forward pass of the fully connected block
        :param input: (torch.tensor) Input coordinates with shape (batch size, channels_in)
        :return: (torch.tensor) Output tensor with shape (batch size, channels_out)
        """
        # First stage
        output_linear_1 = self.linear_1(input)
        if isinstance(self.normalization_1, CBatchNorm1d):
            output_normalization_1 = self.normalization_1(output_linear_1, latent_tensor)
        else:
            output_normalization_1 = self.normalization_1(output_linear_1)
        output_activation_1 = self.activation_1(output_normalization_1)
        if self.dropout_rate > 0.0:  # Perform dropout
            output_activation_1 = F.dropout(output_activation_1, p=self.dropout_rate)
        # Second stage
        output_linear_2 = self.linear_2(output_activation_1)
        if isinstance(self.normalization_2, CBatchNorm1d):
            output_normalization_2 = self.normalization_2(output_linear_2, latent_tensor)
        else:
            output_normalization_2 = self.normalization_2(output_linear_2)
        output = self.activation_2(output_normalization_2)
        return output


class CBatchNorm1d(nn.Module):
    """
    Source: https://github.com/autonomousvision/occupancy_networks/blob/master/im2mesh/layers.py
    """

    def __init__(self, c_dim, f_dim):
        """
        Conditional batch normalization layer class
        :param c_dim: (int) dimension of latent conditioned code c
        :param f_dim: (int) feature dimension
        """
        super(CBatchNorm1d, self).__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        # Submodules
        self.conv_gamma = nn.Conv1d(c_dim, f_dim, 1)
        self.conv_beta = nn.Conv1d(c_dim, f_dim, 1)
        self.bn = nn.BatchNorm1d(f_dim, affine=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.conv_gamma.weight)
        nn.init.zeros_(self.conv_beta.weight)
        nn.init.ones_(self.conv_gamma.bias)
        nn.init.zeros_(self.conv_beta.bias)

    def forward(self, x, c):
        assert (x.size(0) == c.size(0))
        assert (c.size(1) == self.c_dim)

        # c is assumed to be of size batch_size x c_dim x T
        if len(c.size()) == 2:
            c = c.unsqueeze(2)

        # Affine mapping
        gamma = self.conv_gamma(c)
        beta = self.conv_beta(c)

        # Norm
        net = self.bn(x)
        out = gamma[:, :, 0] * net + beta[:, :, 0]

        return out
