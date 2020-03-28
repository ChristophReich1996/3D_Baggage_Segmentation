import torch
import torch.nn as nn
import torch.nn.functional as F

import Misc


class VolumeEncoderBlock(nn.Module):
    """
    Basic Volume Residual Encoder Block
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
        # Init residual mapping
        if input_channels == output_channels:
            self.residual_mapping = nn.Identity()
        else:
            self.residual_mapping = nn.Conv3d(in_channels=input_channels, out_channels=output_channels,
                                              kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=bias)
        # Init downsampling operation
        self.downsampling = Misc.get_downsampling_3d(downsampling=downsampling, factor=downsampling_factor,
                                                     channels=output_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the basic volume decoder block
        :param input: (torch.tensor) Input volume with shape (batch size, channels_in, x_in, y_in, z_in)
        :return: (torch.tensor) Output tensor with shape (batch size, channels_out, x_out, y_out, z_out)
        """
        # First stage
        output = self.convolution_1(input)
        output = self.normalization_1(output)
        output = self.activation_1(output)
        if self.dropout_rate > 0.0:  # Perform dropout
            output = F.dropout(output, p=self.dropout_rate)
        # Second stage
        output = self.convolution_2(output)
        output = self.normalization_2(output)
        output = self.activation_2(output)
        if self.dropout_rate > 0.0:  # Perform dropout
            output = F.dropout(output, p=self.dropout_rate)
        output = output + self.residual_mapping(input)
        # Downsampling stage
        output = self.downsampling(output)
        return output


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
                                                         channels_latent=480)
        self.normalization_2 = Misc.get_normalization_1d(normalization=normalization, channels=output_channels,
                                                         channels_latent=480)
        # Init linear operations
        self.linear_1 = nn.Linear(in_features=input_channels, out_features=output_channels, bias=bias)
        self.linear_2 = nn.Linear(in_features=output_channels, out_features=output_channels, bias=bias)
        # Init residual operation
        if input_channels == output_channels:
            self.residual_mapping = nn.Identity()
        else:
            self.residual_mapping = nn.Linear(in_features=input_channels, out_features=output_channels, bias=bias)

    def forward(self, input: torch.Tensor, latent_tensor: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the fully connected residual block
        :param input: (torch.tensor) Input coordinates with shape (batch size, channels_in)
        :return: (torch.tensor) Output tensor with shape (batch size, channels_out)
        """
        # First stage
        # Linear layer
        output = self.linear_1(input)
        # Normalization
        if isinstance(self.normalization_1, ConditionalBatchNorm1d):
            output = self.normalization_1(output, latent_tensor)
        else:
            output = self.normalization_1(output)
        # Activation
        output = self.activation_1(output)
        # Perform dropout
        if self.dropout_rate > 0.0:
            output = F.dropout(output, p=self.dropout_rate)
        # Second stage
        # Linear layer
        output = self.linear_2(output)
        # Normalization
        if isinstance(self.normalization_2, ConditionalBatchNorm1d):
            output = self.normalization_2(output, latent_tensor)
        else:
            output = self.normalization_2(output)
        # Activation
        output = self.activation_2(output)
        # Perform dropout
        if self.dropout_rate > 0.0:
            output = F.dropout(output, p=self.dropout_rate)
        # Residual mapping
        output = output + self.residual_mapping(input)
        return output


class ConditionalBatchNorm1d(nn.Module):
    """
    Implementation of a conditional batch normalization module using linear operation to predict gamma and beta
    """

    def __init__(self, latent_channels: int, output_channels: int, bias: bool = True,
                 normalization: str = 'batchnorm') -> None:
        """
        Conditional batch normalization module including two 1D convolutions to predict gamma end beta
        :param latent_channels: (int) Features of the latent vector
        :param output_channels: (int) Features of the output vector to be normalized
        :param bias: (int) True if bias should be used in linear layer
        :param normalization: (str) Type of normalization to normalize feature tensor before using gamma and  beta
        """
        super(ConditionalBatchNorm1d, self).__init__()
        # Init operations
        self.linear_gamma = nn.Linear(in_features=latent_channels, out_features=output_channels, bias=bias)
        self.linear_beta = nn.Linear(in_features=latent_channels, out_features=output_channels, bias=bias)
        self.normalization = Misc.get_normalization_1d(normalization=normalization, channels=output_channels,
                                                       affine=False)  # affine=False -> gamma & beta not used
        # Reset parameters of convolutions
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Method resets the parameter of the convolution to predict gamma and beta
        """
        nn.init.zeros_(self.linear_gamma.weight)
        nn.init.zeros_(self.linear_beta.weight)
        nn.init.ones_(self.linear_gamma.bias)
        nn.init.zeros_(self.linear_beta.bias)

    def forward(self, input: torch.Tensor, latent_vector: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor to be normalized of shape (batch size coordinates, features)
        :param latent_vector: (torch.Tensor) Latent vector tensor of shape (batch_size, features)
        :return: (torch.Tensor) Normalized tensor
        """
        # Perform convolutions to estimate gamma and beta
        gamma = self.linear_gamma(latent_vector)
        beta = self.linear_beta(latent_vector)
        # Perform normalization
        output_normalized = self.normalization(input)
        # Repeat gamma and beta to apply factors to every coordinate
        gamma = torch.repeat_interleave(gamma, int(output_normalized.shape[0] / gamma.shape[0]), dim=0)
        beta = torch.repeat_interleave(beta, int(output_normalized.shape[0] / beta.shape[0]), dim=0)
        # Add factors
        output = gamma * output_normalized + beta
        return output


class InstanceNorm1d(nn.Module):
    """
    Implementation of instance normalization for a 2D tensor of shape (batch size, features)
    """

    def __init__(self) -> None:
        # Call super constructor
        super(InstanceNorm1d, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return (input - input.mean(dim=1, keepdim=True)) / input.std(dim=1, keepdim=True)