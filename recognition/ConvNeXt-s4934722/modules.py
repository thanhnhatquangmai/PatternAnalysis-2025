import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt Block (PyTorch implementation).

    This block is inspired by the ConvNeXt architecture, consisting of:
    1. Depthwise convolution (7x7) with padding to preserve spatial dimensions.
    2. Layer normalization applied to channels (after permuting to channels-last format).
    3. Two pointwise (1x1) linear layers with GELU activation in between.
    4. Optional layer scaling.
    5. Residual connection with optional stochastic depth.

    Args:
        input_dimension (int): Number of input channels.
        stochastic_depth_rate (float, optional):
            Drop path rate for stochastic depth. Default: 0.0
        layer_scale_initial_value (float, optional):
            Initial value for layer scaling parameter. Default: 1e-6
    """
    def __init__(
            self,
            input_dimension: int,
            stochastic_depth_rate=0.0,
            layer_scale_initial_value=1e-6
    ):
        super().__init__()
        hidden_dimension = input_dimension * 4  # Expand channels for pointwise layers

        # Depthwise convolution: separate filter per channel
        self.depthwise_convolution = nn.Conv2d(
            in_channels=input_dimension,
            out_channels=input_dimension,
            kernel_size=7,
            padding=3,  # keep spatial dimensions unchanged
            groups=input_dimension
        )

        # LayerNorm expects channels-last, so we permute input before applying
        self.normalization = nn.LayerNorm(normalized_shape=input_dimension, eps=1e-6)

        # First pointwise linear layer (1x1 conv equivalent)
        self.pointwise_convolution_1 = nn.Linear(
            in_features=input_dimension, out_features=hidden_dimension
        )

        # Activation
        self.activation_function = nn.GELU()

        # Second pointwise linear layer (project back to original channels)
        self.pointwise_convolution_2 = nn.Linear(
            in_features=hidden_dimension, out_features=input_dimension
        )

        # Optional layer scaling
        if layer_scale_initial_value > 0:
            self.layer_scale = nn.Parameter(
                data=layer_scale_initial_value * torch.ones((input_dimension), requires_grad=True)
            )
        else:
            self.layer_scale = None

        # Optional stochastic depth (drop path) for regularization
        if stochastic_depth_rate > 0:
            self.stochastic_depth = torchvision.ops.StochasticDepth(
                p=stochastic_depth_rate, mode="batch"
            )
        else:
            self.stochastic_depth = nn.Identity()

    def forward(self, x):
        """
        Forward pass for ConvNeXtBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (N, C, H, W)
        """
        input = x  # Save input for residual connection

        # Depthwise convolution
        x = self.depthwise_convolution(x)

        # Permute to channels-last for LayerNorm
        x = x.permute(0, 2, 3, 1)

        # Apply LayerNorm over channels
        x = self.normalization(x)

        # Pointwise projection + GELU
        x = self.pointwise_convolution_1(x)
        x = self.activation_function(x)
        x = self.pointwise_convolution_2(x)

        # Apply layer scaling if available
        if self.layer_scale is not None:
            x = self.layer_scale * x

        # Permute back to channels-first
        x = x.permute(0, 3, 1, 2)

        # Residual connection + optional stochastic depth
        x = input + self.stochastic_depth(x)
        return x

