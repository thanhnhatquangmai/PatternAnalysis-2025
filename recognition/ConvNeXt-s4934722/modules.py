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
        input_channels (int): Number of input channels.
        stochastic_depth_rate (float, optional): Drop path rate for stochastic depth
        layer_scale_initial_value (float, optional): Initial value for layer scaling parameter
    """
    def __init__(self, input_channels, stochastic_depth_rate, layer_scale_initial_value):
        super().__init__()
        hidden_dimension = input_channels * 4  # Expand channels for pointwise layers

        # Depthwise convolution: separate filter per channel
        self.depthwise_convolution = nn.Conv2d(
            in_channels=input_channels,
            out_channels=input_channels,
            kernel_size=7,
            padding=3,  # keep spatial dimensions unchanged
            groups=input_channels
        )

        # Layer Normalization implemented by using GroupNorm with num_groups = 1
        self.layer_normalization = nn.GroupNorm(num_groups=1, num_channels=input_channels, eps=1e-6)

        # First pointwise linear layer (1x1 conv equivalent)
        self.pointwise_convolution_1 = nn.Conv2d(
            in_channels=input_channels, out_channels=hidden_dimension, kernel_size=1
        )

        # Activation
        self.activation_function = nn.GELU()

        # Second pointwise linear layer (project back to original channels)
        self.pointwise_convolution_2 = nn.Conv2d(
            in_channels=hidden_dimension, out_channels=input_channels, kernel_size=1
        )

        # Optional layer scaling
        if layer_scale_initial_value > 0:
            self.layer_scale = nn.Parameter(
                data=layer_scale_initial_value * torch.ones(
                    (input_channels), requires_grad=True
                )
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

        # Apply Layer Normalizaion over channels
        x = self.layer_normalization(x)

        # Pointwise projection + GELU
        x = self.pointwise_convolution_1(x)
        x = self.activation_function(x)
        x = self.pointwise_convolution_2(x)

        # Apply layer scaling if available
        if self.layer_scale is not None:
            x = self.layer_scale[:, None, None] * x

        # Residual connection + optional stochastic depth
        x = input + self.stochastic_depth(x)
        return x

class ConvNextStage(nn.Module):
    """
    ConvNeXt Stage

    A ConvNeXt stage consists of multiple consecutive ConvNeXt blocks operating
    at the same feature resolution. Each block includes depthwise convolution,
    layer normalization, pointwise MLP layers, and optional stochastic depth.

    Args:
        num_blocks (int): Number of ConvNeXt blocks in this stage.
        input_channels (int): Number of input channels for all blocks in this stage.
        stochastic_depth_rates (List[float]): A list of stochastic depth rates,
            one for each block.
        layer_scale_initial_value (float): Initial scaling factor for the layer scale parameter.
    """
    def __init__(
        self,
        num_blocks,
        input_channels,
        stochastic_depth_rates,
        layer_scale_initial_value
    ):
        super().__init__()

        # Sequentially stack multiple ConvNeXt blocks for this stage
        self.stage = nn.Sequential(*[
            ConvNeXtBlock(
                input_channels=input_channels,
                stochastic_depth_rate=stochastic_depth_rates[i],
                layer_scale_initial_value=layer_scale_initial_value
            )
            for i in range(num_blocks)
        ])

    def forward(self, x):
        """
        Forward pass through all ConvNeXt blocks in this stage.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W)
        Returns:
            torch.Tensor: Output tensor of shape (N, C, H, W)
        """
        return self.stage(x)



