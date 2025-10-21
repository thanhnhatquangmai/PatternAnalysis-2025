"""
modules.py
Author: Thanh Nhat Quang Mai - 49347227
Date: 21st October 2025
Description:
    This file implements the ConvNeXt architecture in PyTorch, including:
    - ConvNeXtBlock: Core building block with depthwise convolution, normalization,
        pointwise layers, optional layer scaling, and stochastic depth.
    - ConvNextStage: A stage consisting of multiple ConvNeXtBlocks operating at the
        same feature resolution.
    - ConvNeXtDownsamplingLayer: Performs patch embedding or spatial downsampling
        between stages.
    - ConvNeXt: Complete hierarchical model for image classification, combining
        patchify stem, multiple feature extraction stages, and a final classification head.

    Key features:
    - Depthwise separable convolutions for computational efficiency.
    - Layer normalization for stable training.
    - Optional stochastic depth for regularization.
    - Scalable design supporting custom block and channel configurations.
"""

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
    
class ConvNeXtDownsamplingLayer(nn.Module):
    """
    Downsampling layer for ConvNeXt architecture.

    This layer reduces the spatial resolution (height and width) of the input feature map
    while increasing or changing the number of channels. It can either:
    - Act as a *patchify stem* (used in the first stage), converting the input image into 
        smaller non-overlapping patches, or
    - Perform standard downsampling between stages.

    Args:
        input_channels (int): Number of channels in the input feature map.
        output_channels (int): Number of channels to output after downsampling.
        has_patchify_stem (bool): 
            If True, uses a 4x4 convolution with stride 4 (patchify stem).
            If False, uses a 2x2 convolution with stride 2 (regular downsampling).
    """
    def __init__(self, input_channels, output_channels, has_patchify_stem=False):
        super().__init__()
        if has_patchify_stem:
            # Initial patch embedding stem
            # Converts raw image patches into feature representations
            self.downsampling_layer = nn.Sequential(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=4, # Non-overlapping 4x4 patches
                    stride=4
                ),
                nn.GroupNorm( # Layer Normalization
                    num_groups=1, num_channels=output_channels, eps=1e-6
                )
            )
        else:
            # Standard downsampling between ConvNeXt stages
            self.downsampling_layer = nn.Sequential(
                nn.GroupNorm( # Layer Normalization
                    num_groups=1, num_channels=input_channels, eps=1e-6
                ),
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=2, # Reduces spatial dimensions by half
                    stride=2
                )
            )

    def forward(self, x):
        """Apply downsampling to the input feature map."""
        return self.downsampling_layer(x)

class ConvNeXt(nn.Module):
    """
    ConvNeXt Model Implementation.

    This class builds a ConvNeXt architecture composed of multiple stages, 
    each consisting of ConvNeXt blocks. It performs patch embedding, 
    hierarchical feature extraction, and classification.

    Args:
        num_input_image_channels (int): Number of input image channels (e.g., 3 for RGB).
        num_classes (int): Number of output classes for classification.
        num_blocks_each_stage (list[int]): Number of ConvNeXt blocks in each stage.
        num_channels (list[int]): Channel dimensions for each stage.
        stochastic_depth_rate (float): Maximum drop path rate for stochastic depth regularization.
        layer_scale_initial_value (float): Initial scaling value for layer scale parameters.

    Attributes:
        downsampling_layers (nn.ModuleList): Sequential patchify + downsampling layers.
        stages (nn.ModuleList): Hierarchical ConvNeXt stages containing multiple ConvNeXtBlocks.
        avgpool (nn.AdaptiveAvgPool2d): Global average pooling layer.
        classifier (nn.Sequential): Flatten + linear layer for classification output.
    """
    def __init__(
        self,
        num_input_image_channels=3,
        num_classes=2,
        num_blocks_each_stage=[3, 3, 9, 3],
        num_channels=[96, 192, 384, 768],
        stochastic_depth_rate=0.1,
        layer_scale_initial_value=1e-6
    ):
        super().__init__()

        self.num_stages = len(num_blocks_each_stage)
        self.downsampling_layers = nn.ModuleList()

        # ---- Patchify stem ----
        # The first layer converts the image into non-overlapping patches using stride=4 convolution
        patchify_layer = ConvNeXtDownsamplingLayer(
            input_channels=num_input_image_channels,
            output_channels=num_channels[0],
            has_patchify_stem=True
        )
        self.downsampling_layers.append(patchify_layer)

        # ---- Downsampling layers ----
        # Subsequent layers reduce spatial resolution while increasing channel depth
        for i in range(len(num_channels) - 1):
            downsampling_layer = ConvNeXtDownsamplingLayer(
                input_channels=num_channels[i],
                output_channels=num_channels[i + 1]
            )
            self.downsampling_layers.append(downsampling_layer)
        
        # ---- Stochastic depth rates ----
        # Linearly increase the drop path rate across all blocks
        stochastic_depth_rates = [
            tensor.item() for tensor in torch.linspace(
                start=0,
                end=stochastic_depth_rate,
                steps=sum(num_blocks_each_stage)
            )
        ]

        # ---- ConvNeXt Stages ----
        # Each stage consists of multiple ConvNeXtBlocks operating at a specific resolution
        self.stages = nn.ModuleList()
        current_num_blocks = 0
        for i in range(self.num_stages):
            next_num_blocks = current_num_blocks + num_blocks_each_stage[i]
            stage = ConvNextStage(
                num_blocks=num_blocks_each_stage[i],
                input_channels=num_channels[i],
                stochastic_depth_rates=stochastic_depth_rates[
                    current_num_blocks : next_num_blocks
                ],
                layer_scale_initial_value=layer_scale_initial_value
            )
            self.stages.append(stage)
            current_num_blocks = next_num_blocks
        
        # ---- Classification head ----
        # Global average pooling followed by linear classifier
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=num_channels[-1], out_features=num_classes)
        )

    def forward(self, x):
        """
        Forward pass through the ConvNeXt model.
        Args:
            x (Tensor): Input tensor of shape (N, C, H, W)
        Returns:
            Tensor: Output logits of shape (N, num_classes)
        """
        # Downsample and process through ConvNeXt stages
        for i in range(self.num_stages):
            x = self.downsampling_layers[i](x)
            x = self.stages[i](x)

        # Global average pooling and classification
        x = self.avgpool(x)
        x = self.classifier(x)
        return x