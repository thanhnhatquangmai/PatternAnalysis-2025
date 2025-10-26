# Alzheimer's Disease Classification using ConvNeXt on the ADNI dataset

## Overview

This project goal is to classify between Alzheimer's Disease (AD) and Normal Control (NC) images in the Alzheimer's Disease Neuroimaging Initiative (ADNI) brain dataset [[1]](#adni-link). Early Alzheimer's detection helps patients take control of their conditions, gain access to necessary support and resources, and make informed plans for the future. ConvNeXt, which is one the latest vision model, is used in this classification problem. Using ConvNeXt architecture, the model reached 80.4% accuracy on the ADNI test dataset.

## Model Architecture

The ConvNeXt architecture is a pure ConvNet that is modernized from a standard ResNet toward the design of a vision Transformer. While being much simpler in design, ConvNeXts are reported to achieve the same level of accuracy and scalability as Transformers [[2]](#convnext). The ConvNeXt block architecture is shown below.

<a id="convnext-block"></a>

![ConvNeXt Block Architecture](images/report/ConvNeXt-Block.png)

*Figure 1. ConvNeXt block architecture, adapted from Liu et al. (2022)* [[2]](#convnext)*.*

The ConvNeXt architecture comprises a series of stages which consists of multiple consecutive ConvNeXt blocks opearting at the same feature resolution. Each block includes depthwise convolution, layer normalization, pointwise convolution, layer scaling and residual connection with stochastic depth. As a ConvNet, this model has several built-in inductive biases that make it well-suited for a wide range of computer vision tasks such as classification of MRI images of the brain. It also proves to be efficient as computations are shared when used in a sliding-window manner [[2]](#convnext).

The ConvNeXt architeture consists of the following main innovations:

### Stage Compute Ratio

ConvNext has 4 stages and the number of blocks each stage is changed from (3, 4, 6, 3) in ResNet-50 to (3, 3, 9, 3).

### "Patchify" Stem

A simple "Patchify" stem (4 $\times$ 4 non-overlapping convolution) are used in this model to downsample input images to mimic the design of ViT to downsample the input images.

### ResNeXt Design Employment

A combination of depthwise convolution and 1 $\times$ 1 convolution is used that imitates the self-attention mechanism in Transformers. Network width increases to 96 channels. Channel-mixing design allows model to combine local features of images, especially medical images like in ADNI MRI dataset between different brain structures. Thus, the model can preserve both local and global pattern leading to better diagnostic.

### Inverted Bottleneck

A convolution with kernel size of 7 $\times$ 7 is used in each block that allows the model to focus on local regions. This enhances the model's ability to capture complex spatial patterns in images by expanding the channels.

### Activation Functions

GELU activation is used in each block. The GELU layers are eliminated from residual block except for one between two 1 $\times$ 1 layers as seen in [Figure 1](#convnext-block). Fewer activation functions per layer helps to increase gradient flow and better preserve the information across layers.

### Normalization Layer

One Layer Normalization layer is used in each block to improve the convergence and reduce overfitting. Layer Normalization is used instead of Batch Normalization as it might have some negative effects on model's performance [[3]](#batchnorm).

### Downsampling Layer

Separate downsampling layers are added between each stages where each of them is a 2 $\times$ 2 convolution layer with stride 2 for spatial downsampling. They reduce the spatial dimension while increasing the number of feature channels which allows the model to concentrate on high-level patterns.

## Dataset Description

## Training Process

## Results

## Usage

## References

<a id="adni-link"></a>[1] Alzheimer's Disease Neuroimaging Initiative (ADNI). [https://adni.loni.usc.edu/](https://adni.loni.usc.edu/)

<a id="convnext"></a>[2] Liu, Z., Mao, H., Wu, C. Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). *A ConvNet for the 2020s*. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 11976–11986). [https://arxiv.org/abs/2201.03545](https://arxiv.org/abs/2201.03545)

<a id="batchnorm"></a>[3] Wu, Y., & Johnson, J. (2021). *Rethinking "Batch" in BatchNorm*. arXiv preprint [arXiv:2105.07576](https://arxiv.org/abs/2105.07576).






