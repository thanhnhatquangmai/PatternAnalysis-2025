# Alzheimer's Disease Classification using ConvNeXt on the ADNI dataset

## Overview

This project goal is to classify between Alzheimer's Disease (AD) and Normal Control (NC) images in the Alzheimer's Disease Neuroimaging Initiative (ADNI) brain dataset [[1]](#adni-link). Early Alzheimer's detection helps patients take control of their conditions, gain access to necessary support and resources, and make informed plans for the future. ConvNeXt, which is one the latest vision model, is used in this classification problem. Using ConvNeXt architecture, the model reached 80.4% accuracy on the ADNI test dataset.

## Model Architecture

The ConvNeXt architecture is a pure ConvNet that is modernized from a standard ResNet toward the design of a vision Transformer. While being much simpler in design, ConvNeXts are reported to achieve the same level of accuracy and scalability as Transformers [[2]](#convnext). The ConvNeXt block architecture is shown below.

![ConvNeXt Block Architecture](images/report/ConvNeXt-Block.png)

*Figure 1. ConvNeXt block architecture, adapted from Liu et al. (2022)* [[2]](#convnext)*.*

The ConvNeXt architecture comprises a serires of stage which consists of multiple consecutive ConvNeXt blocks opearting at the same feature resolution. Each block includes depthwise convolution, layer normalization, pointwise convolution, layer scaling and residual connection with stochastic depth. 

## Dataset Description

## Training Process

## Results

## Usage

## References

<a id="adni-link"></a>[1] Alzheimer's Disease Neuroimaging Initiative (ADNI). [https://adni.loni.usc.edu/](https://adni.loni.usc.edu/)

<a id="convnext"></a>[2] Liu, Z., Mao, H., Wu, C. Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). *A ConvNet for the 2020s*. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 11976–11986). [https://arxiv.org/abs/2201.03545](https://arxiv.org/abs/2201.03545)





