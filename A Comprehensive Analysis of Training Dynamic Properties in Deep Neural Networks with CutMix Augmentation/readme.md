# A Comprehensive Analysis of Training Dynamic Properties in Deep Neural Networks


## Abstract

Data augmentation has significantly advanced deep learning, with techniques like CutMix improving accuracy by blending patches from different images. However, the reasons behind their training stability remain unclear. This study evaluates the effects of standard augmentation versus standard plus CutMix augmentation on training dynamic properties—loss landscape, effective β-smoothness, and gradient predictiveness. We trained ResNet-50 and ConvMixer models on CIFAR-10 and CIFAR-100 datasets to assess these properties and their impact on generalizability. Our results show distinct differences: CutMix not only boosts accuracy but also reduces gradient variability, enhancing stability. Loss landscape analysis reveals ConvMixer achieves lower training loss, while ResNet demonstrates stable training progression. These findings deepen our understanding of augmentation benefits and model selection, emphasizing the role of training dynamics in optimizing neural network performance.

## Introduction

Deep neural networks (DNNs) have transformed computer vision, yet their training remains complex without effective strategies like data augmentation. Standard techniques (e.g., cropping, flipping) and advanced methods like CutMix—which mixes image patches and labels—improve accuracy and robustness. This study explores how these augmentations influence training dynamics, focusing on loss landscape, effective β-smoothness, and gradient predictiveness. By examining these properties, we aim to clarify their role in stabilizing training and enhancing model generalizability.

## Methodology

### Models
- **ResNet-50**: A 50-layer convolutional neural network with skip connections to mitigate vanishing gradients.
- **ConvMixer**: A hybrid model blending convolutional layers and transformer components for enhanced feature extraction.

### Datasets
- **CIFAR-10**: 60,000 32x32 color images across 10 classes (50,000 training, 10,000 test).
- **CIFAR-100**: 60,000 32x32 color images across 100 classes (50,000 training, 10,000 test).

### Augmentation Techniques
- **Standard Augmentation**: Rotation (up to 15°), width/height shifts (10%), and horizontal flipping.
- **CutMix Augmentation**: Combines patches from two images with a 0.3:0.7 ratio, blending their labels accordingly.

## Experiments and Results

### Loss Landscape
We visualized training loss across 50 epochs for different learning rates (0.005, 0.001, 0.0005, 0.0001).

- **CIFAR-100**:
  - ConvMixer with standard augmentation converges quickly, but CutMix introduces instability with a steep loss change after 25 epochs.
  - ResNet shows a wider loss band but maintains stability with CutMix.

- **CIFAR-10**:
  - Both models exhibit stable training with CutMix; ConvMixer achieves lower loss than ResNet.


### Effective β-Smoothness & Gradient Predictiveness
Gradient analysis was conducted at the 5th epoch, 27th layer, and 30th batch across the four learning rates.

- **ConvMixer**:
  - Standard augmentation yields high gradient peaks (0.0025 for CIFAR-100, 0.0040 for CIFAR-10), indicating instability.
  - CutMix reduces peaks (0.0015 for CIFAR-100, 0.0020 for CIFAR-10) and stabilizes gradients.

- **ResNet**:
  - Standard augmentation shows high variability (peaks at 0.35); CutMix lowers this to 0.25 for CIFAR-10 but increases gradients for CIFAR-100.
  - CutMix enhances gradient predictability for CIFAR-10 but varies with CIFAR-100’s complexity.


## Discussion

- **Model Insights**: ConvMixer’s narrow loss band highlights its feature extraction strength, though it falters with complex tasks. ResNet’s broader, smoother loss landscape ensures stable training.
- **Augmentation Impact**: CutMix reduces gradient variability in most cases, enhancing stability and generalizability over standard augmentation.

These results emphasize the interplay between model architecture and augmentation strategies in optimizing training dynamics.

## Conclusion

Our study underscores the value of training dynamic properties in understanding DNN training. CutMix enhances stability and performance, with ConvMixer excelling in loss reduction and ResNet in training consistency. Future work should extend these analyses to larger datasets and diverse models.


## Publication

This work has been published on a conference, check here 

## Citation

Please cite our work as:  
Alam, Sraboni, et al. "A Comprehensive Analysis of Training Dynamic Properties in Deep Neural Networks with CutMix Augmentation." Proceedings of the 2024 7th International Conference on Big Data Technologies. 2024.
