# Image Processing Dataset Collection

A comprehensive repository containing common image processing datasets and denoising algorithms for research and benchmarking purposes.

## Overview

This repository contains two main components:

### 1. Image Datasets
A collection of standard benchmark datasets for image processing tasks including:
- **BSD68** - 68 grayscale natural images (512×512) for denoising evaluation
- **BSD100** - 100 natural images from Berkeley Segmentation Dataset for super-resolution and denoising
- **BSD500** - 500 color natural images with segmentation annotations
- **FMD** - Fluorescence Microscopy Denoising dataset with 12,000 real fluorescence microscopy images
- **Set12** - 12 classic test images (Lena, Barbara, House, etc.) for quick algorithm validation

### 2. Fluorescence Microscopy Denoising Methods
A comprehensive collection of state-of-the-art denoising algorithms for fluorescence microscopy images, implementing Variance Stabilizing Transform (VST) combined with various denoising techniques:

1. **VST-NLM** - Variance Stabilizing Transform with Non-Local Means
2. **VST-BM3D** - Variance Stabilizing Transform with Block-Matching 3D
3. **VST-KSVD** - Variance Stabilizing Transform with K-Singular Value Decomposition
4. **VST-KSVDD** - Variance Stabilizing Transform with KSVD Dictionary
5. **VST-KSVDG** - Variance Stabilizing Transform with KSVD Global
6. **VST-EPLL** - Variance Stabilizing Transform with Expected Patch Log Likelihood
7. **VST-WNNM** - Variance Stabilizing Transform with Weighted Nuclear Norm Minimization
8. **PURE-LET** - Poisson Unbiased Risk Estimator with Linear Expansions

## Repository Structure

```
dataset/
├── BSD68/             # BSD68 dataset images
├── BSD100/            # BSD100 dataset images  
├── BSD500/            # BSD500 dataset images
├── FMD/               # Fluorescence Microscopy Denoising dataset
├── Set12/             # Set12 classic test images
├── fluorescence_denoising_dataset/
│   ├── methods/       # Individual denoising algorithms
│   │   ├── VST_NLM/   # Non-Local Means implementation
│   │   ├── VST_BM3D/  # Block-Matching 3D implementation
│   │   ├── VST_KSVD/  # KSVD implementation
│   │   ├── VST_KSVDD/ # KSVD Dictionary implementation
│   │   ├── VST_KSVDG/ # KSVD Global implementation
│   │   ├── VST_EPLL/  # EPLL implementation
│   │   ├── VST_WNNM/  # WNNM implementation
│   │   └── PURE_LET/  # PURE-LET implementation
│   ├── utils/         # Common utility functions
│   ├── benchmarks/    # Benchmarking scripts
│   └── examples/      # Usage examples
```

## Fluorescence Denoising Methods

### Key Features

- **Variance Stabilizing Transform (VST)**: All methods implement VST to handle Poisson-Gaussian noise
- **Noise Estimation**: Automatic noise parameter estimation using `estimate_noise` function
- **Standardized Interface**: Each method follows the same function signature: `[img_denoise, time] = denoise_METHOD(img_raw)`
- **Comprehensive Utils**: Shared functions for VST transforms, noise estimation, and image processing

### Requirements

- MATLAB R2016b or later
- Image Processing Toolbox
- Signal Processing Toolbox (for some methods)

### Quick Start

```matlab
% Add paths
addpath(genpath('fluorescence_denoising_dataset/utils'));
addpath(genpath('fluorescence_denoising_dataset/methods'));

% Load your fluorescence image
img_raw = imread('your_fluorescence_image.tif');

% Denoise using VST-BM3D
[img_denoised, processing_time] = denoise_VST_BM3D(img_raw);

% Display results
figure;
subplot(1,2,1); imshow(img_raw, []); title('Raw Image');
subplot(1,2,2); imshow(img_denoised, []); title('Denoised Image');
```

### Method Details

#### VST-NLM
- **Algorithm**: Non-Local Means filtering after variance stabilization
- **Best for**: Images with repetitive structures and textures
- **Parameters**: Window size, search window size, Gaussian kernel sigma

#### VST-BM3D
- **Algorithm**: Block-matching and 3D collaborative filtering
- **Best for**: General-purpose denoising with good detail preservation
- **Parameters**: Hard/soft thresholding, block size, search area

#### VST-KSVD
- **Algorithm**: Dictionary learning using K-SVD
- **Best for**: Images with specific texture patterns
- **Parameters**: Dictionary size, sparsity level, number of iterations

#### PURE-LET
- **Algorithm**: Poisson Unbiased Risk Estimator with Linear Expansions
- **Best for**: Fast denoising with theoretical guarantees
- **Parameters**: Number of LET filters, threshold selection

## Dataset Details

### BSD68
- 68 grayscale natural images (512×512)
- Classic image denoising test set
- Used for PSNR/SSIM comparison under different noise levels

### BSD100  
- 100 natural images from Berkeley Segmentation Dataset
- Variable resolutions with rich texture details
- Used for comprehensive evaluation of super-resolution and denoising

### BSD500
- 500 color natural images
- Includes fine human segmentation annotations
- Suitable for edge detection, semantic segmentation, and multi-task training

### FMD
- Fluorescence Microscopy Denoising dataset
- 12,000 real fluorescence microscopy images
- Images acquired using commercial confocal, two-photon, and wide-field microscopes
- Includes cells, zebrafish, and mouse brain tissue samples
- 60,000 noisy images with different noise levels

### Set12
- 12 widely used classic test images
- Includes Lena, Barbara, House, etc.
- Smaller image sizes but diverse textures
- Used for quick algorithm validation and visualization

## Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{image_processing_dataset_collection,
  title={Image Processing Dataset Collection},
  author={Various},
  year={2024},
  publisher={GitHub},
  url={https://github.com/parcone/dataset}
}
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- New datasets
- New denoising methods
- Improvements to existing algorithms
- Additional benchmark datasets
- Documentation enhancements

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This repository consolidates implementations from various research papers and contributions from the image processing and fluorescence microscopy imaging communities.
