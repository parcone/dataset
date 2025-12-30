# Fluorescence Microscopy Denoising Dataset

A comprehensive collection of state-of-the-art denoising algorithms for fluorescence microscopy images, implementing Variance Stabilizing Transform (VST) combined with various denoising techniques.

## Overview

This dataset contains 8 different denoising methods specifically designed for fluorescence microscopy images affected by Poisson-Gaussian noise:

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
fluorescence_denoising_dataset/
├── methods/           # Individual denoising algorithms
│   ├── VST_NLM/       # Non-Local Means implementation
│   ├── VST_BM3D/      # Block-Matching 3D implementation
│   ├── VST_KSVD/      # KSVD implementation
│   ├── VST_KSVDD/     # KSVD Dictionary implementation
│   ├── VST_KSVDG/     # KSVD Global implementation
│   ├── VST_EPLL/      # EPLL implementation
│   ├── VST_WNNM/      # WNNM implementation
│   └── PURE_LET/      # PURE-LET implementation
├── utils/             # Common utility functions
├── benchmarks/        # Benchmarking scripts
└── examples/          # Usage examples
```

## Key Features

- **Variance Stabilizing Transform (VST)**: All methods implement VST to handle Poisson-Gaussian noise
- **Noise Estimation**: Automatic noise parameter estimation using `estimate_noise` function
- **Standardized Interface**: Each method follows the same function signature: `[img_denoise, time] = denoise_METHOD(img_raw)`
- **Comprehensive Utils**: Shared functions for VST transforms, noise estimation, and image processing

## Requirements

- MATLAB R2016b or later
- Image Processing Toolbox
- Signal Processing Toolbox (for some methods)

## Quick Start

```matlab
% Load your fluorescence image
img_raw = imread('your_fluorescence_image.tif');

% Denoise using VST-BM3D
[img_denoised, processing_time] = denoise_VST_BM3D(img_raw);

% Display results
figure;
subplot(1,2,1); imshow(img_raw, []); title('Raw Image');
subplot(1,2,2); imshow(img_denoised, []); title('Denoised Image');
```

## Method Details

### VST-NLM
- **Algorithm**: Non-Local Means filtering after variance stabilization
- **Best for**: Images with repetitive structures and textures
- **Parameters**: Window size, search window size, Gaussian kernel sigma

### VST-BM3D
- **Algorithm**: Block-matching and 3D collaborative filtering
- **Best for**: General-purpose denoising with good detail preservation
- **Parameters**: Hard/soft thresholding, block size, search area

### VST-KSVD
- **Algorithm**: Dictionary learning using K-SVD
- **Best for**: Images with specific texture patterns
- **Parameters**: Dictionary size, sparsity level, number of iterations

### PURE-LET
- **Algorithm**: Poisson Unbiased Risk Estimator with Linear Expansions
- **Best for**: Fast denoising with theoretical guarantees
- **Parameters**: Number of LET filters, threshold selection

## Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{fluorescence_denoising_dataset,
  title={Fluorescence Microscopy Denoising Dataset},
  author={Various},
  year={2024},
  publisher={GitHub},
  url={https://github.com/parcone/dataset}
}
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- New denoising methods
- Improvements to existing algorithms
- Additional benchmark datasets
- Documentation enhancements

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This dataset consolidates implementations from various research papers and contributions from the fluorescence microscopy imaging community.
