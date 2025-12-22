# Poisson-Gaussian Fluorescence Microscopy Denoising Dataset 

Citation:
```
@article{zhang2018poisson,
  title={A Poisson-Gaussian Denoising Dataset with Real Fluorescence Microscopy Images},
  author={Zhang, Yide and Zhu, Yinhao and Nichols, Evan and Wang, Qingfei and Zhang, Siyuan and Smith, Cody and Howard, Scott},
  journal={arXiv preprint arXiv:1812.10366},
  year={2018}
}
```
To appear in CVPR 2019.

Confocal Microscopy Dataset:
 - "Confocal_BPAE_B"
 - "Confocal_BPAE_G"
 - "Confocal_BPAE_R" 
 - "Confocal_MICE"
 - "Confocal_FISH"
--------------------------------------------------------
Two-Photon Microscopy Dataset:
 - "TwoPhoton_BPAE_B"
 - "TwoPhoton_BPAE_G"
 - "TwoPhoton_BPAE_R"
 - "TwoPhoton_MICE"
--------------------------------------------------------
Wide-Field Microscopy Dataset:
 - "WideField_BPAE_B"
 - "WideField_BPAE_G"
 - "WideField_BPAE_R"
--------------------------------------------------------
Mixed Dataset:
 - "test_mix":


Definitions:
 * "BPAE": fixed bovine pulmonary artery endothelial (BPAE) cells; the images have three channels as defined below.
 * "BPAE_B": the blue channel of the BPAE cells, showing the nuclei labeled with DAPI.
 * "BPAE_G": the green channel of the BPAE cells, showing the F-actin labeled with Alexa Fluor 488 phalloidin.
 * "BPAE_R": the red channel of the BPAE cells, showing the mitochondria labeled with MitoTracker Red CMXRos.
 * "MICE": fixed mouse brain tissues stained with DAPI and cleared; the images have a single channel.
 * "FISH": fixed zebrafish embryos (EGFP labeled Tg(sox10:megfp) zebrafish at 2 days post fertilization); the images have a single channel.
 * "test_mix": a mixed dataset containing images from all other datasets.


# File structure
```
data_root/type/noise_level/fov/capture.png
```

- *type* (str): one of 12 types: ['TwoPhoton_BPAE_R', 'TwoPhoton_BPAE_G', 
    'TwoPhoton_BPAE_B', 'TwoPhoton_MICE', 'Confocal_MICE', 'Confocal_BPAE_R',
    'Confocal_BPAE_G', 'Confocal_BPAE_B', 'Confocal_FISH', 'WideField_BPAE_R', 
    'WideField_BPAE_G', 'WideField_BPAE_B']
- *noise_level*: one of 5 noise levels: ['raw', 'avg2', 'avg4', 'avg8', avg16', 'gt']
- *fov* (int): field-of-view for each type, i.e. 1, 2, ..., 20
- *capture.png*: image filename

# API for dataset

## Dataloader for training

```python
def fluore_to_tensor(pic)
```
Convert a ``PIL Image`` to ``torch.Tensor``. Range stays the same. Only output one channel, if RGB, convert to grayscale as well. Currently data is 8 bit depth.
    

```python
def load_denoising(train, batch_size, noise_levels, types=None, captures=2, patch_size=256, transform=None, target_transform=None,loader=pil_loader, test_fov=19)
```

```python
def load_denoising_n2n_train(root, batch_size, noise_levels, types=None,
    patch_size=256, transform=None, target_transform=None, loader=pil_loader,
    test_fov=19)
```

```python
def load_denoising_test_mix(root, batch_size, noise_levels, loader=pil_loader, 
    transform=None, target_transform=None, patch_size=256)
```


## Example usage

```python
from torchvision import transforms
from data_loader import load_denoising, load_denoising_test_mix, load_denoising_n2n_train
root = 'path/to/dataset'
train = True
noise_levels = [1, 2, 4, 8, 16]
test_noise_levels = [1, 2, 4]
types = ['TwoPhoton_MICE', 'Confocal_BPAE_B']
# types = None
captures = 50
patch_size = 128
batch_size = 16
test_batch_size = 4
transform = 'center_crop'
target_transform = None
test_fov = 19

if transform == 'five_crop':
    # wide field images may have complete noise in center-crop case
    transform = transforms.Compose([
        transforms.FiveCrop(args.imsize),
        transforms.Lambda(lambda crops: torch.stack([
            fluore_to_tensor(crop) for crop in crops])),
        transforms.Lambda(lambda x: x.float().div(255).sub(0.5))
        ])
elif transform == 'center_crop':
    # default transform
    transform = None


train_loader = load_denoising(root, train, batch_size, noise_levels=noise_levels,
    types=types, captures=captures, patch_size=patch_size, transform=transform, 
    target_transform=target_transform, test_fov=test_fov)

test_loader = load_denoising_test_mix(root, 
    batch_size=test_batch_size, noise_levels=test_noise_levels, 
    transform=transform, patch_size=patch_size)

train_loader_n2n = load_denoising_n2n_train(root, batch_size, noise_levels,
    patch_size=patch_size, transform=transform, target_transform=transform, 
    test_fov=19)

for batch_size, (noisy, clean) in enumerate(train_loader):
    print(noisy.shape)
    print(clean.shape)
    break

for batch_size, (noisy, clean) in enumerate(test_loader):
    print(noisy.shape)
    print(clean.shape)
    break

for batch_size, (noisy_input, noisy_target, clean) in enumerate(train_loader_n2n):
    print(noisy_input.shape)
    print(noisy_target.shape)
    print(clean.shape)
    break
```


## DataFolder

```python
class DenoisingFolder(root, train, noise_levels, types=None,        test_fov=19, captures=2, transform=None,              target_transform=None, loader=pil_loader)
```
Subclass of ``torch.utils.data.Dataset``, select subset of training/test dataset, with certain ``noise_levels``, ``types``, and number of ``captures`` within each field of view (FOV), where if test, using only ``test_fov``, otherwise using all other fovs except ``test_fov`` for training.

``__get_item__(index)`` returns the `index`-th data pair - ``(noisy, clean)``, where ``clean`` is the ground truth obtained with averaging 50 noisy captures of the same FOV.


```python
class DenoisingFolderN2N(root, noise_levels, types=None, test_fov=19, captures=50, transform=None, target_transform=None, loader=pil_loader)
```
Similiar to ``DenoisingFolder``, except ``__get_item__(index)`` returns ``index``-th noisy capture pairs ``(noisy1, noisy2)`` randomly selected during loading from the same FOV. The number of ``captures`` is default to be 50.


```python
class DenoisingTestMixFolder(root, loader, noise_levels, transform, target_transform)
```
Class for mixed test set ``test_mix`` which includes all types of images in 19-th FOV by default. The file struture is 
```
data_root/test_mix/noise_level/capture.png
```




