# Iterative Visual Reasoning Beyond Convolutions
By Xinlei Chen, Li-Jia Li, Li Fei-Fei and Abhinav Gupta. 

### Disclaimer
  - This is the authors' implementation of the system described in the paper, not an official Google product.

### Prerequisites

  - Tensorflow, tested with version 1.6 with Ubuntu 16.04, installed with:
  ```Shell
  pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.6.0-cp27-none-linux_x86_64.whl
  ```

  - Other packages needed can be installed with `pip`:
  ```Shell
  pip install Cython easydict matplotlib opencv-python Pillow pyyaml scipy
  ```

### Installation

  - Clone the repository.
  ```Shell
  git clone https://github.com/endernewton/iter-reason.git
  ```

  - Set up data, here we use ADE20K as an example to setup.

### Reference

```
@inproceedings{chen18iterative,
    author = {Xinlei Chen and Li-Jia Li and Li Fei-Fei and Abhinav Gupta},
    title = {Iterative Visual Reasoning Beyond Convolutions},
    booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
    Year = {2018}
}
```

The idea of spatial memory was developed in:
```
@inproceedings{chen2017spatial,
    author = {Xinlei Chen and Abhinav Gupta},
    title = {Spatial Memory for Context Reasoning in Object Detection},
    booktitle = {Proceedings of the International Conference on Computer Vision},
    Year = {2017}
}
```