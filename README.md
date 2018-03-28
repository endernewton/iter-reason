# Iterative Visual Reasoning Beyond Convolutions
By Xinlei Chen, Li-Jia Li, Li Fei-Fei and Abhinav Gupta. 

### Disclaimer
  - This is the authors' implementation of the system described in the paper, not an official Google product.
  - 

### Prerequisites

1. Tensorflow, tested with version 1.6 with Ubuntu 16.04, installed with:
  ```Shell
  pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.6.0-cp27-none-linux_x86_64.whl
  ```

2. Other packages needed can be installed with `pip`:
  ```Shell
  pip install Cython easydict matplotlib opencv-python Pillow pyyaml scipy
  ```

### Installation

1. Clone the repository.
  ```Shell
  git clone https://github.com/endernewton/iter-reason.git
  ```

2. Set up data, here we use ADE20K as an example.
  ```Shell
  cd data/ADE
  wget -v http://groups.csail.mit.edu/vision/datasets/ADE20K/ADE20K_2016_07_26.zip
  tar -xzvf ADE20K_2016_07_26.zip
  mv ADE20K_2016_07_26/* ./
  rmdir ADE20K_2016_07_26
  # then get the train/val/test split
  wget -v http://xinleic.xyz/data/ADE_split.tar.gz
  tar -xzvf ADE_split.tar.gz
  rm -vf ADE_split.tar.gz
  cd ../..
  ```

3. Set up pre-trained ImageNet models. This is similarly done in [tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn). For example (Resnet 50):
  ```Shell
   cd data/imagenet_weights
   wget -v http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz
   tar -xzvf resnet_v1_50_2016_08_28.tar.gz
   mv resnet_v1_50.ckpt res50.ckpt
   cd ../..
   ```

### References

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