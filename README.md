# Iterative Visual Reasoning Beyond Convolutions
By Xinlei Chen, Li-Jia Li, Li Fei-Fei and Abhinav Gupta. 

### Disclaimer
  - This is the authors' implementation of the system described in the paper, not an official Google product.
  - Right now the available reasoning module is based on convolutions and spatial memory.

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

3. Set up pre-trained ImageNet models. This is similarly done in [tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn). Here by default we use Resnet-50 as the backbone:
  ```Shell
   cd data/imagenet_weights
   wget -v http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz
   tar -xzvf resnet_v1_50_2016_08_28.tar.gz
   mv resnet_v1_50.ckpt res50.ckpt
   cd ../..
   ```

4. Compile the library.
  ```Shell
  cd lib
  make
  ```

5. Now you are ready to run! For example, to train and test the baseline:
  ```Shell
  ./experiments/scripts/train.sh [GPU_ID] [DATASET] [STEPS] [ITER] 
  # GPU_ID is the GPU you want to test on
  # DATASET in {ade, coco, vg} is the dataset to train/test on, defined in the script
  # STEPS (x10K) is the number of iterations before it reduces learning rate, can support multiple steps separated by character 'a'
  # ITER (x10K) is the total number of iterations to run
  # Examples:
  # train on ADE20K for 320K iterations, reducing learning rate at 280K.
  ./experiments/scripts/train.sh 0 ade 28 32
  # train on COCO for 640K iterations, reducing at 500K.
  ./experiments/scripts/train.sh 1 coco 50 64 
  ```
  To train and test the reasoning modules:
  ```Shell
  ./experiments/scripts/train_memory.sh [GPU_ID] [DATASET] [MEM] [STEPS] [ITER] 
  # MEM is the type of reasoning modules to use, for example 
  # Examples:
  # train on ADE20K on the attention based memory.
  ./experiments/scripts/train.sh 0 ade sepat 28 32
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