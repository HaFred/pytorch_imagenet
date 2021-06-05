## Fred's update
For BP, 180 epochs training, 
* ResNet-18 reaches 70.782% top1 accuracy.
* AlexNet reaches ...

For Efficient-Grad, 180 epochs training,
* ResNet-18
* AlexNet 


## Original
The implementation of AlexNet in [PyTorch Vision](https://github.com/pytorch/vision) is not actually the ordinary version. In this case, this repository reimplements some of the networks for the author's usage.

## Pre-requirements
- [Caffe](https://github.com/BVLC/caffe)
- [PyTorch](https://github.com/pytorch/pytorch)

# Data Preparation
The original data loader ([link](https://github.com/pytorch/vision#imagenet-12)) is slow. Therefore, I build a new data loader with Caffe utils.
### Genearte LMDB
The preprocessed datasets can be found [here](https://drive.google.com/uc?export=download&id=0B-7I62GOSnZ8aENhOEtESVFHa2M). Please download it and uncompress it into the directory of ```./networks/data/```.  
To generate the dataset from raw images, please follow the [instructions](http://caffe.berkeleyvision.org/gathered/examples/imagenet.html) for Caffe to build the LMDB dataset of ImageNet. However, the ```key``` used in the LMDB dataset is not suitable for accessing. Therefore, please use the script ```./tools/fix_key.sh``` to convert the keys.
Preprocessed data will be available soon.
### Load LMDB
Please change the variable ```lmdb_dir``` in ```./datasets/folder.py``` to the directory which includes the training and validating LMDB datasets.
# AlexNet
The implementation is in ```./networks/model_list/alexnet.py```. Since PyTorch does not support local response normalization (LRN) layer, I implements it also. The trained model will be available soon.
### Training from scratch
```bash
$ cd networks
$ python main.py --arch alexnet
```

### Evaluate pretained model
Pretrained model is available [here](https://drive.google.com/uc?export=download&id=0B-7I62GOSnZ8NzVxZndDU2dYcHM). The pretained model achieves an accuracy of 57.494% (Top-1) and 80.588% (Top-5). Please download it and put it under the directory of ```./networks/model_list/```. Run the evaluation by:
```bash
$ python main.py --arch alexnet --pretrained --evaluate
```
