# subspace-attack.pytorch
Code for our NeurIPS 2019 paper [Subspace Attack: Exploiting Promising Subspaces for Query-Efficient Black-box Attacks](https://papers.nips.cc/paper/8638-subspace-attack-exploiting-promising-subspaces-for-query-efficient-black-box-attacks).

Our paper is also available on [arxiv](https://arxiv.org/pdf/1906.04392.pdf).

## Environments
* Python 3.5
* PyTorch 1.1.0
* torchvision 0.2.2
* glog 0.3.1

## Datasets and Reference Models
We use CIFAR-10 and ImageNet in our experiment, and these two datasets should be prepared into the following structure:

```
subspace-attack.pytorch  
└───data
    ├── cifar10 
    │   ├── cifar-10-batches-py
    │   └── cifar-10-python.tar.gz
    └── imagenet 
        ├── train
        └── val
```

For reference models, please download at [this link](https://drive.google.com/file/d/1aXTmN2AyNLdZ8zOeyLzpVbRHZRZD0fW0/view?usp=sharing), extract and put them into ```data/``` directory.

## Usage
To mount an untargeted l-inf attack on a WRN victim model on CIFAR-10 using AlexNet+VGGNets as references (i.e., the CIFAR-10->WRN->Ours row in Table 1 of our paper)

```
python3 attack.py --arch wrn-28-10-drop --attack-type untargeted --norm-type linf --dataset cifar10 --ref-arch alexnet_bn vgg11_bn vgg13_bn vgg16_bn vgg19_bn --ref-arch-train-data cifar10.1
```

We also provide many logs for experiments used in our paper at [this link](https://drive.google.com/file/d/1hNpWgmlUGjEfFpNcPq1ULfyYHf6RkoM-/view?usp=sharing).

## Acknowledgements
The following resources are very helpful for our work:

* [Pretrained models and for ImageNet](https://github.com/Cadene/pretrained-models.pytorch)
* [Pretrained models for CIFAR-10](https://github.com/bearpaw/pytorch-classification)
* [GDAS](https://github.com/D-X-Y/GDAS)
* [Official AutoAugment implementation](https://github.com/tensorflow/models/tree/master/research/autoaugment)
* [ImageNet FGSM adversarially trained Inception-V3 model](https://github.com/tensorflow/models/tree/master/research/adv_imagenet_models)
* [Carlini's CIFAR-10 ConvNet](https://github.com/carlini/nn_robust_attacks)

## Citation
Please cite our work in your publications if it helps your research:

```
@inproceedings{subspaceattack,
  title={Subspace Attack: Exploiting Promising Subspaces for Query-Efficient Black-box Attacks},
  author={Guo, Yiwen and Yan, Ziang and Zhang, Changshui},
  booktitle={Advances in Neural Information Processing Systems},
  pages={3820--3829},
  year={2019}
}
```
