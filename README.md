# AC-MT for Semi-supervised LA Segmentation

![](https://img.shields.io/github/downloads/lemoshu/AC-MT/)

This repository will hold the PyTorch implementation of the paper [Ambiguity-selective consistency regularization for mean-teacher semi-supervised medical image segmentation](https://www.sciencedirect.com/science/article/pii/S1361841523001408). 

## Introduction
### Abstract
Semi-supervised learning has greatly advanced medical image segmentation since it effectively alleviates the need of acquiring abundant annotations from experts, wherein the mean-teacher model, known as a milestone of perturbed consistency learning, commonly serves as a standard and simple baseline. Inherently, learning from consistency can be regarded as learning from stability under perturbations. Recent improvement leans toward more complex consistency learning frameworks, yet, little attention is paid to the consistency target selection. Considering that the ambiguous regions from unlabeled data contain more informative complementary clues, we improve the mean-teacher model to a novel ambiguity-consensus mean-teacher (AC-MT) model. Particularly, we comprehensively introduce and benchmark a family of plug-and-play strategies for ambiguous target selection from the perspectives of entropy, model uncertainty and label noise self-identification, respectively. Then, the estimated ambiguity map is incorporated into the consistency loss to encourage consensus between the two models' predictions in these informative regions. In essence, our AC-MT aims to find out the most worthwhile voxel-wise targets from the unlabeled data, and the model especially learns from the perturbed stability of these informative regions. 

### Key Features
- A new consistency target selection perspective to boost the mean-teacher semi-supervised medical volume segmentation.
- Four plug-and-play ambiguity selection strategies are provided.


## Requirements
Check requirements.txt.
* Pytorch version >=0.4.1.
* Python == 3.6 
* Cleanlab (for S-Err. strategy)

## Usage

1. Clone the repo:
```
cd ./AC-MT
```

2. Data Preparation
Refer to ./data for details


3. Train
```
cd ./code
python train_ACMT_{}_3D.py --labeled_num {} --gpu 0
```

4. Test 
```
cd ./code
python test_3D.py
```


## Citation

If you find this paper useful, please cite as:
```
@article{xu2023ambiguity,
  title={Ambiguity-selective consistency regularization for mean-teacher semi-supervised medical image segmentation},
  author={Xu, Zhe and Wang, Yixin and Lu, Donghuan and Luo, Xiangde and Yan, Jiangpeng and Zheng, Yefeng and Tong, Raymond Kai-yu},
  journal={Medical Image Analysis},
  pages={102880},
  year={2023},
  publisher={Elsevier}
}
```
The implementation of other SSL approaches can be referred to by the author Dr. Luo's [SSL4MIS project](https://github.com/HiLab-git/SSL4MIS).
