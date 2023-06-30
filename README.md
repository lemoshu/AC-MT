# AC-MT for Semi-supervised LA Segmentation

This repository will hold the PyTorch implementation of the paper [Ambiguity-selective consistency regularization for mean-teacher semi-supervised medical image segmentation](https://www.sciencedirect.com/science/article/pii/S1361841523001408). 

## Requirements
Check requirements.txt.
* Pytorch version >=0.4.1.
* Python == 3.6 
* Cleanlab

## Usage

1. Clone the repo:
```
cd ./AC-MT
```

2. Data Preparation
Refer to ./data


3. Train
```
cd ./code
python train_ACMT_{}_3D.py
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
  year={2023},
  publisher={Elsevier}
}
```
