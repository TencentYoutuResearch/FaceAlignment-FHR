# Fractional Heatmap Regression
### [[Paper]](https://arxiv.org/pdf/1811.00342.pdf) [[Slides]](https://drive.google.com/open?id=12llt9uzYIUG4Xgx0G9YqnR8rUs1tGYN-) [[Supp]](https://drive.google.com/open?id=1cFyjZWdGOBZ8t-63bZehERMaKpTkawwe)

### Citation
If you find this work useful in your research, please consider citing (* indicates equal contributions):

	@inproceedings{tai-FHR-2019,
	  title={Towards Highly Accurate and Stable Face Alignment for High-Resolution Videos},
	  author={Tai, Ying* and Liang, Yicong* and Liu, Xiaoming and Duan, Lei and Li, Jilin and Wang, Chengjie and Huang, Feiyue and Chen, Yu},
	  booktitle={The AAAI Conference on Artificial Intelligence (AAAI)},
	  year={2019}
	}

### Demo
<p align='center'>
  <img src='imgs/demo.gif' width='1000'/>
</p>

## Prerequisites
- Torch
- Linux
- NVIDIA GPU + CUDA CuDNN 


## Getting Started
### Setup

Clone the github repository:

```bash
git  clone https://github.com/tyshiwo/FHR_alignment.git
cd FHR_alignment
```

### Data & Models
1. Download the training and testing data include datasetsets 300W and 300VW from [[Google drive]](https://drive.google.com/open?id=1NFJ4AhtM33Gus-Ri_lhwgMEZnNCDLqdl). Put it into the root path (i.e., FHR_alignment/data)
2. Download the pretrained models for datasets 300W and 300VW from [[Google drive]](https://drive.google.com/open?id=1pqL1oRnbvP8zTm50a_xn94S1r6cOVo6C). Put it into the root path (i.e., FHR_alignment/models)
### Training

For dataset 300W:

```bash
cd training_code
sh exec_train_300W_fhr.sh
```

For dataset 300VW:

```bash
sh exec_train_300VW_fhr.sh
```

### Testing
For dataset 300W:

```bash
cd testing_code
sh test_300W_fhr.sh
```

For dataset 300VW:

```bash
sh test_300VW_fhr.sh
```




