# Single-Stage Instance Shadow Detection with Bidirectional Relation Learning (CVPR 2021 **Oral**)


[Tianyu Wang](https://stevewongv.github.io)\*, [Xiaowei Hu](https://xw-hu.github.io)\*, [Chi-Wing Fu](http://www.cse.cuhk.edu.hk/~cwfu/), and Pheng-Ann Heng
 (\* Joint first authors.)

Instance Shadow Detection aims to find shadow instances, object instances and shadow-object associations; this task benefits many vision applications, such as light direction estimation and photo editing.

In this paper, we present a new single-stage fully convolutional network architecture with a bidirectional relation learning module to directly learn the relations of shadow and object instances in an end-to-end manner.

[[📄 Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Single-Stage_Instance_Shadow_Detection_With_Bidirectional_Relation_Learning_CVPR_2021_paper.pdf)] [[👇🏼 Video](http://www.youtube.com/watch?v=p0b_2SsFypw)]  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1y9UpS5uA1YuoMyvYVzcKL4ltA_FDu_x0?usp=sharing)

[![YouTube](https://cdn.jsdelivr.net/gh/stevewongv/image-hosting@master/20210618/CVPR 2021-video-2364-0001.668t20smqb28.png)](http://www.youtube.com/watch?v=p0b_2SsFypw "Single-Stage Instance Shadow Detection with Bidirectional Relation Learning (CVPR'21 Oral)")



## Requirement

```
pip install -r requirement.txt
```

Note that we tested on CUDA10.2 / PyTorch 1.6.0, CUDA11.1 / PyTorch 1.8.0 and [Colab](https://colab.research.google.com/drive/1y9UpS5uA1YuoMyvYVzcKL4ltA_FDu_x0?usp=sharing).

## Installation

This repo is implemented on [AdelaiDet](https://github.com/aim-uofa/AdelaiDet), so first build it with:
```bash
$ cd SSIS
$ python setup.py build develop
```

## Dataset and pre-trained model



| Method | SOAP mask  | SOAP bbox  | mask AP | box AP |
| :-----:| :--------: | :--------: |:------: |:-----: |
| [LISA]() | 21.2     | 21.7       | 37.0    | 38.1   |
| **Ours** | 27.4     | 25.5       | 40.3    | 39.6   |

Download the dataset and `model_final.pth` from [Google drive](https://drive.google.com/drive/folders/1MKxyq3R6AUeyLai9i9XWzG2C_n5f0ppP). Put dataset file in the `../dataset/` and put pretrained model in the `tools/output/SSIS_MS_R_101_bifpn_with_offset_class/`. Note that we add new annotation file in the SOBA dataset. 

## Quick Start 
### Demo
To evaluate the results, try the command example:

```bash
$ cd demo
$ python demo.py --input ./samples
```

### Training
```bash
$ cd tools
$ python train_net.py \
    --config-file ../configs/SSIS/MS_R_101_BiFPN_with_offset_class.yaml \
    --num-gpus 2 
``` 

### Evaluation
```bash
$ python train_net.py \
    --config-file ../configs/SSIS/MS_R_101_BiFPN_with_offset_class.yaml \
    --num-gpus 2 --resume --eval-only
$ python SOAP.py --path PATH_TO_YOUR_DATASET/SOBA \ 
    --input-name ./output/SSIS_MS_R_101_bifpn_with_offset_class
``` 

# Citation
If you use LISA, SSIS, SOBA, or SOAP, please use the following BibTeX entry.

```
@InProceedings{Wang_2020_CVPR,
author    = {Wang, Tianyu and Hu, Xiaowei and Wang, Qiong and Heng, Pheng-Ann and Fu, Chi-Wing},
title     = {Instance Shadow Detection},
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month     = {June},
year      = {2020}
}

@InProceedings{Wang_2021_CVPR,
author    = {Wang, Tianyu and Hu, Xiaowei and Fu, Chi-Wing and Heng, Pheng-Ann},
title     = {Single-Stage Instance Shadow Detection With Bidirectional Relation Learning},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month     = {June},
Year      = {2021},
pages     = {1-11}
}
```