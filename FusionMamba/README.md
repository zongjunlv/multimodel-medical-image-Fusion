# FusionMamba: Dynamic Feature Enhancement for Multimodal Image Fusion with Mamba


[Arxiv](https://arxiv.org/abs/2404.09498)| [Code](https://github.com/millieXie/FusionMamba) | 

## 1. Create Environment

conda create -n FusionMamba python=3.8

conda activate FusionMamba 

pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117 

pip install packaging pip install timm==0.4.12 

pip install pytest chardet yacs termcolor

pip install submitit tensorboardX 

pip install triton==2.0.0

pip install causal_conv1d==1.0.0 # causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl

pip install mamba_ssm==1.0.1 # mmamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl

pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs

## 2. Prepare Your Dataset
dataset
```

/dataset/
        set00-setXX/
                        V000-VXXX/
                                        IRimages
                                        VISimages

```

## 3. Pretrain Weights
IRVIS:
This file is an infrared and visible light model file. Its training hyperparameters have been written so you can use them as you wish. 
link：https://pan.baidu.com/s/1wHqLA3R2ovZyEfTC00wwsg?pwd=6yr2 
password：6yr2

CT-MRI:
link：https://pan.baidu.com/s/1lrEiVLs2p8cMyCV-GN_ZKQ?pwd=a3v8 
password：a3v8

PET-MRI:

Apologies, the dataset link has expired. Let me know if you need the dataset, and I'll send it directly to your email.


(CT-MRI-tinymodel) Note: You need to modify line 794 in the file `FusionMamba/models/vmamba_Fusion_efficross.py` by changing `{depths=[2, 2, 9, 2], depths_decoder=[2, 9, 2, 2]}` to `{depths=[2, 2, 1, 2], depths_decoder=[2, 1, 2, 2]}`.


 
 ## 4.Train
 
```
python train.py
```
## 5.Test

```
python test.py
```
## 6.Datasets
KAIST:(https://github.com/SoonminHwang/rgbt-ped-detection)

Link：https://pan.baidu.com/s/1xIlpL21EA7PdFC5PpLviow?pwd=gf0u 
password：gf0u

medical image fusion data:

This dataset is sourced from the Harvard Public Medical Imaging Collection (https://www.med.harvard.edu/aanlib/home.html), consisting of paired medical images. You may download the original data individually by visiting the official website of the Harvard Public Medical Imaging Database. To facilitate research, we have gathered and processed these paired datasets, which are provided solely for academic research purposes. If you find our processed dataset and this study helpful to your work, we kindly ask that you cite the FusionMamba project in your research. Thank you for your understanding and support.

CT-MRI:

link: https://pan.baidu.com/s/1qcuY6AokvPyIsUSMSCqzOg?pwd=cjcf 
password: cjcf

PET-MRI:

link：https://pan.baidu.com/s/1Cs5fUy4EoI-MtX9DUPRFGA?pwd=blt4 
password：blt4

SPECT-MRI:

link: https://pan.baidu.com/s/1bwRxFbj9u4gDo32ol4zGaA?pwd=74my 
password: 74my



## 7.Path
You need to modify the data input path in lines 47-48 of the TaskFusion_dataset.py file. If you intend to train using CT-MRI data, ensure that the structure of the CT-MRI data, after data augmentation, matches the structure of the KAIST dataset. Additionally, replace lwir with CT/PET/SPECT and visible with MRI.


## 8.Citation

>@article{xie2024fusionmamba,  
 >&nbsp;&nbsp;&nbsp;&nbsp;title={Fusionmamba: Dynamic feature enhancement for multimodal image fusion with mamba},      
 >&nbsp;&nbsp;&nbsp;&nbsp;author={Xie, Xinyu and Cui, Yawen and Tan, Tao and Zheng, Xubin and Yu, Zitong},  
 >&nbsp;&nbsp;&nbsp;&nbsp;journal={Visual Intelligence},
 >&nbsp;&nbsp;&nbsp;&nbsp;volume={2},
 >&nbsp;&nbsp;&nbsp;&nbsp;number={37},
 >&nbsp;&nbsp;&nbsp;&nbsp;year={2024}
 >}  
