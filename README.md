# Periodontal disease analysis system


### Introduction
This repository is for our paper 'A digital system for periodontal disease diagnosis from in vivo intra oral scan and cone beam CT image.
### Installation
-- Step 1: IOS model segmentation
```
cd IOSModelSegmentation
conda create -n dental python=3.7
conda activate dental
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```
-- Step 2:
```
cd..
conda env create -f env.yaml
```
This repository is based on PyTorch 1.12.1, CUDA 11.3. (Other versions should be also OK, but not validate).

For other libs, please use "pip install xxx" or "conda install xxx".

Please note that you need to install the "Blender" software to calculate the distance.

### Setting up Paths
Example tree structure:
```
    data/Patient1
    ├── Patient1_LowerJawScan.off
    ├── Patient1_UpperJawScan.off 
    └── Patient1_image.nii.gz
    data/Patient2
    ├── Patient2_LowerJawScan.off
    ├── Patient2_UpperJawScan.off
    └── Patient2_image.nii.gz
    ...
 ```       
### Model
Download the pretrained models of the CBCT image segmentation on the Baidu drive (link: /model/model.txt), and put them on the model folder.

### Inference
1. IOS model segmentation
```
cd IOSModelSegmentation
python inference.py -i path_to_data -m ../model
```
2. CBCT image segmentation & model fusion & measure
```
cd ..
python inference.py -i path_to_data -m 2_model -six False -b_r 5000
```

### Questions
Please contact Minhui Tan
