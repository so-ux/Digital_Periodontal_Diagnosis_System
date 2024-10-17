# DentalModelSegmentation
## Installation
```
conda env create -f env.yaml
cd src/pointnet2 & pip install -e .
```

## Start
```
python inference.py -i path_to_dental_model -o path_to_save_results -m path_to_model_weights
```
