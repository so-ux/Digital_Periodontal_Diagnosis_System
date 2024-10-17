import os
import yaml
import glob
from easydict import EasyDict as edict


def get_path(base_dir):
    cfg_path = 'ModelFusion/config_model_fusion.yaml'
    cfg = yaml.safe_load(open(cfg_path))

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    for key, value in cfg['root_file'].items():
        cfg['root_file'][key] = glob.glob(os.path.join(base_dir, '..', value))[0]

    for key, value in cfg['path'].items():
        new_path = os.path.join(base_dir, value)
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        cfg['path'][key] = new_path

    for key, value in cfg['file'].items():
        cfg['file'][key] = os.path.join(base_dir, value)

    for key, value in cfg['oral_scan_seg_file'].items():
        cfg['oral_scan_seg_file'][key] = glob.glob(os.path.join(base_dir,'..', 'oral scan seg', value))[0]

    return edict(cfg)
