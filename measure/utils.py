import os
import pandas as pd
import yaml
import glob
from easydict import EasyDict as edict
from collections import OrderedDict

class Logger():
    def __init__(self, save_path,save_name):
        self.save_path = save_path
        self.log = None
        self.save_name = save_name

    def update(self, epoch,train_log):
        item = OrderedDict({'name':epoch})
        item.update(train_log)
        # print("\033[0;33mTrain:\033[0m", train_log)
        self.update_csv(item)

    def update_csv(self,item):
        tmp = pd.DataFrame(item, index=[0])
        if self.log is not None:
            self.log = self.log._append(tmp, ignore_index=True)
        else:
            self.log = tmp
        self.log.to_csv('%s/%s.csv' % (self.save_path, self.save_name), index=False)

def get_path(base_dir):
    cfg_path = 'measure/config.yaml'
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

