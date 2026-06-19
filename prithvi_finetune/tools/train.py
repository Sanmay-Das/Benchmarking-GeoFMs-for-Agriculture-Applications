import argparse
import os
import sys
import mmcv
from mmcv import Config
from mmcv.runner import set_random_seed
from mmseg.apis import train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='config file path')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    cfg = Config.fromfile(args.config)
    
    set_random_seed(args.seed, deterministic=False)
    
    datasets = [build_dataset(cfg.data.train)]
    model = build_segmentor(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    
    train_segmentor(model, datasets, cfg, distributed=False, validate=True)

if __name__ == '__main__':
    main()