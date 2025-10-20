"""
eval pretained model.
"""
import os
import numpy as np
from os.path import join
import cv2
import random
import datetime
import time
import yaml
import pickle
from tqdm import tqdm
from copy import deepcopy
from PIL import Image as pil_image
from metrics.utils import get_test_metrics
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from dataset.ff_blend import FFBlendDataset
from dataset.fwa_blend import FWABlendDataset
from dataset.pair_dataset import pairDataset
import csv

from trainer.trainer import Trainer
from detectors import DETECTOR
from metrics.base_metrics_class import Recorder
from collections import defaultdict

import argparse
from logger import create_logger

parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str, 
                    default='/home/zhiyuanyan/DeepfakeBench/training/config/detector/resnet34.yaml',
                    help='path to detector YAML file')
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument('--weights_path', type=str, 
                    default='/mntcephfs/lab_data/zhiyuanyan/benchmark_results/auc_draw/cnn_aug/resnet34_2023-05-20-16-57-22/test/FaceForensics++/ckpt_epoch_9_best.pth')
#parser.add_argument("--lmdb", action='store_true', default=False)

parser.add_argument('--save_csv', type=str, default=None,
                    help='Path to save per-sample predictions CSV. If multiple test datasets, suffix _{name}.csv will be used.')
parser.add_argument('--save_feat_dir', type=str, default=None,
                    help='Directory to save features as .npy per dataset (optional).')
parser.add_argument('--csv_append', action='store_true',
                    help='If set, append to CSV when it exists instead of overwriting.')


args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])


def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        config['test_dataset'] = test_name  # specify the current test dataset
        test_set = DeepfakeAbstractBaseDataset(
                config=config,
                mode='test', 
            )
        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set, 
                batch_size=config['test_batchSize'],
                shuffle=False, 
                num_workers=int(config['workers']),
                collate_fn=test_set.collate_fn,
                drop_last=False
            )
        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring


def test_one_dataset(model, data_loader):
    prediction_lists = []
    feature_lists = []
    label_lists = []
    for i, data_dict in tqdm(enumerate(data_loader), total=len(data_loader)):
        # get data
        data, label, mask, landmark = \
        data_dict['image'], data_dict['label'], data_dict['mask'], data_dict['landmark']
        label = torch.where(data_dict['label'] != 0, 1, 0)
        # move data to GPU
        data_dict['image'], data_dict['label'] = data.to(device), label.to(device)
        if mask is not None:
            data_dict['mask'] = mask.to(device)
        if landmark is not None:
            data_dict['landmark'] = landmark.to(device)

        # model forward without considering gradient computation
        predictions = inference(model, data_dict)
        label_lists += list(data_dict['label'].cpu().detach().numpy())
        prediction_lists += list(predictions['prob'].cpu().detach().numpy())
        feature_lists += list(predictions['feat'].cpu().detach().numpy())
    
    return np.array(prediction_lists), np.array(label_lists),np.array(feature_lists)
    
def test_epoch(model, test_data_loaders):
    # set model to eval mode
    model.eval()

    metrics_all_datasets = {}
    multiple_sets = len(test_data_loaders.keys()) > 1

    # 准备 CSV 输出（多数据集时自动加后缀）
    def _resolve_csv_path(base_csv, ds_name):
        if base_csv is None:
            return None
        if multiple_sets:
            root, ext = os.path.splitext(base_csv)
            return f"{root}_{ds_name}{ext or '.csv'}"
        return base_csv

    # 如果要存特征，先准备目录
    if args.save_feat_dir is not None:
        os.makedirs(args.save_feat_dir, exist_ok=True)

    # testing for all test data
    for key in test_data_loaders.keys():
        dl = test_data_loaders[key]
        data_dict_all = dl.dataset.data_dict  # 包含 image 路径/键
        # 逐样本推理
        predictions_nps, label_nps, feat_nps = test_one_dataset(model, dl)

        # 评测指标
        metric_one_dataset = get_test_metrics(
            y_pred=predictions_nps,
            y_true=label_nps,
            img_names=data_dict_all['image']
        )
        metrics_all_datasets[key] = metric_one_dataset

        # 打印
        tqdm.write(f"dataset: {key}")
        for k, v in metric_one_dataset.items():
            tqdm.write(f"{k}: {v}")

        # ====== 保存逐样本结果 ======
        csv_path = _resolve_csv_path(args.save_csv, key)
        if csv_path is not None:
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            write_header = (not os.path.exists(csv_path)) or (not args.csv_append)
            mode = 'a' if (args.csv_append and os.path.exists(csv_path)) else 'w'
            with open(csv_path, mode, newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(['dataset', 'image', 'label', 'prob'])
                for img, lab, prob in zip(data_dict_all['image'], label_nps, predictions_nps):
                    writer.writerow([key, img, int(lab), float(prob)])
            tqdm.write(f"[Saved] per-sample predictions → {csv_path}")

        # ====== 可选：保存特征 ======
        if args.save_feat_dir is not None and feat_nps is not None and len(feat_nps) > 0:
            feat_path = os.path.join(args.save_feat_dir, f"{key}_features.npy")
            np.save(feat_path, feat_nps)
            tqdm.write(f"[Saved] features → {feat_path}")

    return metrics_all_datasets

@torch.no_grad()
def inference(model, data_dict):
    predictions = model(data_dict, inference=True)
    return predictions

def main():
    # 1) 读取配置
    with open(args.detector_path, 'r') as f:
        det_cfg = yaml.safe_load(f) or {}

    with open('./training/config/test_config.yaml', 'r') as f:
        test_cfg = yaml.safe_load(f) or {}

    # 2) 以 detector.yaml 为基准，先复制一份
    config = dict(det_cfg)

    # 3) 用 test_config.yaml 只“补缺”，并对少数白名单键允许覆盖
    allow_override = {'log_dir'}  # 你要全局固定日志目录
    for k, v in (test_cfg or {}).items():
        if k in allow_override:
            config[k] = v
        elif k not in config:
            config[k] = v

    # 4) CLI 覆盖（最高优先级）
    # --test_dataset: 允许传一个或多个；统一成 list
    if args.test_dataset:
        if isinstance(args.test_dataset, (list, tuple)):
            config['test_dataset'] = list(args.test_dataset)
        else:
            config['test_dataset'] = [args.test_dataset]

    # --weights_path: 作为最终权重，同时同步到 pretrained 字段
    weights_path = None
    if args.weights_path:
        weights_path = args.weights_path
        config['weights_path'] = args.weights_path
        config['pretrained'] = args.weights_path

    # 5) 兼容 label_dict（如果 det_cfg 里带了）
    if 'label_dict' in det_cfg:
        config['label_dict'] = det_cfg['label_dict']

    # 6) 随机种子 / cudnn
    init_seed(config)
    if config.get('cudnn', False):
        cudnn.benchmark = True

    # 7) DataLoader
    test_data_loaders = prepare_testing_data(config)

    # 8) 模型构建 & 加载权重
    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)

    # 权重加载：优先 CLI；否则尝试 config['pretrained']
    if not weights_path:
        weights_path = config.get('pretrained', None)

    if weights_path:
        try:
            try:
                epoch = int(os.path.basename(weights_path).split('.')[0].split('_')[2])
            except Exception:
                epoch = 0
            ckpt = torch.load(weights_path, map_location=device)
            # 兼容存了 state_dict / 全量权重两种格式
            state = ckpt.get('state_dict', ckpt)
            model.load_state_dict(state, strict=True)
            print(f'===> Load checkpoint done! ({weights_path})')
        except Exception as e:
            print(f'!! Failed to load weights: {weights_path}\n{e}')
            return
    else:
        print('!! No weights provided (use --weights_path or set pretrained).')
        return

    # 9) 测试
    best_metric = test_epoch(model, test_data_loaders)
    print('===> Test Done!')


if __name__ == '__main__':
    main()
