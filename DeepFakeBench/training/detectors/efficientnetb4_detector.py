'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706
# description: Class for the EfficientDetector

Functions in the Class are summarized as:
1. __init__: Initialization
2. build_backbone: Backbone-building
3. build_loss: Loss-function-building
4. features: Feature-extraction
5. classifier: Classification
6. get_losses: Loss-computation
7. get_train_metrics: Training-metrics-computation
8. get_test_metrics: Testing-metrics-computation
9. forward: Forward-propagation

Reference:
@inproceedings{tan2019efficientnet,
  title={Efficientnet: Rethinking model scaling for convolutional neural networks},
  author={Tan, Mingxing and Le, Quoc},
  booktitle={International conference on machine learning},
  pages={6105--6114},
  year={2019},
  organization={PMLR}
}
'''

import os
import datetime
import logging
import numpy as np
from sklearn import metrics
from typing import Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC
import random

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='efficientnetb4')
class EfficientDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.loss_func = self.build_loss(config)

    def build_backbone(self, config):
        # 只构图；不要把整模权重路径传到骨干，避免 from_pretrained 误读
        backbone_class = BACKBONE[config['backbone_name']]
        model_config = dict(config['backbone_config'])  # 拷贝一份，避免污染原 config
        model_config.pop('pretrained', None)  # 关键：去掉 pretrained 字段
        backbone = backbone_class(model_config)
        logger.info('EfficientNet-B4 backbone built (weights will be loaded by test.py --weights_path).')
        return backbone

    def build_loss(self, config):
        loss_class = LOSSFUNC[config['loss_func']]
        return loss_class()

    def features(self, data_dict: dict) -> torch.Tensor:
        return self.backbone.features(data_dict['image'])

    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        return self.backbone.classifier(features)

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label'];
        pred = pred_dict['cls']
        loss = self.loss_func(pred, label)
        return {'overall': loss}

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label'];
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}

    def forward(self, data_dict: dict, inference=False) -> dict:
        feat = self.features(data_dict)
        logits = self.classifier(feat)
        prob = torch.softmax(logits, dim=1)[:, 1]
        return {'cls': logits, 'prob': prob, 'feat': feat}


