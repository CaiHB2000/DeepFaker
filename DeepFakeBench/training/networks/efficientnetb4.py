'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706

The code is for EfficientNetB4 backbone.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from efficientnet_pytorch import EfficientNet
from metrics.registry import BACKBONE
import os
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
import torch.nn as nn
import logging

log = logging.getLogger(__name__)

@BACKBONE.register_module(module_name="efficientnetb4")
class EfficientNetB4(nn.Module):
    def __init__(self, efficientnetb4_config):
        super(EfficientNetB4, self).__init__()
        """ Constructor
        Args:
            efficientnetb4_config: dict
        """
        efficientnetb4_config = efficientnetb4_config or {}

        self.num_classes = efficientnetb4_config["num_classes"]
        inc = efficientnetb4_config["inc"]
        self.dropout = efficientnetb4_config["dropout"]
        self.mode = efficientnetb4_config["mode"]

        # —— 关键改动：默认只构图；除非明确提供的是 ImageNet B4 权重 —— #
        w = efficientnetb4_config.get('pretrained', None)
        if w:
            try:
                self.efficientnet = EfficientNet.from_pretrained('efficientnet-b4', weights_path=w)
                log.info(f"EfficientNet-B4 loaded ImageNet weights from {w}")
            except Exception as e:
                log.warning(f"EfficientNet-B4 failed to load imagenet weights ({w}): {e}; fallback to from_name")
                self.efficientnet = EfficientNet.from_name('efficientnet-b4')
        else:
            self.efficientnet = EfficientNet.from_name('efficientnet-b4')

        # 修改输入通道
        self.efficientnet._conv_stem = nn.Conv2d(inc, 48, kernel_size=3, stride=2, bias=False)

        # 去掉分类头
        self.efficientnet._fc = nn.Identity()

        if self.dropout:
            self.dropout_layer = nn.Dropout(p=self.dropout)

        self.last_layer = nn.Linear(1792, self.num_classes)

        if self.mode == 'adjust_channel':
            self.adjust_channel = nn.Sequential(
                nn.Conv2d(1792, 512, 1, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            )

    def block_part1(self, x):
        x = self.efficientnet._swish(self.efficientnet._bn0(self.efficientnet._conv_stem(x)))
        for idx, block in enumerate(self.efficientnet._blocks[:10]):
            dcr = self.efficientnet._global_params.drop_connect_rate
            if dcr:
                dcr *= float(idx+0) / len(self.efficientnet._blocks)
            x = block(x, drop_connect_rate=dcr)
        return x

    def block_part2(self, x):
        for idx, block in enumerate(self.efficientnet._blocks[10:22]):
            dcr = self.efficientnet._global_params.drop_connect_rate
            if dcr:
                dcr *= float(idx+10) / len(self.efficientnet._blocks)
            x = block(x, drop_connect_rate=dcr)
        return x

    def block_part3(self, x):
        for idx, block in enumerate(self.efficientnet._blocks[22:]):
            dcr = self.efficientnet._global_params.drop_connect_rate
            if dcr:
                dcr *= float(idx+22) / len(self.efficientnet._blocks)
            x = block(x, drop_connect_rate=dcr)
        x = self.efficientnet._swish(self.efficientnet._bn1(self.efficientnet._conv_head(x)))
        return x

    def features(self, x):
        x = self.efficientnet.extract_features(x)
        if self.mode == 'adjust_channel':
            x = self.adjust_channel(x)
        return x

    def end_points(self, x):
        return self.efficientnet.extract_endpoints(x)

    def classifier(self, x):
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        if self.dropout:
            x = self.dropout_layer(x)
        self.last_emb = x
        y = self.last_layer(x)
        return y

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
