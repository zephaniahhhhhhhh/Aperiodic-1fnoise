# -*- coding: utf-8 -*-
"""
config.py
数据集配置文件 - 集中管理不同数据集的参数
"""

from pathlib import Path
import numpy as np


class DatasetConfig:
    """数据集配置基类"""

    def __init__(self, name):
        self.name = name
        self.output_dir = Path(f"results_{name}")
        self.output_dir.mkdir(exist_ok=True)
        self.figure_dir = self.output_dir / "figures"
        self.figure_dir.mkdir(exist_ok=True)

    def get_info(self):
        """打印配置信息"""
        print(f"\n{'=' * 60}")
        print(f"📋 数据集配置: {self.name}")
        print(f"{'=' * 60}")
        print(f"  数据文件: {self.data_path}")
        print(f"  通道数: {self.n_channels}")
        print(f"  通道名: {self.ch_names[:5]}... (共{len(self.ch_names)}个)")
        print(f"  对称通道对: {self.n_pairs}对")
        print(f"  采样率: {self.sfreq} Hz")
        print(f"  类别数: {self.n_classes}")
        print(f"  类别名称: {self.class_names}")
        print(f"{'=' * 60}\n")


class SADConfig(DatasetConfig):
    """SAD数据集配置"""

    def __init__(self):
        super().__init__("SAD")

        # 数据路径
        self.data_path = 'SAD.mat'

        # 数据集参数
        self.sfreq = 128
        self.n_channels = 30

        # 通道信息
        self.ch_names = [
            'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT7', 'FC3', 'FCz',
            'FC4', 'FT8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'TP7', 'CP3', 'CPz',
            'CP4', 'TP8', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'Oz', 'O2'
        ]

        # 对称通道对
        self.pair_names = [
            ('Fp1', 'Fp2'), ('F3', 'F4'), ('F7', 'F8'),
            ('FC3', 'FC4'), ('C3', 'C4'), ('CP3', 'CP4'),
            ('P3', 'P4'), ('O1', 'O2'),
            ('T3', 'T4'), ('T5', 'T6'),
            ('FT7', 'FT8'), ('TP7', 'TP8')
        ]

        # 分类信息
        self.n_classes = 2
        self.class_names = ['Alert', 'Fatigue']

        # 标签处理（SAD数据集标签已经是0/1，不需要重映射）
        self.need_label_mapping = False
        self.label_key = 'substate'

        # 其他数据键
        self.data_key = 'EEGsample'
        self.has_subject_index = False

        # 计算派生参数
        self.pair_idx = [(self.ch_names.index(a), self.ch_names.index(b))
                         for a, b in self.pair_names]
        self.n_pairs = len(self.pair_idx)

        # 参考通道（用于全局MI计算）
        self.ref_channel_idx = 14  # Cz
        self.ref_channel_name = 'Cz'


class SEEDConfig(DatasetConfig):
    """SEED数据集配置"""

    def __init__(self):
        super().__init__("SEED")

        # 数据路径
        self.data_path = 'seed.mat'

        # 数据集参数
        self.sfreq = 128
        self.n_channels = 17

        # 通道信息
        self.ch_names = [
            'FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8', 'CP1', 'CP2',
            'P1', 'PZ', 'P2', 'PO3', 'POZ', 'PO4', 'O1', 'OZ', 'O2'
        ]

        # 对称通道对
        self.pair_names = [
            ('FT7', 'FT8'), ('T7', 'T8'), ('TP7', 'TP8'),
            ('CP1', 'CP2'), ('P1', 'P2'), ('PO3', 'PO4'), ('O1', 'O2')
        ]

        # 分类信息（SEED是3分类：负面/中性/正面情绪）
        self.n_classes = 3
        self.class_names = ['Relax', 'Tired', 'Sleepy']

        # 标签处理（SEED需要重映射）
        self.need_label_mapping = True
        self.label_key = 'substate'

        # 其他数据键
        self.data_key = 'EEGsample'
        self.has_subject_index = True
        self.subject_key = 'subindex'

        # 计算派生参数
        self.pair_idx = [(self.ch_names.index(a), self.ch_names.index(b))
                         for a, b in self.pair_names]
        self.n_pairs = len(self.pair_idx)

        # 参考通道
        self.ref_channel_idx = self.n_channels // 2  # 中间通道
        self.ref_channel_name = self.ch_names[self.ref_channel_idx]


class CustomConfig(DatasetConfig):
    """自定义数据集配置模板"""

    def __init__(self,
                 name="Custom",
                 data_path="custom.mat",
                 sfreq=128,
                 ch_names=None,
                 pair_names=None,
                 n_classes=2,
                 class_names=None,
                 need_label_mapping=False):

        super().__init__(name)

        self.data_path = data_path
        self.sfreq = sfreq
        self.ch_names = ch_names or []
        self.n_channels = len(self.ch_names)

        self.pair_names = pair_names or []
        self.n_classes = n_classes
        self.class_names = class_names or [f'Class_{i}' for i in range(n_classes)]

        self.need_label_mapping = need_label_mapping
        self.label_key = 'substate'
        self.data_key = 'EEGsample'
        self.has_subject_index = False

        # 计算派生参数
        if self.pair_names:
            self.pair_idx = [(self.ch_names.index(a), self.ch_names.index(b))
                             for a, b in self.pair_names]
            self.n_pairs = len(self.pair_idx)
        else:
            self.pair_idx = []
            self.n_pairs = 0

        self.ref_channel_idx = self.n_channels // 2
        self.ref_channel_name = self.ch_names[self.ref_channel_idx] if self.ch_names else 'Unknown'


# 频段配置（通用）
STANDARD_BANDS = [
    ("1-20Hz", 1, 20),
    ("1-40Hz", 1, 40),
    ("5-40Hz", 5, 40),
    ("5-20Hz", 5, 20),
    ("20-40Hz", 20, 40),
]


def get_config(dataset_name):
    """根据名称获取配置"""
    configs = {
        'SAD': SADConfig,
        'SEED': SEEDConfig,
    }

    if dataset_name not in configs:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(configs.keys())}")

    return configs[dataset_name]()
