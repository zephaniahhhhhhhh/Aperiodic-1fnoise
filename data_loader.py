# -*- coding: utf-8 -*-
"""
data_loader.py
通用数据加载模块
"""

import numpy as np
import scipy.io as sio


def load_dataset(config):
    """
    根据配置加载数据集

    Parameters:
    -----------
    config : DatasetConfig
        数据集配置对象

    Returns:
    --------
    data_dict : dict
        包含 'X' (EEG数据), 'y' (标签), 可选 'subject' (被试编号)
    """
    print(f"\n{'=' * 60}")
    print(f"📂 加载数据集: {config.name}")
    print(f"{'=' * 60}")

    # 加载MAT文件
    data = sio.loadmat(config.data_path)

    # 提取EEG数据
    EEGsample = data[config.data_key]

    # 提取标签
    labels = data[config.label_key].flatten()

    # 标签重映射（如果需要）
    if config.need_label_mapping:
        unique_labels = np.unique(labels)
        label_map = {old_val: new_val for new_val, old_val in enumerate(sorted(unique_labels))}
        labels_original = labels.copy()
        labels = np.array([label_map[val] for val in labels])
        print(f"  ✅ 标签已重映射: {dict(zip(unique_labels, range(len(unique_labels))))}")
    else:
        labels_original = None

    # 构建返回字典
    data_dict = {
        'X': EEGsample,
        'y': labels,
        'y_original': labels_original,
        'sfreq': config.sfreq,
        'n_channels': config.n_channels,
        'ch_names': config.ch_names
    }

    # 被试信息（如果有）
    if config.has_subject_index and config.subject_key in data:
        data_dict['subject'] = data[config.subject_key].flatten()
        print(f"  ✅ 包含被试信息: {len(np.unique(data_dict['subject']))} 名被试")

    # 打印数据集摘要
    print(f"\n  数据集信息:")
    print(f"    样本数: {EEGsample.shape[0]}")
    print(f"    通道数: {config.n_channels}")
    print(f"    采样点: {EEGsample.shape[2]} (时长: {EEGsample.shape[2] / config.sfreq:.1f}秒)")
    print(f"    类别数: {config.n_classes}")
    print(f"    标签分布: {dict(zip(*np.unique(labels, return_counts=True)))}")
    print(f"    类别名称: {config.class_names}")
    print(f"{'=' * 60}\n")

    return data_dict
