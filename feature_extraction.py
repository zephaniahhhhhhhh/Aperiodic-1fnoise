# -*- coding: utf-8 -*-
"""
feature_extraction.py
通用特征提取模块
"""

import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from specparam import SpectralModel

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from utils import (
    bandpower, sample_entropy, approximate_entropy,
    mutual_information, pair_time_features, pair_1f_diffs
)


def extract_traditional_features(EEGsample, config):
    """
    提取传统EEG特征（配置驱动）

    Parameters:
    -----------
    EEGsample : ndarray
        EEG数据 (n_samples, n_channels, n_timepoints)
    config : DatasetConfig
        数据集配置对象

    Returns:
    --------
    X : ndarray
        特征矩阵
    feature_info : dict
        特征信息
    """
    all_features = []

    for trial in EEGsample:
        feat_list = []

        # 1. 每通道9个特征
        for ch_data in trial:
            theta = bandpower(ch_data, config.sfreq, (4, 8))
            alpha = bandpower(ch_data, config.sfreq, (8, 13))
            ratio = theta / (alpha + 1e-6)
            se = sample_entropy(ch_data)
            ae = approximate_entropy(ch_data)
            skewness_val = skew(ch_data, bias=False, nan_policy='omit')
            kurtosis_val = kurtosis(ch_data, fisher=True, bias=False, nan_policy='omit')
            median_val = np.median(ch_data)
            max_val = np.max(ch_data)
            min_val = np.min(ch_data)
            fourth_moment = np.mean((ch_data - np.mean(ch_data)) ** 4)

            feat_list.extend([ratio, se, ae, skewness_val, kurtosis_val,
                              median_val, max_val, min_val, fourth_moment])

        # 2. 全局互信息统计（使用配置中的参考通道）
        ref = trial[config.ref_channel_idx]
        mi_vals = [mutual_information(ref, trial[j])
                   for j in range(config.n_channels) if j != config.ref_channel_idx]
        feat_list.extend([np.mean(mi_vals), np.std(mi_vals)])

        # 3. 通道对特征（使用配置中的通道对）
        if config.n_pairs > 0:
            feat_list.extend(pair_time_features(trial, config.pair_idx))

        all_features.append(feat_list)

    X = np.array(all_features, dtype=float)

    # 构建特征信息
    feature_info = {
        'total': X.shape[1],
        'per_channel': config.n_channels * 9,
        'global_stats': 2,
        'pair_features': 2 * config.n_pairs,
        'breakdown': {
            'channel_features': list(range(config.n_channels * 9)),
            'global_mi': list(range(config.n_channels * 9, config.n_channels * 9 + 2)),
            'pair_features': list(range(config.n_channels * 9 + 2, X.shape[1]))
        }
    }

    return X, feature_info


def extract_1f_features_single_band_enhanced(EEGsample, config, fmin=1, fmax=40):
    """
    提取1/f特征（配置驱动）

    Parameters:
    -----------
    EEGsample : ndarray
        EEG数据
    config : DatasetConfig
        数据集配置
    fmin, fmax : float
        频率范围

    Returns:
    --------
    X : ndarray
        特征矩阵
    feature_info : dict
        特征信息（包含索引、名称等）
    """
    all_features = []

    for trial in EEGsample:
        exps, offs = [], []
        per_chan_total_bw, per_chan_cf_range, per_chan_n_peaks = [], [], []

        # 通道对时域特征
        if config.n_pairs > 0:
            pair_time_feat = pair_time_features(trial, config.pair_idx)
        else:
            pair_time_feat = np.array([])

        for ch_data in trial:
            f, psd = welch(ch_data, fs=config.sfreq, nperseg=min(len(ch_data), config.sfreq * 2))
            mask = (f >= fmin) & (f <= fmax)
            sm = SpectralModel(aperiodic_mode='fixed', max_n_peaks=6, verbose=False)

            try:
                sm.fit(f[mask], psd[mask])
                offset, exponent = sm.get_params('aperiodic_params')
                peaks = sm.get_params('peak_params')
                peaks = np.atleast_2d(peaks) if np.ndim(peaks) == 1 else peaks
            except Exception:
                offset, exponent = np.nan, np.nan
                peaks = np.empty((0, 3))

            offs.append(offset)
            exps.append(exponent)

            if peaks.shape[0] == 0:
                per_chan_total_bw.append(0.0)
                per_chan_cf_range.append(0.0)
                per_chan_n_peaks.append(0)
            else:
                total_bw = float(np.sum(peaks[:, 2]))
                cf_range = float(np.max(peaks[:, 0]) - np.min(peaks[:, 0]))
                n_peaks = int(peaks.shape[0])
                per_chan_total_bw.append(total_bw)
                per_chan_cf_range.append(cf_range)
                per_chan_n_peaks.append(n_peaks)

        # 填充缺失值
        exps = np.nan_to_num(exps, nan=np.nanmean(exps))
        offs = np.nan_to_num(offs, nan=np.nanmean(offs))

        # 构建特征向量
        feat_basic = np.concatenate([
            exps, offs,
            [np.mean(exps), np.std(exps)],
            [np.mean(offs), np.std(offs)]
        ])

        tb = np.array(per_chan_total_bw, dtype=float)
        cr = np.array(per_chan_cf_range, dtype=float)
        npk = np.array(per_chan_n_peaks, dtype=float)
        feat_peakstats = np.array([
            float(np.mean(tb)), float(np.std(tb)),
            float(np.mean(cr)), float(np.std(cr)),
            float(np.mean(npk)), float(np.std(npk))
        ], dtype=float)

        # 通道对差异
        if config.n_pairs > 0:
            feat_pair_diffs = pair_1f_diffs(exps, offs, config.pair_idx)
        else:
            feat_pair_diffs = np.array([])

        feats = np.concatenate([feat_basic, feat_peakstats, feat_pair_diffs, pair_time_feat])
        all_features.append(feats)

    X = np.array(all_features, dtype=float)

    # 构建索引字典
    s = 0
    idx = {}
    idx['exps'] = slice(s, s + config.n_channels);
    s += config.n_channels
    idx['offs'] = slice(s, s + config.n_channels);
    s += config.n_channels
    idx['exps_stats'] = slice(s, s + 2);
    s += 2
    idx['offs_stats'] = slice(s, s + 2);
    s += 2
    idx['peakstats'] = slice(s, s + 6);
    s += 6
    idx['pair_diffs'] = slice(s, s + 2 * config.n_pairs);
    s += 2 * config.n_pairs
    idx['pair_time'] = slice(s, s + 2 * config.n_pairs);
    s += 2 * config.n_pairs

    # 构建特征名称
    feature_names = []
    for ch in config.ch_names:
        feature_names.append(f'exponent_{ch}')
    for ch in config.ch_names:
        feature_names.append(f'offset_{ch}')
    feature_names.extend(['exponent_mean', 'exponent_std', 'offset_mean', 'offset_std'])
    feature_names.extend(['peak_bw_mean', 'peak_bw_std', 'peak_cf_mean',
                          'peak_cf_std', 'peak_n_mean', 'peak_n_std'])

    if config.n_pairs > 0:
        for pair_name in config.pair_names:
            feature_names.append(f'exp_diff_{pair_name[0]}-{pair_name[1]}')
            feature_names.append(f'off_diff_{pair_name[0]}-{pair_name[1]}')
        for pair_name in config.pair_names:
            feature_names.append(f'pearson_{pair_name[0]}-{pair_name[1]}')
            feature_names.append(f'MI_{pair_name[0]}-{pair_name[1]}')

    # 特征信息汇总
    feature_info = {
        'total': X.shape[1],
        'indices': idx,
        'names': feature_names,
        'breakdown': {
            'channel_params': 2 * config.n_channels,
            'global_stats': 4,
            'peak_stats': 6,
            'pair_diffs': 2 * config.n_pairs,
            'pair_time': 2 * config.n_pairs
        },
        'freq_range': f'{fmin}-{fmax}Hz'
    }

    return X, feature_info


# 复合特征函数（保持不变）
def build_pca_cross_features(X_trad, X_onef, n_trad=10, n_onef=10, random_state=42):
    """PCA交叉特征"""
    X_trad = np.nan_to_num(X_trad, nan=0.0, posinf=0.0, neginf=0.0)
    X_onef = np.nan_to_num(X_onef, nan=0.0, posinf=0.0, neginf=0.0)

    scaler_trad = StandardScaler()
    scaler_onef = StandardScaler()
    trad_scaled = scaler_trad.fit_transform(X_trad)
    onef_scaled = scaler_onef.fit_transform(X_onef)

    pca_trad = PCA(n_components=min(n_trad, X_trad.shape[1]), random_state=random_state)
    pca_onef = PCA(n_components=min(n_onef, X_onef.shape[1]), random_state=random_state)
    T = pca_trad.fit_transform(trad_scaled)
    O = pca_onef.fit_transform(onef_scaled)

    T2 = T ** 2
    O2 = O ** 2
    cross = np.einsum('ni,nj->nij', T, O).reshape(T.shape[0], -1)

    X_comp = np.concatenate([T, O, T2, O2, cross], axis=1)
    return X_comp


def oof_probs(X, y, base_clf=None, n_splits=5, random_state=42):
    """Out-of-fold概率预测"""
    if base_clf is None:
        base_clf = SVC(kernel='rbf', probability=True, random_state=random_state)

    n_classes = len(np.unique(y))
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('clf', base_clf)
    ])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.zeros((X.shape[0], n_classes), dtype=float)

    for tr, te in skf.split(X, y):
        pipe.fit(X[tr], y[tr])
        oof[te, :] = pipe.predict_proba(X[te])

    return oof


def build_stacking_features(y, X_trad, X_onef_best, n_splits=5, random_state=42):
    """堆叠泛化特征"""
    p_trad = oof_probs(X_trad, y, n_splits=n_splits, random_state=random_state)
    p_onef = oof_probs(X_onef_best, y, n_splits=n_splits, random_state=random_state)

    n_classes = p_trad.shape[1]
    cross_prods = []
    for i in range(n_classes):
        for j in range(n_classes):
            cross_prods.append((p_trad[:, i] * p_onef[:, j]).reshape(-1, 1))

    X_meta = np.concatenate([p_trad, p_onef] + cross_prods, axis=1)
    return X_meta
