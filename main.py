# -*- coding: utf-8 -*-
"""
main.py
通用主程序 - 支持多数据集
"""

import numpy as np
import pandas as pd
from datetime import datetime
import argparse

from config import get_config, STANDARD_BANDS
from data_loader import load_dataset
from feature_extraction import (
    extract_traditional_features,
    extract_1f_features_single_band_enhanced,
    build_pca_cross_features,
    build_stacking_features
)
from evaluation import evaluate_with_cv


# from visualization import plot_feature_dimensions, ...  # 可选

def main(dataset_name='SAD'):
    """
    主实验流程

    Parameters:
    -----------
    dataset_name : str
        数据集名称 ('SAD', 'SEED', 或自定义)
    """
    # 1. 加载配置
    config = get_config(dataset_name)
    config.get_info()

    # 2. 加载数据
    data_dict = load_dataset(config)
    EEGsample = data_dict['X']
    labels = data_dict['y']

    # 3. 提取传统特征
    print("\n🚀 提取传统 EEG 特征...")
    X_trad, feature_info_trad = extract_traditional_features(EEGsample, config)
    print(f"✅ 传统特征完成，形状：{X_trad.shape}")

    # 4. 提取1/f特征（主频段）
    print("\n🚀 提取1/f特征 (1-40Hz)...")
    X_1f_main, feature_info_1f = extract_1f_features_single_band_enhanced(
        EEGsample, config, fmin=1, fmax=40
    )
    print(f"✅ 1/f特征完成，形状：{X_1f_main.shape}")

    # 5. 实验：多频段对比
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_rows = []

    print(f"\n{'=' * 60}")
    print(f"📊 多频段实验开始 ({config.n_classes}分类)")
    print(f"{'=' * 60}")

    # 基线：传统特征
    print("\n--- 基线：传统EEG特征 ---")
    summary, _, _, _ = evaluate_with_cv(
        X_trad, labels, config.n_classes,
        name=f"{dataset_name}_传统EEG特征",
        fold_tag_prefix="Baseline_"
    )
    auc_m, auc_s, acc_m, acc_s = summary
    results_rows.append({
        "Dataset": dataset_name,
        "Band": "Baseline",
        "Group": "传统EEG特征",
        "AUC_mean": auc_m,
        "AUC_std": auc_s,
        "Acc_mean": acc_m,
        "Acc_std": acc_s,
        "n_features": X_trad.shape[1]
    })

    # 遍历各频段
    for label, fmin, fmax in STANDARD_BANDS:
        print(f"\n{'=' * 60}")
        print(f"--- 频段 {label} ---")
        print(f"{'=' * 60}")

        X_1f_band, _ = extract_1f_features_single_band_enhanced(
            EEGsample, config, fmin=fmin, fmax=fmax
        )

        # 仅1/f
        summary, _, _, _ = evaluate_with_cv(
            X_1f_band, labels, config.n_classes,
            name=f"{dataset_name}_仅1/f({label})",
            fold_tag_prefix=f"1f_{label}_"
        )
        auc_m, auc_s, acc_m, acc_s = summary
        results_rows.append({
            "Dataset": dataset_name,
            "Band": label,
            "Group": "仅1/f",
            "AUC_mean": auc_m,
            "AUC_std": auc_s,
            "Acc_mean": acc_m,
            "Acc_std": acc_s,
            "n_features": X_1f_band.shape[1]
        })

        # 融合
        X_fused = np.hstack([X_1f_band, X_trad])
        summary, _, _, _ = evaluate_with_cv(
            X_fused, labels, config.n_classes,
            name=f"{dataset_name}_融合({label})",
            fold_tag_prefix=f"fused_{label}_"
        )
        auc_m, auc_s, acc_m, acc_s = summary
        results_rows.append({
            "Dataset": dataset_name,
            "Band": label,
            "Group": "融合",
            "AUC_mean": auc_m,
            "AUC_std": auc_s,
            "Acc_mean": acc_m,
            "Acc_std": acc_s,
            "n_features": X_fused.shape[1]
        })

    # 6. 保存结果
    df_summary = pd.DataFrame(results_rows)
    output_file = config.output_dir / f"results_{dataset_name}_{ts}.csv"
    df_summary.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 结果已保存: {output_file}")
    print("\n📊 结果摘要：")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='通用EEG疲劳检测实验')
    parser.add_argument('--dataset', type=str, default='SAD',
                        choices=['SAD', 'SEED', 'all'],  # 新增 'all'
                        help='数据集名称（all=运行全部）')

    args = parser.parse_args()

    if args.dataset == 'all':
        # 运行所有数据集
        for dataset_name in ['SAD', 'SEED']:
            print(f"\n{'#' * 60}")
            print(f"# 开始运行数据集: {dataset_name}")
            print(f"{'#' * 60}\n")
            main(dataset_name=dataset_name)
    else:
        # 运行指定数据集
        main(dataset_name=args.dataset)
