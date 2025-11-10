# -*- coding: utf-8 -*-
"""
main.py
通用主程序 - 支持单数据集和跨数据集自动顺序执行
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
from evaluation import EnhancedModelEvaluator  # ✅ 正确的导入


def run_single_dataset_experiment(dataset_name='SAD'):
    """
    单数据集实验流程

    Parameters:
    -----------
    dataset_name : str
        数据集名称 ('SAD', 'SEED', 或自定义)

    Returns:
    --------
    dict: 包含特征和标签的字典，用于后续跨数据集验证
    """
    print(f"\n{'#' * 60}")
    print(f"# 单数据集实验: {dataset_name}")
    print(f"{'#' * 60}\n")

    # ==================== 1. 加载配置和数据 ====================
    config = get_config(dataset_name)
    config.get_info()

    data_dict = load_dataset(config)
    EEGsample = data_dict['X']
    labels = data_dict['y']

    print(f"\n数据加载完成: {len(labels)} samples, {config.n_classes} classes")

    # ==================== 2. 初始化评估器 ====================
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = config.output_dir / f"single_dataset_{dataset_name}_{ts}"

    evaluator = EnhancedModelEvaluator(
        n_splits=5,
        random_state=42,
        save_dir=str(save_dir)
    )
    print(f"✅ 评估器初始化完成，结果保存到: {save_dir}")

    # ==================== 3. 提取传统特征 ====================
    print("\n🚀 提取传统 EEG 特征...")
    X_trad, feature_info_trad = extract_traditional_features(EEGsample, config)
    print(f"✅ 传统特征完成，形状：{X_trad.shape}")

    # ==================== 4. 提取1/f特征（主频段）====================
    print("\n🚀 提取1/f特征 (1-40Hz)...")
    X_1f_main, feature_info_1f = extract_1f_features_single_band_enhanced(
        EEGsample, config, fmin=1, fmax=40
    )
    print(f"✅ 1/f特征完成，形状：{X_1f_main.shape}")

    # ==================== 5. 基线评估 ====================
    print(f"\n{'=' * 60}")
    print(f"📊 基线评估：传统EEG特征")
    print(f"{'=' * 60}")

    class_names = config.class_names

    baseline_results = evaluator.evaluate_single_dataset(
        X_trad, labels,
        class_names=class_names,
        method_name='Baseline_Traditional'
    )

    # ==================== 6. 多频段实验 ====================
    print(f"\n{'=' * 60}")
    print(f"📊 多频段1/f特征评估")
    print(f"{'=' * 60}")

    roc_methods_configs = [
        {'name': 'Baseline (Traditional)', 'X': X_trad}
    ]

    for label, fmin, fmax in STANDARD_BANDS:
        print(f"\n{'=' * 60}")
        print(f"--- 频段: {label} ({fmin}-{fmax} Hz) ---")
        print(f"{'=' * 60}")

        X_1f_band, _ = extract_1f_features_single_band_enhanced(
            EEGsample, config, fmin=fmin, fmax=fmax
        )

        method_name_1f = f"1f_only_{label}"
        evaluator.evaluate_single_dataset(
            X_1f_band, labels,
            class_names=class_names,
            method_name=method_name_1f
        )

        X_fused = np.hstack([X_1f_band, X_trad])
        method_name_fused = f"Fused_{label}"
        evaluator.evaluate_single_dataset(
            X_fused, labels,
            class_names=class_names,
            method_name=method_name_fused
        )

        roc_methods_configs.append({
            'name': f'{label} (Fused)',
            'X': X_fused
        })

    # ==================== 7. 统计显著性检验 ====================
    print(f"\n{'=' * 60}")
    print(f"📊 统计显著性检验")
    print(f"{'=' * 60}")

    comparison_methods = [k for k in evaluator.results.keys()
                          if k != 'Baseline_Traditional']

    if len(comparison_methods) > 0:
        significance_results = evaluator.statistical_significance_test(
            baseline_method='Baseline_Traditional',
            comparison_methods=comparison_methods
        )

    # ==================== 8. ROC曲线对比 ====================
    print(f"\n{'=' * 60}")
    print(f"📊 ROC曲线对比（所有方法）")
    print(f"{'=' * 60}")

    evaluator.plot_roc_curves_comparison(
        X_trad, labels,
        roc_methods_configs,
        class_names=class_names
    )

    # ==================== 9. 特征重要性分析 ====================
    if X_1f_main.shape[1] <= 500:
        print(f"\n{'=' * 60}")
        print(f"📊 特征重要性分析（1/f主频段）")
        print(f"{'=' * 60}")

        feature_names_1f = feature_info_1f['names']

        evaluator.feature_importance_analysis(
            X_1f_main, labels,
            feature_names=feature_names_1f,
            method_name='1f_main_band',
            top_k=min(20, len(feature_names_1f))
        )

    # ==================== 10. 被试级分析 ====================
    if 'subject' in data_dict:
        print(f"\n{'=' * 60}")
        print(f"📊 被试级分析（Leave-One-Subject-Out）")
        print(f"{'=' * 60}")

        subject_indices = data_dict['subject']

        subject_results = evaluator.leave_one_subject_out_analysis(
            X_1f_main, labels, subject_indices,
            class_names=class_names,
            method_name='1f_main_LOSO'
        )

        print(f"✅ 被试级分析完成")

    # ==================== 11. 生成汇总报告 ====================
    print(f"\n{'=' * 60}")
    print(f"📊 生成汇总报告")
    print(f"{'=' * 60}")

    summary_df = evaluator.generate_summary_report()

    # ==================== 12. 保存传统格式结果 ====================
    results_rows = []

    for method_name, results in evaluator.results.items():
        if 'mean_metrics' in results:
            metrics = results['mean_metrics']

            if method_name == 'Baseline_Traditional':
                band = 'Baseline'
                group = '传统EEG特征'
            elif method_name.startswith('1f_only_'):
                band = method_name.replace('1f_only_', '')
                group = '仅1/f'
            elif method_name.startswith('Fused_'):
                band = method_name.replace('Fused_', '')
                group = '融合'
            else:
                band = 'Other'
                group = method_name

            results_rows.append({
                "Dataset": dataset_name,
                "Band": band,
                "Group": group,
                "AUC_mean": f"{metrics['auc']:.4f}",
                "Acc_mean": f"{metrics['accuracy']:.4f}",
                "Precision": f"{metrics['precision']:.4f}",
                "Recall": f"{metrics['recall']:.4f}",
                "F1-Score": f"{metrics['f1']:.4f}"
            })

    df_legacy = pd.DataFrame(results_rows)
    legacy_file = save_dir / f"legacy_results_{dataset_name}.csv"
    df_legacy.to_csv(legacy_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 传统格式结果已保存: {legacy_file}")

    print(f"\n{'=' * 60}")
    print(f"✅ {dataset_name} 单数据集评估完成！")
    print(f"{'=' * 60}")
    print(f"📁 所有结果保存在: {save_dir}")

    # 返回数据供跨数据集使用
    return {
        'dataset_name': dataset_name,
        'config': config,
        'EEGsample': EEGsample,
        'labels': labels,
        'X_trad': X_trad,
        'X_1f_main': X_1f_main,
        'class_names': class_names
    }


def run_cross_dataset_experiment(train_data, test_data):
    """
    跨数据集验证实验

    Parameters:
    -----------
    train_data : dict
        训练数据集信息（来自run_single_dataset_experiment）
    test_data : dict
        测试数据集信息（来自run_single_dataset_experiment）
    """
    train_dataset = train_data['dataset_name']
    test_dataset = test_data['dataset_name']

    print(f"\n{'#' * 60}")
    print(f"# 跨数据集验证: {train_dataset} → {test_dataset}")
    print(f"{'#' * 60}\n")

    config_train = train_data['config']
    config_test = test_data['config']

    # 确保类别数一致
    if config_train.n_classes != config_test.n_classes:
        print(f"⚠️ 警告: 类别数不一致 ({config_train.n_classes} vs {config_test.n_classes})")
        print("跨数据集验证需要相同的类别数，跳过此验证")
        return

    # 初始化评估器
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = config_train.output_dir / f"cross_dataset_{train_dataset}_to_{test_dataset}_{ts}"

    evaluator = EnhancedModelEvaluator(
        n_splits=5,
        random_state=42,
        save_dir=str(save_dir)
    )

    # 提取训练集特征
    print("\n🚀 提取训练集特征...")
    X_train_trad, _ = extract_traditional_features(
        train_data['EEGsample'], config_train
    )
    X_train_1f, _ = extract_1f_features_single_band_enhanced(
        train_data['EEGsample'], config_train, fmin=1, fmax=40
    )
    X_train_fused = np.hstack([X_train_1f, X_train_trad])

    # 提取测试集特征
    print("🚀 提取测试集特征...")
    X_test_trad, _ = extract_traditional_features(
        test_data['EEGsample'], config_test
    )
    X_test_1f, _ = extract_1f_features_single_band_enhanced(
        test_data['EEGsample'], config_test, fmin=1, fmax=40
    )
    X_test_fused = np.hstack([X_test_1f, X_test_trad])

    y_train = train_data['labels']
    y_test = test_data['labels']
    class_names = train_data['class_names']

    # 跨数据集验证
    print(f"\n{'=' * 60}")
    print(f"📊 跨数据集验证")
    print(f"{'=' * 60}")

    # 传统特征
    print("\n--- 传统特征 ---")
    evaluator.cross_dataset_validation(
        X_train_trad, y_train, X_test_trad, y_test,
        class_names, train_name=train_dataset,
        test_name=test_dataset + '_Traditional'
    )

    # 1/f特征
    print("\n--- 1/f特征 ---")
    evaluator.cross_dataset_validation(
        X_train_1f, y_train, X_test_1f, y_test,
        class_names, train_name=train_dataset,
        test_name=test_dataset + '_1f'
    )

    # 融合特征
    print("\n--- 融合特征 ---")
    evaluator.cross_dataset_validation(
        X_train_fused, y_train, X_test_fused, y_test,
        class_names, train_name=train_dataset,
        test_name=test_dataset + '_Fused'
    )

    print(f"\n{'=' * 60}")
    print(f"✅ {train_dataset} → {test_dataset} 跨数据集验证完成！")
    print(f"{'=' * 60}")
    print(f"📁 结果保存在: {save_dir}")


def run_complete_pipeline(datasets=['SAD', 'SEED']):
    """
    完整实验流程：先单数据集，再跨数据集

    Parameters:
    -----------
    datasets : list
        要运行的数据集列表
    """
    print(f"\n{'#' * 80}")
    print(f"# 开始完整实验流程")
    print(f"# 数据集: {', '.join(datasets)}")
    print(f"{'#' * 80}\n")

    # ==================== 阶段1: 单数据集实验 ====================
    print(f"\n{'=' * 80}")
    print(f"阶段 1/2: 单数据集评估")
    print(f"{'=' * 80}\n")

    dataset_results = {}

    for dataset_name in datasets:
        data_info = run_single_dataset_experiment(dataset_name)
        dataset_results[dataset_name] = data_info

    # ==================== 阶段2: 跨数据集验证 ====================
    if len(datasets) >= 2:
        print(f"\n{'=' * 80}")
        print(f"阶段 2/2: 跨数据集验证")
        print(f"{'=' * 80}\n")

        # 进行所有可能的跨数据集验证
        for i, train_dataset in enumerate(datasets):
            for j, test_dataset in enumerate(datasets):
                if i != j:  # 不同数据集之间
                    run_cross_dataset_experiment(
                        dataset_results[train_dataset],
                        dataset_results[test_dataset]
                    )
    else:
        print(f"\n⚠️ 只有一个数据集，跳过跨数据集验证")

    # ==================== 总结 ====================
    print(f"\n{'#' * 80}")
    print(f"✅ 完整实验流程完成！")
    print(f"{'#' * 80}")
    print(f"\n已完成的实验:")
    print(f"  - 单数据集评估: {len(datasets)} 个数据集")
    if len(datasets) >= 2:
        n_cross = len(datasets) * (len(datasets) - 1)
        print(f"  - 跨数据集验证: {n_cross} 个组合")
    print(f"\n所有结果已保存到各自的目录中")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='通用EEG疲劳检测实验（增强版）')
    parser.add_argument('--dataset', type=str, default='SAD',
                        choices=['SAD', 'SEED', 'all'],
                        help='数据集名称（all=运行全部并自动跨数据集验证）')
    parser.add_argument('--mode', type=str, default='auto',
                        choices=['single', 'cross', 'auto'],
                        help='实验模式: single=仅单数据集, cross=仅跨数据集, auto=自动完整流程（默认）')
    parser.add_argument('--train_dataset', type=str, default='SAD',
                        help='跨数据集实验的训练集（仅在mode=cross时使用）')
    parser.add_argument('--test_dataset', type=str, default='SEED',
                        help='跨数据集实验的测试集（仅在mode=cross时使用）')

    args = parser.parse_args()

    if args.mode == 'auto':
        # ✅ 自动完整流程（推荐）
        if args.dataset == 'all':
            run_complete_pipeline(['SAD', 'SEED'])
        else:
            run_complete_pipeline([args.dataset])

    elif args.mode == 'single':
        # 仅单数据集实验
        if args.dataset == 'all':
            for dataset_name in ['SAD', 'SEED']:
                run_single_dataset_experiment(dataset_name)
        else:
            run_single_dataset_experiment(args.dataset)

    elif args.mode == 'cross':
        # 仅跨数据集验证
        train_data = run_single_dataset_experiment(args.train_dataset)
        test_data = run_single_dataset_experiment(args.test_dataset)
        run_cross_dataset_experiment(train_data, test_data)
