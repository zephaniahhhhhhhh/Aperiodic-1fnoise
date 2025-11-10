import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    precision_recall_fscore_support, roc_curve, auc
)
from scipy.stats import ttest_rel
import time
import warnings

warnings.filterwarnings('ignore')


class EnhancedModelEvaluator:
    """增强版模型评估器 - 支持多数据集、统计检验、可视化"""

    def __init__(self, n_splits=5, random_state=42, save_dir='./results'):
        self.n_splits = n_splits
        self.random_state = random_state
        self.save_dir = save_dir
        self.results = {}

        # 创建保存目录
        import os
        os.makedirs(save_dir, exist_ok=True)

    def create_pipeline(self, kernel='rbf', C=1.0, gamma='scale'):
        """创建包含预处理和分类器的Pipeline"""
        return Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('classifier', SVC(
                kernel=kernel, C=C, gamma=gamma,
                probability=True, random_state=self.random_state
            ))
        ])

    # ==================== 1. 单数据集评估（含每类别性能报告）====================
    def evaluate_single_dataset(self, X, y, class_names, method_name='Method'):
        """
        单数据集交叉验证评估
        返回：每类别的详细性能报告
        """
        print(f"\n{'=' * 60}")
        print(f"评估方法: {method_name}")
        print(f"{'=' * 60}")

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                              random_state=self.random_state)

        # 存储结果
        fold_results = {
            'accuracy': [],
            'auc': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'time_train': [],
            'time_test': []
        }

        all_y_true = []
        all_y_pred = []
        all_y_proba = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            print(f"\n处理 Fold {fold}/{self.n_splits}...")

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # 训练
            pipeline = self.create_pipeline()
            start_time = time.time()
            pipeline.fit(X_train, y_train)
            train_time = time.time() - start_time

            # 测试
            start_time = time.time()
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)
            test_time = time.time() - start_time

            # 计算指标
            acc = accuracy_score(y_test, y_pred)

            # 多分类AUC (one-vs-rest)
            if len(np.unique(y)) > 2:
                auc_score = roc_auc_score(y_test, y_proba,
                                          multi_class='ovr', average='macro')
            else:
                auc_score = roc_auc_score(y_test, y_proba[:, 1])

            # 每类别指标
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='macro'
            )

            # 保存结果
            fold_results['accuracy'].append(acc)
            fold_results['auc'].append(auc_score)
            fold_results['precision'].append(precision)
            fold_results['recall'].append(recall)
            fold_results['f1'].append(f1)
            fold_results['time_train'].append(train_time)
            fold_results['time_test'].append(test_time)

            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)
            all_y_proba.append(y_proba)

            print(f"  Accuracy: {acc:.4f}, AUC: {auc_score:.4f}")

        # ========== 汇总结果 ==========
        print(f"\n{'=' * 60}")
        print(f"交叉验证结果 ({self.n_splits} folds):")
        print(f"{'=' * 60}")
        print(f"Accuracy:  {np.mean(fold_results['accuracy']):.4f} ± {np.std(fold_results['accuracy']):.4f}")
        print(f"AUC:       {np.mean(fold_results['auc']):.4f} ± {np.std(fold_results['auc']):.4f}")
        print(f"Precision: {np.mean(fold_results['precision']):.4f} ± {np.std(fold_results['precision']):.4f}")
        print(f"Recall:    {np.mean(fold_results['recall']):.4f} ± {np.std(fold_results['recall']):.4f}")
        print(f"F1-Score:  {np.mean(fold_results['f1']):.4f} ± {np.std(fold_results['f1']):.4f}")
        print(f"\n时间统计:")
        print(f"训练时间:  {np.mean(fold_results['time_train']):.3f}s ± {np.std(fold_results['time_train']):.3f}s")
        print(f"测试时间:  {np.mean(fold_results['time_test']):.3f}s ± {np.std(fold_results['time_test']):.3f}s")

        # ========== 每类别性能报告 ==========
        print(f"\n{'=' * 60}")
        print(f"每类别性能报告:")
        print(f"{'=' * 60}")

        precisions, recalls, f1s, supports = precision_recall_fscore_support(
            all_y_true, all_y_pred, average=None
        )

        per_class_df = pd.DataFrame({
            'Class': class_names,
            'Precision': precisions,
            'Recall': recalls,
            'F1-Score': f1s,
            'Support': supports
        })
        print(per_class_df.to_string(index=False))

        # 保存到CSV
        per_class_df.to_csv(f'{self.save_dir}/{method_name}_per_class_report.csv',
                            index=False)

        # ========== 混淆矩阵 ==========
        self._plot_confusion_matrix(all_y_true, all_y_pred, class_names,
                                    f'{method_name}_confusion_matrix')

        # 保存结果
        self.results[method_name] = {
            'fold_results': fold_results,
            'per_class': per_class_df,
            'mean_metrics': {
                'accuracy': np.mean(fold_results['accuracy']),
                'auc': np.mean(fold_results['auc']),
                'precision': np.mean(fold_results['precision']),
                'recall': np.mean(fold_results['recall']),
                'f1': np.mean(fold_results['f1'])
            }
        }

        return fold_results

    # ==================== 2. 跨数据集验证 ====================
    def cross_dataset_validation(self, X_train_full, y_train, X_test_full, y_test,
                                 class_names, train_name='SAD', test_name='SEED'):
        """
        跨数据集验证：一个数据集训练，另一个数据集测试
        """
        print(f"\n{'=' * 60}")
        print(f"跨数据集验证: {train_name} → {test_name}")
        print(f"{'=' * 60}")

        # 训练
        pipeline = self.create_pipeline()
        start_time = time.time()
        pipeline.fit(X_train_full, y_train)
        train_time = time.time() - start_time

        # 测试
        start_time = time.time()
        y_pred = pipeline.predict(X_test_full)
        y_proba = pipeline.predict_proba(X_test_full)
        test_time = time.time() - start_time

        # 计算指标
        acc = accuracy_score(y_test, y_pred)

        if len(np.unique(y_test)) > 2:
            auc_score = roc_auc_score(y_test, y_proba,
                                      multi_class='ovr', average='macro')
        else:
            auc_score = roc_auc_score(y_test, y_proba[:, 1])

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='macro'
        )

        # 每类别报告
        precisions, recalls, f1s, supports = precision_recall_fscore_support(
            y_test, y_pred, average=None
        )

        per_class_df = pd.DataFrame({
            'Class': class_names,
            'Precision': precisions,
            'Recall': recalls,
            'F1-Score': f1s,
            'Support': supports
        })

        print(f"\n跨数据集测试结果:")
        print(f"Accuracy:  {acc:.4f}")
        print(f"AUC:       {auc_score:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"\n训练时间: {train_time:.3f}s")
        print(f"测试时间: {test_time:.3f}s")

        print(f"\n每类别性能:")
        print(per_class_df.to_string(index=False))

        # 保存结果
        method_name = f'CrossDataset_{train_name}_to_{test_name}'
        per_class_df.to_csv(f'{self.save_dir}/{method_name}_per_class_report.csv',
                            index=False)

        # 混淆矩阵
        self._plot_confusion_matrix(y_test, y_pred, class_names,
                                    f'{method_name}_confusion_matrix')

        return {
            'accuracy': acc,
            'auc': auc_score,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'per_class': per_class_df
        }

    # ==================== 3. 统计显著性检验 ====================
    def statistical_significance_test(self, baseline_method='Baseline',
                                      comparison_methods=None):
        """
        对不同方法进行配对t检验
        baseline_method: 基线方法名
        comparison_methods: 要比较的方法列表
        """
        print(f"\n{'=' * 60}")
        print(f"统计显著性检验 (配对t检验)")
        print(f"{'=' * 60}")

        if baseline_method not in self.results:
            print(f"错误: 基线方法 '{baseline_method}' 不存在")
            return

        baseline_aucs = self.results[baseline_method]['fold_results']['auc']
        baseline_accs = self.results[baseline_method]['fold_results']['accuracy']

        if comparison_methods is None:
            comparison_methods = [k for k in self.results.keys() if k != baseline_method]

        test_results = []

        for method in comparison_methods:
            if method not in self.results:
                print(f"警告: 方法 '{method}' 不存在，跳过")
                continue

            method_aucs = self.results[method]['fold_results']['auc']
            method_accs = self.results[method]['fold_results']['accuracy']

            # AUC的t检验
            t_stat_auc, p_val_auc = ttest_rel(baseline_aucs, method_aucs)

            # Accuracy的t检验
            t_stat_acc, p_val_acc = ttest_rel(baseline_accs, method_accs)

            test_results.append({
                'Comparison': f'{baseline_method} vs. {method}',
                'AUC_t_stat': t_stat_auc,
                'AUC_p_value': p_val_auc,
                'AUC_significant': '***' if p_val_auc < 0.001 else '**' if p_val_auc < 0.01 else '*' if p_val_auc < 0.05 else 'ns',
                'ACC_t_stat': t_stat_acc,
                'ACC_p_value': p_val_acc,
                'ACC_significant': '***' if p_val_acc < 0.001 else '**' if p_val_acc < 0.01 else '*' if p_val_acc < 0.05 else 'ns'
            })

            print(f"\n{baseline_method} vs. {method}:")
            print(f"  AUC: t={t_stat_auc:.3f}, p={p_val_auc:.4f} {test_results[-1]['AUC_significant']}")
            print(f"  ACC: t={t_stat_acc:.3f}, p={p_val_acc:.4f} {test_results[-1]['ACC_significant']}")

        # 保存结果
        test_df = pd.DataFrame(test_results)
        test_df.to_csv(f'{self.save_dir}/statistical_significance_test.csv', index=False)
        print(f"\n显著性: *** p<0.001, ** p<0.01, * p<0.05, ns=not significant")

        return test_df

    # ==================== 4. 特征重要性分析 ====================
    def feature_importance_analysis(self, X, y, feature_names, method_name='Method',
                                    top_k=20):
        """
        使用线性SVM分析特征重要性
        """
        print(f"\n{'=' * 60}")
        print(f"特征重要性分析: {method_name}")
        print(f"{'=' * 60}")

        # 使用线性SVM
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('classifier', SVC(kernel='linear', random_state=self.random_state))
        ])

        pipeline.fit(X, y)

        # 获取系数
        coef = pipeline.named_steps['classifier'].coef_

        if coef.shape[0] > 1:  # 多分类
            # 取所有类别系数的绝对值平均
            importance = np.abs(coef).mean(axis=0)
        else:  # 二分类
            importance = np.abs(coef[0])

        # 排序
        indices = np.argsort(importance)[::-1][:top_k]

        importance_df = pd.DataFrame({
            'Feature': [feature_names[i] for i in indices],
            'Importance': importance[indices]
        })

        print(f"\nTop {top_k} 重要特征:")
        print(importance_df.to_string(index=False))

        # 保存
        importance_df.to_csv(f'{self.save_dir}/{method_name}_feature_importance.csv',
                             index=False)

        # 可视化
        plt.figure(figsize=(10, 6))
        plt.barh(range(top_k), importance[indices][::-1])
        plt.yticks(range(top_k), [feature_names[i] for i in indices[::-1]])
        plt.xlabel('Importance')
        plt.title(f'Top {top_k} Feature Importance - {method_name}')
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/{method_name}_feature_importance.png', dpi=300)
        plt.close()

        return importance_df

    # ==================== 5. ROC曲线对比 ====================
    def plot_roc_curves_comparison(self, X, y, methods_configs, class_names):
        """
        在同一张图上绘制多个方法的ROC曲线
        methods_configs: list of dict, e.g. [{'name': 'Baseline', 'X': X_baseline}, ...]
        """
        print(f"\n{'=' * 60}")
        print(f"绘制ROC曲线对比图")
        print(f"{'=' * 60}")

        n_classes = len(np.unique(y))

        if n_classes == 2:
            # 二分类ROC
            plt.figure(figsize=(10, 8))

            for config in methods_configs:
                method_name = config['name']
                X_method = config['X']

                # 使用交叉验证获取预测概率
                skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                                      random_state=self.random_state)

                tprs = []
                aucs = []
                mean_fpr = np.linspace(0, 1, 100)

                for train_idx, test_idx in skf.split(X_method, y):
                    pipeline = self.create_pipeline()
                    pipeline.fit(X_method[train_idx], y[train_idx])
                    y_proba = pipeline.predict_proba(X_method[test_idx])[:, 1]

                    fpr, tpr, _ = roc_curve(y[test_idx], y_proba)
                    tprs.append(np.interp(mean_fpr, fpr, tpr))
                    aucs.append(auc(fpr, tpr))

                mean_tpr = np.mean(tprs, axis=0)
                mean_auc = np.mean(aucs)
                std_auc = np.std(aucs)

                plt.plot(mean_fpr, mean_tpr,
                         label=f'{method_name} (AUC = {mean_auc:.3f} ± {std_auc:.3f})',
                         linewidth=2)

            plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title('ROC Curve Comparison', fontsize=14, fontweight='bold')
            plt.legend(loc="lower right", fontsize=10)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{self.save_dir}/roc_curves_comparison.png', dpi=300)
            plt.close()

        else:
            # 多分类ROC (One-vs-Rest)
            fig, axes = plt.subplots(1, n_classes, figsize=(6 * n_classes, 5))
            if n_classes == 1:
                axes = [axes]

            for class_idx in range(n_classes):
                ax = axes[class_idx]

                for config in methods_configs:
                    method_name = config['name']
                    X_method = config['X']

                    # 二值化标签
                    y_binary = (y == class_idx).astype(int)

                    skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                                          random_state=self.random_state)

                    tprs = []
                    aucs = []
                    mean_fpr = np.linspace(0, 1, 100)

                    for train_idx, test_idx in skf.split(X_method, y):
                        pipeline = self.create_pipeline()
                        pipeline.fit(X_method[train_idx], y[train_idx])
                        y_proba = pipeline.predict_proba(X_method[test_idx])[:, class_idx]

                        fpr, tpr, _ = roc_curve(y_binary[test_idx], y_proba)
                        tprs.append(np.interp(mean_fpr, fpr, tpr))
                        aucs.append(auc(fpr, tpr))

                    mean_tpr = np.mean(tprs, axis=0)
                    mean_auc = np.mean(aucs)
                    std_auc = np.std(aucs)

                    ax.plot(mean_fpr, mean_tpr,
                            label=f'{method_name} (AUC = {mean_auc:.3f})',
                            linewidth=2)

                ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title(f'ROC - {class_names[class_idx]}')
                ax.legend(loc="lower right", fontsize=8)
                ax.grid(alpha=0.3)

            plt.tight_layout()
            plt.savefig(f'{self.save_dir}/roc_curves_comparison_multiclass.png', dpi=300)
            plt.close()

        print(f"ROC曲线已保存到: {self.save_dir}/")

    # ==================== 6. 被试级分析（Leave-One-Subject-Out）====================
    def leave_one_subject_out_analysis(self, X, y, subject_indices, class_names,
                                       method_name='Method'):
        """
        被试级交叉验证分析
        subject_indices: array of subject IDs for each sample
        """
        print(f"\n{'=' * 60}")
        print(f"被试级分析 (Leave-One-Subject-Out): {method_name}")
        print(f"{'=' * 60}")

        unique_subjects = np.unique(subject_indices)
        n_subjects = len(unique_subjects)

        print(f"总被试数: {n_subjects}")

        subject_results = []
        all_y_true = []
        all_y_pred = []

        for subject_id in unique_subjects:
            # 留一法
            test_mask = (subject_indices == subject_id)
            train_mask = ~test_mask

            X_train = X[train_mask]
            y_train = y[train_mask]
            X_test = X[test_mask]
            y_test = y[test_mask]

            # 训练和测试
            pipeline = self.create_pipeline()
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            subject_results.append({
                'Subject': subject_id,
                'Accuracy': acc,
                'N_samples': len(y_test)
            })

            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)

            print(f"Subject {subject_id}: Accuracy = {acc:.4f} (n={len(y_test)})")

        # 汇总
        mean_acc = np.mean([r['Accuracy'] for r in subject_results])
        std_acc = np.std([r['Accuracy'] for r in subject_results])
        overall_acc = accuracy_score(all_y_true, all_y_pred)

        print(f"\n被试级平均准确率: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"总体准确率: {overall_acc:.4f}")

        # 保存结果
        subject_df = pd.DataFrame(subject_results)
        subject_df.to_csv(f'{self.save_dir}/{method_name}_subject_level_analysis.csv',
                          index=False)

        # 可视化被试间差异
        plt.figure(figsize=(12, 5))
        plt.bar(range(n_subjects), [r['Accuracy'] for r in subject_results])
        plt.axhline(mean_acc, color='r', linestyle='--', label=f'Mean = {mean_acc:.3f}')
        plt.xlabel('Subject')
        plt.ylabel('Accuracy')
        plt.title(f'Per-Subject Accuracy - {method_name}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/{method_name}_subject_level_accuracy.png', dpi=300)
        plt.close()

        return subject_df

    # ==================== 辅助函数 ====================
    def _plot_confusion_matrix(self, y_true, y_pred, class_names, save_name):
        """绘制混淆矩阵（原始和归一化）"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 原始混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=axes[0])
        axes[0].set_title('Confusion Matrix (Counts)')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')

        # 归一化混淆矩阵
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=axes[1])
        axes[1].set_title('Confusion Matrix (Normalized)')
        axes[1].set_ylabel('True Label')
        axes[1].set_xlabel('Predicted Label')

        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/{save_name}.png', dpi=300)
        plt.close()

    def generate_summary_report(self):
        """生成所有方法的汇总报告"""
        print(f"\n{'=' * 60}")
        print(f"汇总报告")
        print(f"{'=' * 60}")

        summary_data = []
        for method_name, results in self.results.items():
            if 'mean_metrics' in results:
                metrics = results['mean_metrics']
                summary_data.append({
                    'Method': method_name,
                    'Accuracy': f"{metrics['accuracy']:.4f}",
                    'AUC': f"{metrics['auc']:.4f}",
                    'Precision': f"{metrics['precision']:.4f}",
                    'Recall': f"{metrics['recall']:.4f}",
                    'F1-Score': f"{metrics['f1']:.4f}"
                })

        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))

        # 保存
        summary_df.to_csv(f'{self.save_dir}/summary_report.csv', index=False)
        print(f"\n所有结果已保存到: {self.save_dir}/")

        return summary_df


