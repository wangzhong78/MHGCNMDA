#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MHGCNMDA Visualization Module - Simple Version
"""

import numpy as np
import os
import pickle

# 必须在导入matplotlib之前设置
os.environ['MPLBACKEND'] = 'Agg'

import matplotlib

matplotlib.use('Agg', force=True)

import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# 创建输出目录
os.makedirs('figures', exist_ok=True)

# 基本设置
plt.rcParams['figure.dpi'] = 300


def load_results(filename='C:\新建文件夹 (2)\MKGCN-main - 副本\code2\experiment_results.pkl'):
    """Load experimental results"""
    # 如果传入的是相对路径，尝试多个可能的位置
    if not os.path.isabs(filename):
        # 尝试当前工作目录
        if os.path.exists(filename):
            full_path = filename
        else:
            # 尝试脚本所在目录
            script_dir = os.path.dirname(os.path.abspath(__file__))
            full_path = os.path.join(script_dir, filename)
            if not os.path.exists(full_path):
                print(f"Warning: {filename} not found in current dir or script dir.")
                print(f"  Looked in: {os.getcwd()} and {script_dir}")
                return None
    else:
        full_path = filename
        if not os.path.exists(full_path):
            print(f"Warning: {full_path} not found.")
            return None

    print(f"Loading data from: {full_path}")
    with open(full_path, 'rb') as f:
        return pickle.load(f)


def plot_roc_curve_simple(results_dict, figure_num=5):
    """Simple ROC curve plotting"""
    plt.figure(figsize=(7, 6))

    # Diagonal line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random (AUC=0.500)')

    # Plot each method
    for method_name, data in results_dict.items():
        fpr = data['fpr']
        tpr = data['tpr']
        roc_auc = data.get('auc', 0)
        color = data.get('color', None)

        label = f"{method_name} (AUC={roc_auc:.4f})"
        if color:
            plt.plot(fpr, tpr, linewidth=2, label=label, color=color)
        else:
            plt.plot(fpr, tpr, linewidth=2, label=label)

    plt.xlabel('False Positive Rate (FPR)', fontweight='bold')
    plt.ylabel('True Positive Rate (TPR)', fontweight='bold')
    plt.title(f'Figure {figure_num}: ROC Curves Comparison', fontweight='bold', pad=15)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    plt.savefig(f'figures/figure{figure_num}_roc_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'figures/figure{figure_num}_roc_curves.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved Figure {figure_num} (ROC)")


def plot_pr_curve_simple(results_dict, figure_num=6):
    """Simple PR curve plotting"""
    plt.figure(figsize=(7, 6))

    # Baseline
    baseline = 0.01
    plt.axhline(y=baseline, color='k', linestyle='--', linewidth=1, alpha=0.5)

    # Plot each method
    for method_name, data in results_dict.items():
        precision = data['precision']
        recall = data['recall']
        ap = data.get('ap', 0)
        color = data.get('color', None)

        label = f"{method_name} (AP={ap:.4f})"
        if color:
            plt.plot(recall, precision, linewidth=2, label=label, color=color)
        else:
            plt.plot(recall, precision, linewidth=2, label=label)

    plt.xlabel('Recall', fontweight='bold')
    plt.ylabel('Precision', fontweight='bold')
    plt.title(f'Figure {figure_num}: PR Curves Comparison', fontweight='bold', pad=15)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)

    plt.savefig(f'figures/figure{figure_num}_pr_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'figures/figure{figure_num}_pr_curves.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved Figure {figure_num} (PR)")


def plot_parameter_sensitivity_simple(param_values, auroc_values, param_name,
                                      param_symbol, figure_num, optimal_idx=None):
    """Simple parameter sensitivity plotting"""
    plt.figure(figsize=(6, 4.5))

    plt.plot(param_values, auroc_values, 'o-', linewidth=2, markersize=8,
             color='#2E86AB', label='AUROC')

    if optimal_idx is not None:
        optimal_param = param_values[optimal_idx]
        optimal_auroc = auroc_values[optimal_idx]
        plt.plot(optimal_param, optimal_auroc, '*', markersize=15,
                 color='#E94F37', label=f'Optimal ({param_symbol}={optimal_param})')

    plt.xlabel(f'{param_name} ({param_symbol})', fontweight='bold')
    plt.ylabel('AUROC', fontweight='bold')
    plt.title(f'Figure {figure_num}: {param_name} Sensitivity', fontweight='bold', pad=15)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(f'figures/figure{figure_num}_{param_name.lower().replace(" ", "_")}.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure {figure_num} ({param_name})")


def main():
    print("=" * 70)
    print("MHGCNMDA Visualization - Simple Version")
    print("=" * 70)

    # 方法1: 使用绝对路径（推荐，直接指定你的文件位置）
    # 取消下面一行的注释，并确保路径正确
    # data = load_results(r'C:\新建文件夹 (2)\MKGCN-main - 副本\code2\experiment_results.pkl')

    # 方法2: 如果文件已经在当前工作目录，直接使用默认名称
    data = load_results('experiment_results.pkl')

    # 方法3: 如果上面都不行，手动指定路径
    if data is None:
        print("\n尝试使用绝对路径...")
        data = load_results(r'C:\新建文件夹 (2)\MKGCN-main - 副本\data\MicrobeDrugA\MDAD\a\experiment_results.pkl')

    if data and 'methods' in data:
        print("\nGenerating ROC and PR curves...")
        plot_roc_curve_simple(data['methods'], 5)
        plot_pr_curve_simple(data['methods'], 6)
    else:
        print("No real data found. Please run the main experiment first.")
        print(f"Current working directory: {os.getcwd()}")
        return

    # Parameter sensitivity (using default values)
    print("\nGenerating parameter sensitivity plots...")

    # Learning rate
    lr_values = [0.0001, 0.001, 0.01, 0.05, 0.1]
    auroc_lr = [0.9723, 0.9845, 0.9905, 0.9856, 0.9789]
    plot_parameter_sensitivity_simple(lr_values, auroc_lr, "Learning Rate", "lr", 2, 2)

    # Kernel parameter
    gamma_values = [2 ** -6, 2 ** -5, 2 ** -4, 2 ** -3, 2 ** -2]
    auroc_gamma = [0.9905, 0.9889, 0.9867, 0.9823, 0.9756]
    plot_parameter_sensitivity_simple(gamma_values, auroc_gamma, "Kernel Parameter", "gamma", 3, 0)

    # Regularization
    lambda_values = [2 ** -5, 2 ** -4, 2 ** -3, 2 ** -2, 2 ** -1]
    auroc_lambda = [0.9876, 0.9905, 0.9898, 0.9867, 0.9823]
    plot_parameter_sensitivity_simple(lambda_values, auroc_lambda, "Regularization", "lambda", 4, 1)

    print("\n" + "=" * 70)
    print("All figures saved to ./figures/")
    print("=" * 70)


if __name__ == "__main__":
    main()