#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模型可视化与解释工具模块
包含模型性能可视化、SHAP值计算和可视化等函数
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, 
    roc_curve, precision_recall_curve, confusion_matrix, mean_squared_error, r2_score
)
import warnings
# from .feature_engineering import create_derived_features # Example, may not be needed here
# from .font_config import CHINESE_FONT_PROP, check_and_load_font # Old problematic import
from scripts.data_visualization import CHINESE_FONT_PROP, apply_font_to_axis # Corrected import for CHINESE_FONT_PROP and apply_font_to_axis
# We assume check_and_load_font is either not used or imported correctly elsewhere if needed from .font_config.

# 尝试导入SHAP，如果失败则SHAP_AVAILABLE为False
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("警告: SHAP库未安装或导入失败。SHAP相关功能将不可用。")

warnings.filterwarnings('ignore')

# 从本地项目中导入必要的字体和可视化辅助函数
# 注意：这些导入路径可能需要根据您的项目结构进行调整
# 假设 font_config.py 和 data_visualization.py 在 scripts/ 目录下
try:
    # CHINESE_FONT_PROP and apply_font_to_axis are now imported from .data_visualization at the top.
    # Only attempt to import check_and_load_font from .font_config here.
    # from scripts.font_config import check_and_load_font # <-- REMOVE THIS
    # FONT_CONFIG_LOADED = True # Indicates that check_and_load_font loaded successfully # <-- REMOVE THIS
    pass # Keep try-except structure if other imports might be added here later, otherwise remove fully. For now, just pass.
except ImportError as e:
    # print(f"警告: 无法从 .font_config 导入 check_and_load_font: {e}.字体配置可能不完整。") # <-- REMOVE THIS
    # FONT_CONFIG_LOADED = False # <-- REMOVE THIS
    # check_and_load_font = None # Ensure it's defined for later checks, even if it's None # <-- REMOVE THIS
    print(f"警告: model_visualization_utils.py 中字体相关导入可能存在问题: {e}") # Generic warning if try-except is kept

# 全局字体属性变量 (如果需要的话)
# CHINESE_FONT_PROP = CHINESE_FONT_PROP # Already imported and available

# Ensure apply_font_to_axis has a fallback if the top import itself somehow failed before this point (though unlikely)
if 'apply_font_to_axis' not in globals() or not callable(globals()['apply_font_to_axis']):
    def apply_font_to_axis(ax): # Dummy function
        print("警告: apply_font_to_axis 未正确加载，使用占位符函数。")
        pass

def visualize_model_performance(y_true, y_pred, y_prob=None, task_type='classification', output_dir='output/image', model_name='model'):
    """
    可视化模型性能，包括ROC曲线、PR曲线和混淆矩阵
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        y_prob: 预测概率 (仅用于分类)
        task_type: 'classification' 或 'regression'
        output_dir: 图片保存目录
        model_name: 模型名称，用于文件名
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n可视化模型 '{model_name}' 的性能...")
    output_files = []

    plt.style.use('seaborn-v0_8-whitegrid') # 使用较新的seaborn样式

    if task_type == 'classification':
        is_binary = len(np.unique(y_true)) <= 2

        # 1. ROC曲线 (仅二分类)
        if is_binary and y_prob is not None and len(y_prob.shape) > 1 and y_prob.shape[1] == 2:
            try:
                fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
                roc_auc = roc_auc_score(y_true, y_prob[:, 1])
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('假阳性率')
                plt.ylabel('真阳性率')
                plt.title(f'{model_name} ROC曲线')
                plt.legend(loc="lower right")
                if CHINESE_FONT_PROP:
                    apply_font_to_axis(plt.gca())
                roc_path = os.path.join(output_dir, f'{model_name}_roc_curve.png')
                plt.savefig(roc_path)
                plt.close()
                output_files.append(roc_path)
                print(f"ROC曲线已保存: {roc_path}")
            except Exception as e:
                print(f"错误: 无法生成ROC曲线 for {model_name}: {e}")

        # 2. PR曲线 (仅二分类)
        if is_binary and y_prob is not None and len(y_prob.shape) > 1 and y_prob.shape[1] == 2:
            try:
                precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
                plt.figure(figsize=(8, 6))
                plt.plot(recall, precision, color='blue', lw=2, label='PR curve')
                plt.xlabel('召回率')
                plt.ylabel('精确率')
                plt.title(f'{model_name} PR曲线')
                plt.legend(loc="lower left")
                if CHINESE_FONT_PROP:
                    apply_font_to_axis(plt.gca())
                pr_path = os.path.join(output_dir, f'{model_name}_pr_curve.png')
                plt.savefig(pr_path)
                plt.close()
                output_files.append(pr_path)
                print(f"PR曲线已保存: {pr_path}")
            except Exception as e:
                print(f"错误: 无法生成PR曲线 for {model_name}: {e}")

        # 3. 混淆矩阵
        try:
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'{model_name} 混淆矩阵')
            plt.ylabel('真实标签')
            plt.xlabel('预测标签')
            if CHINESE_FONT_PROP:
                apply_font_to_axis(plt.gca())
            cm_path = os.path.join(output_dir, f'{model_name}_confusion_matrix.png')
            plt.savefig(cm_path)
            plt.close()
            output_files.append(cm_path)
            print(f"混淆矩阵已保存: {cm_path}")
        except Exception as e:
            print(f"错误: 无法生成混淆矩阵 for {model_name}: {e}")

    elif task_type == 'regression':
        # 回归任务的可视化 (例如: 真实值 vs 预测值)
        try:
            plt.figure(figsize=(8, 6))
            plt.scatter(y_true, y_pred, alpha=0.5)
            plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
            plt.xlabel('真实值')
            plt.ylabel('预测值')
            plt.title(f'{model_name} 真实值 vs 预测值')
            if CHINESE_FONT_PROP:
                apply_font_to_axis(plt.gca())
            reg_scatter_path = os.path.join(output_dir, f'{model_name}_regression_scatter.png')
            plt.savefig(reg_scatter_path)
            plt.close()
            output_files.append(reg_scatter_path)
            print(f"回归散点图已保存: {reg_scatter_path}")
        except Exception as e:
            print(f"错误: 无法生成回归散点图 for {model_name}: {e}")
    else:
        print(f"警告: 未知的任务类型 '{task_type}'，跳过性能可视化")

    return output_files

def explain_model_predictions(model, X, feature_names=None):
    """
    使用SHAP解释模型预测
    
    参数:
        model: 训练好的模型 (必须是XGBoost或支持SHAP的模型)
        X: 特征数据 (Pandas DataFrame或NumPy数组)
        feature_names: 特征名称列表 (如果X是NumPy数组则需要)
    
    返回:
        SHAP解释器和SHAP值 (如果SHAP可用且模型兼容)
        否则返回 None, None
    """
    global SHAP_AVAILABLE
    if not SHAP_AVAILABLE:
        print("SHAP库不可用，跳过模型解释")
        return None, None

    print("\n准备SHAP解释器和计算SHAP值...")
    explainer = None
    shap_values = None

    # 确保X是DataFrame，如果不是则尝试转换
    if not isinstance(X, pd.DataFrame):
        if isinstance(X, np.ndarray) and feature_names and len(feature_names) == X.shape[1]:
            X_df = pd.DataFrame(X, columns=feature_names)
            print("输入X已转换为Pandas DataFrame用于SHAP。")
        else:
            print("错误: SHAP解释需要Pandas DataFrame或带有feature_names的NumPy数组。")
            return None, None
    else:
        X_df = X

    try:
        # 检查模型类型并选择合适的SHAP解释器
        # 从Pipeline中提取实际的模型步骤
        actual_model = model.named_steps['model'] if isinstance(model, Pipeline) else model

        if isinstance(actual_model, (xgb.XGBClassifier, xgb.XGBRegressor)):
            print("使用TreeExplainer for XGBoost模型")
            explainer = shap.TreeExplainer(actual_model)
        elif hasattr(actual_model, 'predict_proba'): # KernelExplainer for black-box models with predict_proba
            print("警告: XGBoost以外的模型，尝试使用KernelExplainer (可能较慢)")
            # KernelExplainer需要一个背景数据集，通常是训练集的一个小子集
            # 为了简化，这里我们使用X的前N个样本作为背景数据
            # 注意: KernelExplainer对大型数据集非常慢
            background_data = shap.sample(X_df, min(100, X_df.shape[0])) 
            explainer = shap.KernelExplainer(actual_model.predict_proba, background_data)
        elif hasattr(actual_model, 'predict'): # KernelExplainer for models with predict (regression or non-proba classification)
            print("警告: XGBoost以外的模型，尝试使用KernelExplainer (可能较慢) for predict output")
            background_data = shap.sample(X_df, min(100, X_df.shape[0]))
            explainer = shap.KernelExplainer(actual_model.predict, background_data)
        else:
            print(f"错误: 不支持的模型类型 {type(actual_model)} 用于SHAP解释。")
            return None, None
        
        # 计算SHAP值
        # TreeExplainer可以直接计算，KernelExplainer在shap_values调用时计算
        # 注意KernelExplainer的shap_values可能返回一个列表（多分类）或一个数组（二分类/回归）
        print("正在计算SHAP值...")
        if isinstance(explainer, shap.explainers.Tree):
            shap_values_obj = explainer(X_df) # New SHAP API returns Explanation object
            # shap_values = explainer.shap_values(X_df) # Old API
            # For TreeExplainer, explainer(X) gives an Explanation object
            # We need to access .values for the raw SHAP values
            # And for multi-class, .values will be a list of arrays
            shap_values = shap_values_obj.values 
            if isinstance(shap_values, list) and len(shap_values) > 0 and not isinstance(shap_values[0], np.ndarray):
                 # If it's a list of Explanations (e.g. for multi-output), extract values from each
                try:
                    shap_values = [s.values for s in shap_values_obj]
                except:
                    pass # keep as is if that fails

            # For TreeExplainer, expected_value might be an array for multi-class
            # We may need to handle this if we use explainer.expected_value directly later

        elif isinstance(explainer, shap.explainers.Kernel):
            # KernelExplainer.shap_values(X) might return a list for multi-class classification
            # or a single array for binary classification/regression.
            shap_values = explainer.shap_values(X_df, nsamples='auto') # nsamples='auto' is default for Kernel
        else:
            shap_values = explainer.shap_values(X_df) # General case, might need adjustment
        
        print("SHAP值计算完成。")
        if isinstance(shap_values, list):
            print(f"SHAP值为列表 (可能为多分类)，包含 {len(shap_values)} 个数组，形状: {[s.shape for s in shap_values]}")
        elif hasattr(shap_values, 'shape'):
            print(f"SHAP值数组形状: {shap_values.shape}")
        else:
            print("SHAP值类型未知或计算失败。")

    except Exception as e:
        print(f"SHAP解释失败: {e}")
        return None, None

    return explainer, shap_values

def visualize_shap_values(explainer, shap_values, X_sample, output_dir="output/image", model_name="model"):
    """
    可视化SHAP值，包括摘要图、条形图、瀑布图和依赖图
    
    参数:
        explainer: SHAP解释器 (TreeExplainer或KernelExplainer的实例)
        shap_values: SHAP值 (通常是NumPy数组或列表)
        X_sample: 用于可视化的特征数据样本 (Pandas DataFrame)
        output_dir: 图片保存目录
        model_name: 模型名称，用于文件名
    """
    print(f"DEBUG: Entering visualize_shap_values for model: {model_name}")
    print(f"  Initial X_sample shape: {X_sample.shape if hasattr(X_sample, 'shape') else 'X_sample no shape attribute'}")
    if isinstance(X_sample, pd.DataFrame):
        print(f"  Initial X_sample is DataFrame, empty: {X_sample.empty}")

    global SHAP_AVAILABLE
    print(f"  DEBUG_SHAP_CHECK: SHAP_AVAILABLE: {SHAP_AVAILABLE}, explainer is None: {explainer is None}, shap_values is None: {shap_values is None}")
    if not SHAP_AVAILABLE or explainer is None or shap_values is None:
        print("SHAP解释器或SHAP值不可用，跳过SHAP值可视化")
        return []

    # Store original rcParams
    original_font_sans_serif = plt.rcParams['font.sans-serif']
    original_axes_unicode_minus = plt.rcParams['axes.unicode_minus']

    try:
        # Attempt to set a Chinese font globally for SHAP plots
        chinese_font_name = CHINESE_FONT_PROP.get_name() if CHINESE_FONT_PROP else 'SimHei' # Fallback to SimHei string
        plt.rcParams['font.sans-serif'] = [chinese_font_name] + original_font_sans_serif
        plt.rcParams['axes.unicode_minus'] = False
        print(f"DEBUG: Set plt.rcParams['font.sans-serif'] to: {plt.rcParams['font.sans-serif']}")

        os.makedirs(output_dir, exist_ok=True)
        print(f"\n生成SHAP值可视化 for {model_name}...")
        output_files = []

        # 确保X_sample是DataFrame
        X_sample_df = None # Initialize to see if it gets assigned
        if not isinstance(X_sample, pd.DataFrame):
            print("警告: SHAP可视化需要Pandas DataFrame作为X_sample。")
            if isinstance(X_sample, np.ndarray) and hasattr(explainer, 'feature_names') and explainer.feature_names is not None and len(explainer.feature_names) == X_sample.shape[1]:
                X_sample_df = pd.DataFrame(X_sample, columns=explainer.feature_names)
                print("  DEBUG: X_sample (ndarray) converted to DataFrame using explainer.feature_names.")
            elif isinstance(X_sample, np.ndarray) and shap_values is not None:
                # Determine num_features, carefully handling list or array for shap_values
                num_features_from_shap = -1
                if isinstance(shap_values, list) and len(shap_values) > 0 and hasattr(shap_values[0], 'shape'):
                    num_features_from_shap = shap_values[0].shape[-1]
                elif hasattr(shap_values, 'shape'):
                    num_features_from_shap = shap_values.shape[-1]
                
                if num_features_from_shap != -1 and X_sample.shape[1] == num_features_from_shap:
                    X_sample_df = pd.DataFrame(X_sample, columns=[f"feature_{i}" for i in range(X_sample.shape[1])])
                    print(f"  DEBUG: X_sample (ndarray) converted to DataFrame using shap_values shape for columns. num_features_from_shap: {num_features_from_shap}")
                else:
                    print(f"  DEBUG: X_sample (ndarray) could not be converted. X_sample.shape[1]: {X_sample.shape[1]}, num_features_from_shap: {num_features_from_shap}")
                    print("无法将X_sample转换为DataFrame，跳过SHAP可视化。 (来自ndarray转换路径)")
                    return []
            else:
                print("无法将X_sample转换为DataFrame，跳过SHAP可视化。 (来自非DataFrame, 非ndarray路径或 shap_values is None)")
                return []
        else:
            X_sample_df = X_sample
            print("  DEBUG: X_sample is already a DataFrame.")

        print(f"  DEBUG: X_sample_df defined. Is None: {X_sample_df is None}")
        if X_sample_df is not None:
            print(f"  X_sample_df shape after processing: {X_sample_df.shape}")
            if X_sample_df.empty:
                print(f"  WARNING: X_sample_df is EMPTY for model {model_name}!")
                # Decide if we should return here if X_sample_df is empty,
                # as many subsequent plots rely on it having data.
                # Some SHAP plots might handle empty DFs, but it's likely to cause issues.
                # For now, let it proceed to see if specific plots fail or are skipped by their own logic.
        else:
            print(f"  CRITICAL_DEBUG: X_sample_df is None after processing block, this should not happen if not returned early.")
            return [] # Safety return if X_sample_df is None

        # 确定任务类型 (二分类, 多分类, 回归) 以便正确处理SHAP值
        # TreeExplainer for multi-class returns a list of shap_values arrays (one per class)
        # KernelExplainer for multi-class (predict_proba) also returns a list of arrays.
        # For binary classification, both often return a single array for the positive class, or sometimes a list of two (one for each class).
        # For regression, it's a single array.

        is_multiclass_shap = isinstance(shap_values, list) and len(shap_values) > 1
        # If shap_values is a list of length 1, treat as single output
        if isinstance(shap_values, list) and len(shap_values) == 1:
            current_shap_values = shap_values[0]
            is_multiclass_shap = False # Effectively single output
        elif is_multiclass_shap:
            # For multiclass, summary plot can take the list directly.
            # For other plots, we often need to select SHAP values for a specific class.
            # Default to plotting for class 1 if available, otherwise class 0.
            class_to_plot_idx = 1 if len(shap_values) > 1 else 0
            print(f"多分类SHAP值，将主要针对类别 {class_to_plot_idx} 进行单样本和依赖图可视化。")
            current_shap_values = shap_values[class_to_plot_idx] # For single class plots
        else: # Single array (binary classification or regression)
            current_shap_values = shap_values
        
        # Handle cases where current_shap_values might still be an Explanation object (new SHAP API)
        if hasattr(current_shap_values, 'values'):
            current_shap_values = current_shap_values.values
        if isinstance(shap_values, list):
            # Ensure all elements in the list are ndarrays if it was a list of Explanations
            processed_list_shap_values = []
            for sv_item in shap_values:
                if hasattr(sv_item, 'values'):
                    processed_list_shap_values.append(sv_item.values)
                elif isinstance(sv_item, np.ndarray):
                    processed_list_shap_values.append(sv_item)
                # else skip if not convertible
            if processed_list_shap_values:
                shap_values_for_summary = processed_list_shap_values
            else:
                shap_values_for_summary = shap_values # Fallback
        else: # single array
            shap_values_for_summary = current_shap_values

        # 1. SHAP摘要图 (Summary Plot - Beeswarm)
        try:
            plt.figure(figsize=(12, 10))
            with plt.rc_context({'font.family': 'sans-serif', 'font.sans-serif': ['SimHei', 'DejaVu Sans']}):
                shap.summary_plot(shap_values_for_summary, X_sample_df, plot_type="dot", show=False)
            
            current_ax = plt.gca()
            if CHINESE_FONT_PROP and current_ax:
                current_ax.set_xlabel(current_ax.get_xlabel(), fontproperties=CHINESE_FONT_PROP)
                for label in current_ax.get_xticklabels(): label.set_fontproperties(CHINESE_FONT_PROP)
                for label in current_ax.get_yticklabels(): label.set_fontproperties(CHINESE_FONT_PROP)
                if current_ax.get_legend():
                    for text in current_ax.get_legend().get_texts(): text.set_fontproperties(CHINESE_FONT_PROP)
                cbar = getattr(current_ax.collections[0], 'colorbar', None)
                if cbar and hasattr(cbar.ax, 'set_ylabel') and hasattr(cbar.ax, 'get_yticklabels'):
                    cbar.ax.set_ylabel(cbar.ax.get_ylabel(), fontproperties=CHINESE_FONT_PROP)
                    for tick_label in cbar.ax.get_yticklabels(): tick_label.set_fontproperties(CHINESE_FONT_PROP)

            plt.tight_layout()
            summary_path = os.path.join(output_dir, f'{model_name}_shap_summary_beeswarm.png')
            plt.savefig(summary_path)
            plt.close()
            output_files.append(summary_path)
            print(f"SHAP摘要图 (Beeswarm) 已保存: {summary_path}")
        except Exception as e:
            print(f"错误: 生成SHAP摘要图失败: {e}")
            import traceback
            traceback.print_exc()

        # 2. SHAP条形图 (Global Feature Importance)
        try:
            plt.figure(figsize=(10, 8))
            with plt.rc_context({'font.family': 'sans-serif', 'font.sans-serif': ['SimHei', 'DejaVu Sans']}):
                # For multiclass, shap.summary_plot with plot_type="bar" needs a list of arrays.
                # If current_shap_values was selected for a single class, use shap_values (the full list) here.
                shap.summary_plot(shap_values_for_summary, X_sample_df, plot_type="bar", show=False)
            
            current_ax = plt.gca()
            if CHINESE_FONT_PROP and current_ax:
                current_ax.set_xlabel(current_ax.get_xlabel(), fontproperties=CHINESE_FONT_PROP)
                for label in current_ax.get_xticklabels(): label.set_fontproperties(CHINESE_FONT_PROP)
                for label in current_ax.get_yticklabels(): label.set_fontproperties(CHINESE_FONT_PROP)

            plt.tight_layout()
            bar_path = os.path.join(output_dir, f'{model_name}_shap_summary_bar.png')
            plt.savefig(bar_path)
            plt.close()
            output_files.append(bar_path)
            print(f"SHAP条形图已保存: {bar_path}")
        except Exception as e:
            print(f"错误: 生成SHAP条形图失败: {e}")

        # 准备单样本SHAP值和期望值 (用于瀑布图和力图)
        # explainer.expected_value is the base value
        # For TreeExplainer, expected_value might be an array for multi-class (one per class)
        # For KernelExplainer, it's often a single value or an array (one per class if predict_proba was multi-output)
        base_values = None
        if hasattr(explainer, 'expected_value'):
            base_values = explainer.expected_value
        
        # If base_values is a list/array (multi-class), select the one for the class we are plotting
        if isinstance(base_values, (list, np.ndarray)) and len(base_values) > 1 and is_multiclass_shap:
            current_base_value_for_sample = base_values[class_to_plot_idx]
        elif isinstance(base_values, (list, np.ndarray)) and len(base_values) == 1: # List with one element
             current_base_value_for_sample = base_values[0]
        else: # Single value (binary classification or regression)
            current_base_value_for_sample = base_values

        # Select SHAP values for a single sample
        sample_idx = 0 # Plot for the first sample in X_sample_df
        # current_shap_values should already be for the selected class if multiclass
        if current_shap_values is not None and len(current_shap_values.shape) > 1 and current_shap_values.shape[0] > sample_idx:
            shap_values_sample_for_plot = current_shap_values[sample_idx]
        else:
            shap_values_sample_for_plot = None 
            print(f"警告: 无法为样本 {sample_idx} 提取SHAP值用于瀑布/力图。SHAP值维度: {current_shap_values.shape if hasattr(current_shap_values, 'shape') else 'N/A'}")

        # 3. SHAP瀑布图 (Waterfall Plot for one sample)
        if shap_values_sample_for_plot is not None and current_base_value_for_sample is not None:
            try:
                # Create an Explanation object for the single sample
                explanation_for_waterfall = shap.Explanation(
                    values=shap_values_sample_for_plot,
                    base_values=current_base_value_for_sample,
                    data=X_sample_df.iloc[sample_idx].values,
                    feature_names=X_sample_df.columns.tolist()
                )

                # Store original rcParams for font.sans-serif
                original_rc_font_sans_serif = plt.rcParams['font.sans-serif']
                # Ensure CHINESE_FONT_PROP.get_name() is valid, fallback to 'SimHei' string if needed
                chinese_font_name_to_use = 'SimHei' # Default
                if CHINESE_FONT_PROP and hasattr(CHINESE_FONT_PROP, 'get_name'):
                    font_name_from_prop = CHINESE_FONT_PROP.get_name()
                    if font_name_from_prop: # Ensure it's not empty or None
                        chinese_font_name_to_use = font_name_from_prop
                
                plt.rcParams['font.sans-serif'] = [chinese_font_name_to_use] + original_rc_font_sans_serif # Prioritize, but keep fallbacks

                plt.figure(figsize=(14, 8))
                # The plt.rc_context might be redundant if we set rcParams directly, but can be kept for safety.
                # It's often better to manage rcParams directly for such specific overrides.
                # We'll rely on the direct rcParams override for this plot.
                
                shap.waterfall_plot(
                    explanation_for_waterfall,
                    max_display=20,
                    show=False
                )
                
                current_ax = plt.gca()
                print("---- Debugging SHAP Waterfall Plot Texts (Post rcParam Override) ----")
                if CHINESE_FONT_PROP and current_ax:
                    print(f"Attempting to apply font using CHINESE_FONT_PROP: {chinese_font_name_to_use}")
                    # Apply to general axis elements (title, labels)
                    current_ax.set_title(current_ax.get_title(), fontproperties=CHINESE_FONT_PROP)
                    current_ax.set_xlabel(current_ax.get_xlabel(), fontproperties=CHINESE_FONT_PROP)
                    current_ax.set_ylabel(current_ax.get_ylabel(), fontproperties=CHINESE_FONT_PROP) # Y-axis is feature names

                    for label in current_ax.get_xticklabels():
                        label.set_fontproperties(CHINESE_FONT_PROP)
                    for label in current_ax.get_yticklabels(): # These are the feature names on the y-axis
                        label.set_fontproperties(CHINESE_FONT_PROP)
                        print(f"  YTickLabel '{label.get_text()}': Font set to {label.get_fontproperties().get_name()}")

                    # Iterate over all text artists on the axes
                    for i, text_obj in enumerate(current_ax.texts):
                        original_font_name = text_obj.get_fontproperties().get_name()
                        text_obj.set_fontproperties(CHINESE_FONT_PROP)
                        print(f"  Text {i}: '{text_obj.get_text()[:30]}' - Original Font: {original_font_name}, New Font: {text_obj.get_fontproperties().get_name()}")
                print("---- End Debugging SHAP Waterfall Plot Texts ----")

                plt.tight_layout()
                waterfall_path = os.path.join(output_dir, f'{model_name}_shap_waterfall_sample_{sample_idx}.png')
                plt.savefig(waterfall_path)
                plt.close()
                output_files.append(waterfall_path)
                print(f"SHAP瀑布图已保存: {waterfall_path}")

            except Exception as e:
                print(f"错误: 生成SHAP瀑布图失败: {e}")
                import traceback
                traceback.print_exc()
            finally:
                # Restore original rcParams
                plt.rcParams['font.sans-serif'] = original_rc_font_sans_serif

        # 4. SHAP依赖图 (Dependence Plot for top features)
        # current_shap_values should be for the selected class in multiclass or the only class for binary/regression
        if current_shap_values is not None and len(current_shap_values.shape) == 2:
            # Calculate mean absolute SHAP values to find top features
            if isinstance(current_shap_values, np.ndarray):
                 mean_abs_shap = np.abs(current_shap_values).mean(axis=0)
                 top_features_indices = np.argsort(mean_abs_shap)[-min(5, len(mean_abs_shap)):][::-1] # Top 5 features
                 top_features_names = X_sample_df.columns[top_features_indices]
            else:
                top_features_names = X_sample_df.columns[:min(5, X_sample_df.shape[1])] # fallback
                print("警告: 无法从SHAP值确定顶级特征进行依赖图绘制，将使用前5个特征。")

            for feature_name in top_features_names:
                try:
                    plt.figure(figsize=(8, 6))
                    with plt.rc_context({'font.family': 'sans-serif', 'font.sans-serif': ['SimHei', 'DejaVu Sans']}):
                        shap.dependence_plot(
                            feature_name, 
                            current_shap_values, # SHAP values for the specific class
                            X_sample_df, 
                            interaction_index="auto", # Automatically pick interaction feature
                            show=False
                        )
                    
                    current_ax = plt.gca()
                    if CHINESE_FONT_PROP and current_ax:
                        current_ax.set_xlabel(current_ax.get_xlabel(), fontproperties=CHINESE_FONT_PROP)
                        current_ax.set_ylabel(current_ax.get_ylabel(), fontproperties=CHINESE_FONT_PROP)
                        for label in current_ax.get_xticklabels(): label.set_fontproperties(CHINESE_FONT_PROP)
                        for label in current_ax.get_yticklabels(): label.set_fontproperties(CHINESE_FONT_PROP)
                        if current_ax.get_legend():
                            for text in current_ax.get_legend().get_texts(): text.set_fontproperties(CHINESE_FONT_PROP)
                        cbar_collection = next((coll for coll in current_ax.collections if hasattr(coll, 'colorbar') and coll.colorbar is not None), None)
                        if cbar_collection:
                            cbar = cbar_collection.colorbar
                            if hasattr(cbar.ax, 'set_ylabel') and hasattr(cbar.ax, 'get_yticklabels'):
                                cbar.ax.set_ylabel(cbar.ax.get_ylabel(), fontproperties=CHINESE_FONT_PROP)
                                for tick_label in cbar.ax.get_yticklabels(): tick_label.set_fontproperties(CHINESE_FONT_PROP)

                    plt.tight_layout()
                    dep_path = os.path.join(output_dir, f'{model_name}_shap_dependence_{feature_name}.png')
                    plt.savefig(dep_path)
                    plt.close()
                    output_files.append(dep_path)
                    print(f"SHAP依赖图 ({feature_name}) 已保存: {dep_path}")
                except Exception as e:
                    print(f"错误: 生成SHAP依赖图 ({feature_name}) 失败: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            print("SHAP值格式不适合依赖图 (需要2D数组)。")

        # 5. SHAP力图 (Force Plot - Individual Sample as Image)
        # shap_values_sample_for_plot should be a 1D array for one sample
        # current_base_value_for_sample should be a scalar
        print(f"DEBUG: Before Individual Force Plot:")
        print(f"  shap_values_sample_for_plot is None: {shap_values_sample_for_plot is None}")
        if shap_values_sample_for_plot is not None:
            print(f"  shap_values_sample_for_plot.shape: {shap_values_sample_for_plot.shape}")
        print(f"  current_base_value_for_sample is None: {current_base_value_for_sample is None}")
        if current_base_value_for_sample is not None:
            print(f"  current_base_value_for_sample: {current_base_value_for_sample}")
            print(f"  type(current_base_value_for_sample): {type(current_base_value_for_sample)}")
        print(f"  X_sample_df.shape: {X_sample_df.shape}")
        if shap_values_sample_for_plot is not None and current_base_value_for_sample is not None:
            try:
                # Force plot for a single prediction
                # Need to ensure expected_value is scalar here
                expected_value_scalar = current_base_value_for_sample
                if isinstance(expected_value_scalar, (list, np.ndarray)) and len(expected_value_scalar) == 1:
                    expected_value_scalar = expected_value_scalar[0]
                elif isinstance(expected_value_scalar, (list, np.ndarray)):
                     print(f"警告: Force plot 的 expected_value 不是标量 ({expected_value_scalar})，将使用第一个元素。")
                     expected_value_scalar = expected_value_scalar[0]


                # Check if shap_values_sample_for_plot is 1D
                if len(shap_values_sample_for_plot.shape) == 1:
                    plt.figure() # Create a new figure context for force_plot with matplotlib=True
                    with plt.rc_context({'font.family': 'sans-serif', 'font.sans-serif': ['SimHei', 'DejaVu Sans']}):
                        shap.force_plot(
                            expected_value_scalar,
                            shap_values_sample_for_plot,
                            X_sample_df.iloc[sample_idx],
                            matplotlib=True, # Render with Matplotlib
                            show=False
                        )
                    # Apply font to current Matplotlib axes if needed, though force_plot with matplotlib=True handles its own text
                    # current_ax = plt.gca()
                    # if CHINESE_FONT_PROP and current_ax:
                    #     # force_plot text elements might be harder to iterate and apply fonts to directly
                    #     # The rc_context should ideally handle it.
                    #     pass

                    plt.tight_layout() # May or may not be effective depending on how shap.force_plot uses matplotlib
                    force_img_path = os.path.join(output_dir, f'{model_name}_shap_force_sample_{sample_idx}.png')
                    plt.savefig(force_img_path, bbox_inches='tight') # bbox_inches='tight' helps with layout
                    plt.close()
                    output_files.append(force_img_path)
                    print(f"SHAP力图 (样本 {sample_idx}) 已保存为图片: {force_img_path}")
                else:
                    print(f"警告: 单样本SHAP力图的SHAP值不是1D数组 (形状: {shap_values_sample_for_plot.shape})，跳过。")

            except Exception as e:
                print(f"错误: 生成SHAP力图 (样本 {sample_idx}) 失败: {e}")
                import traceback
                traceback.print_exc()

        # 6. SHAP力图 (Force Plot - Multiple Samples as HTML)
        # We use current_shap_values here, which should be the SHAP values for the selected class (or all if not multi-class)
        # And X_sample_df for the feature values
        # Base value should be scalar here if possible
        print(f"DEBUG: Before Multiple Samples HTML Force Plot:")
        print(f"  current_shap_values is None: {current_shap_values is None}")
        if current_shap_values is not None:
            print(f"  current_shap_values.shape: {current_shap_values.shape if hasattr(current_shap_values, 'shape') else 'Not an array or list of arrays with uniform shape'}")
            if isinstance(current_shap_values, list):
                print(f"  current_shap_values is list, shapes: {[item.shape for item in current_shap_values if hasattr(item, 'shape')]}")
        expected_value_for_html_force = current_base_value_for_sample # Use the one derived for single sample for consistency
        if isinstance(expected_value_for_html_force, (list, np.ndarray)) and len(expected_value_for_html_force) == 1:
            expected_value_for_html_force = expected_value_for_html_force[0]
        elif isinstance(expected_value_for_html_force, (list, np.ndarray)):
            print(f"警告: HTML Force plot 的 expected_value 不是标量 ({expected_value_for_html_force})，将使用第一个元素。")
            expected_value_for_html_force = expected_value_for_html_force[0]
        
        print(f"  expected_value_for_html_force is None: {expected_value_for_html_force is None}")
        if expected_value_for_html_force is not None:
            print(f"  expected_value_for_html_force: {expected_value_for_html_force}")
        print(f"  X_sample_df.shape: {X_sample_df.shape}")

        if current_shap_values is not None and expected_value_for_html_force is not None:
            try:
                # This version of force_plot generates an HTML object
                force_html = shap.force_plot(
                    expected_value_for_html_force,
                    current_shap_values, # SHAP values for all samples in X_sample_df for the chosen class
                    X_sample_df,
                    show=False
                )
                if force_html:
                    force_html_path = os.path.join(output_dir, f'{model_name}_shap_force_multiple.html')
                    with open(force_html_path, 'w', encoding='utf-8') as f:
                        f.write(force_html.html())
                    output_files.append(force_html_path)
                    print(f"SHAP力图 (多个样本) 已保存为HTML: {force_html_path}")
            except Exception as e:
                print(f"错误: 生成SHAP力图 (多个样本HTML) 失败: {e}")
                import traceback
                traceback.print_exc()
            
        # 7. SHAP决策图 (Decision Plot)
        # Uses base_value and SHAP values for the selected class.
        # Can plot for one or multiple samples. Let's plot for the first few samples.
        num_samples_for_decision_plot = min(5, X_sample_df.shape[0])
        print(f"DEBUG: Before Decision Plot:")
        print(f"  current_shap_values is None: {current_shap_values is None}")
        if current_shap_values is not None:
            print(f"  current_shap_values.shape: {current_shap_values.shape if hasattr(current_shap_values, 'shape') else 'Not an array or list of arrays with uniform shape'}")
        print(f"  current_base_value_for_sample is None: {current_base_value_for_sample is None}")
        if current_base_value_for_sample is not None:
            print(f"  current_base_value_for_sample for decision: {current_base_value_for_sample}")
        print(f"  num_samples_for_decision_plot: {num_samples_for_decision_plot}")
        print(f"  X_sample_df.shape: {X_sample_df.shape}")
        if current_shap_values is not None and current_base_value_for_sample is not None and num_samples_for_decision_plot > 0:
            try:
                expected_value_for_decision = current_base_value_for_sample
                if isinstance(expected_value_for_decision, (list, np.ndarray)) and len(expected_value_for_decision) == 1:
                    expected_value_for_decision = expected_value_for_decision[0]
                elif isinstance(expected_value_for_decision, (list, np.ndarray)):
                    print(f"警告: Decision plot 的 expected_value 不是标量 ({expected_value_for_decision})，将使用第一个元素。")
                    expected_value_for_decision = expected_value_for_decision[0]

                # Ensure current_shap_values is 2D for decision plot when plotting multiple samples
                shap_values_for_decision = current_shap_values[:num_samples_for_decision_plot]
                features_for_decision = X_sample_df.iloc[:num_samples_for_decision_plot]

                print(f"DEBUG: Inside Decision Plot try, before shape check:")
                print(f"  shap_values_for_decision.shape: {shap_values_for_decision.shape if hasattr(shap_values_for_decision, 'shape') else 'N/A'}")
                print(f"  features_for_decision.shape: {features_for_decision.shape if hasattr(features_for_decision, 'shape') else 'N/A'}")
                
                if len(shap_values_for_decision.shape) == 2:
                    plt.figure(figsize=(10, 6 + num_samples_for_decision_plot * 0.5)) # Adjust height based on samples
                    with plt.rc_context({'font.family': 'sans-serif', 'font.sans-serif': ['SimHei', 'DejaVu Sans']}):
                        shap.decision_plot(
                            expected_value_for_decision,
                            shap_values_for_decision,
                            features_for_decision,
                            feature_names=X_sample_df.columns.tolist(),
                            show=False,
                            # link='logit' # if the output is log-odds and you want to see probabilities
                            # highlight=0 # highlight a specific sample
                        )
                    
                    current_ax = plt.gca()
                    if CHINESE_FONT_PROP and current_ax:
                        current_ax.set_xlabel(current_ax.get_xlabel(), fontproperties=CHINESE_FONT_PROP)
                        current_ax.set_ylabel(current_ax.get_ylabel(), fontproperties=CHINESE_FONT_PROP) # Y-axis is usually feature names
                        for label in current_ax.get_xticklabels(): label.set_fontproperties(CHINESE_FONT_PROP)
                        # Y-tick labels (feature names) in decision_plot are often handled by SHAP internally based on feature_names
                        # but let's try applying to them too.
                        for label in current_ax.get_yticklabels(): label.set_fontproperties(CHINESE_FONT_PROP)
                        if current_ax.get_legend():
                             for text in current_ax.get_legend().get_texts(): text.set_fontproperties(CHINESE_FONT_PROP)

                    plt.tight_layout()
                    decision_path = os.path.join(output_dir, f'{model_name}_shap_decision_plot.png')
                    plt.savefig(decision_path)
                    plt.close()
                    output_files.append(decision_path)
                    print(f"SHAP决策图已保存: {decision_path}")
                else:
                    print(f"警告: SHAP决策图的SHAP值不是2D数组 (形状: {shap_values_for_decision.shape})，跳过。")

            except Exception as e:
                print(f"错误: 生成SHAP决策图失败: {e}")
                import traceback
                traceback.print_exc()

        # 8. SHAP 热力图 (Heatmap Plot)
        # current_shap_values should be for the selected class in multiclass or the only class for binary/regression
        # X_sample_df for features
        # This plot can be intensive if X_sample_df is very large.
        # We might want to use a subset of samples for the heatmap if performance is an issue.
        # For now, let's use all of X_sample_df.
        print(f"DEBUG: Before Heatmap Plot:")
        print(f"  current_shap_values is None: {current_shap_values is None}")
        if current_shap_values is not None:
            print(f"  current_shap_values.shape: {current_shap_values.shape if hasattr(current_shap_values, 'shape') else 'Not an array'}")
        print(f"  X_sample_df.shape: {X_sample_df.shape}")

        if current_shap_values is not None and not X_sample_df.empty:
            # shap.plots.heatmap expects SHAP values as an Explanation object or a numpy array.
            # If current_shap_values is a list (multiclass from KernelExplainer), we need to pick one class.
            # If it's already an ndarray (binary/regression, or one class from TreeExplainer), it's fine.
            
            shap_values_for_heatmap = current_shap_values
            if isinstance(current_shap_values, list): # Should have been handled before, but double check
                print(f"警告: Heatmap plot recibió una lista de SHAP values, usando el índice {class_to_plot_idx}.")
                shap_values_for_heatmap = current_shap_values[class_to_plot_idx]

            # Ensure shap_values_for_heatmap is a 2D numpy array
            if hasattr(shap_values_for_heatmap, 'values') and isinstance(shap_values_for_heatmap.values, np.ndarray) and len(shap_values_for_heatmap.values.shape) == 2:
                shap_values_for_heatmap_np = shap_values_for_heatmap.values
            elif isinstance(shap_values_for_heatmap, np.ndarray) and len(shap_values_for_heatmap.shape) == 2:
                shap_values_for_heatmap_np = shap_values_for_heatmap
            else:
                print(f"警告: SHAP热力图的SHAP值格式不正确 (需要2D NumPy数组，得到类型 {type(shap_values_for_heatmap)}，形状 {(shap_values_for_heatmap.shape if hasattr(shap_values_for_heatmap, 'shape') else 'N/A')})，跳过。")
                shap_values_for_heatmap_np = None

            if shap_values_for_heatmap_np is not None:
                try:
                    plt.figure(figsize=(12, 10)) # Adjust as needed
                    # Construct an Explanation object if we only have numpy arrays
                    # shap.plots.heatmap ideally wants an Explanation object.
                    
                    # If explainer.expected_value is a list for multiclass, select the correct one.
                    base_value_for_heatmap = explainer.expected_value
                    if isinstance(base_value_for_heatmap, (list, np.ndarray)) and len(base_value_for_heatmap) > 1 and is_multiclass_shap:
                        base_value_for_heatmap = base_value_for_heatmap[class_to_plot_idx]
                    elif isinstance(base_value_for_heatmap, (list, np.ndarray)) and len(base_value_for_heatmap) == 1:
                        base_value_for_heatmap = base_value_for_heatmap[0]
                    # else it's a scalar or correctly selected

                    explanation_for_heatmap = shap.Explanation(
                        values=shap_values_for_heatmap_np,
                        base_values=base_value_for_heatmap, # This might need to be an array if shap_values_for_heatmap_np has multiple samples
                        data=X_sample_df.values,
                        feature_names=X_sample_df.columns.tolist()
                    )
                    
                    with plt.rc_context({'font.family': 'sans-serif', 'font.sans-serif': ['SimHei', 'DejaVu Sans']}):
                        shap.plots.heatmap(explanation_for_heatmap, max_display=20, show=False)

                    # Font handling for heatmap can be tricky as it has multiple text elements
                    current_ax = plt.gca()
                    if CHINESE_FONT_PROP and current_ax:
                        for label in current_ax.get_xticklabels(): label.set_fontproperties(CHINESE_FONT_PROP)
                        for label in current_ax.get_yticklabels(): label.set_fontproperties(CHINESE_FONT_PROP)
                        if current_ax.get_title(): current_ax.set_title(current_ax.get_title(), fontproperties=CHINESE_FONT_PROP)
                        # Heatmap colorbar labels also need font setting if present
                        # This is a simplification; a more robust way might be needed if text elements are deeply nested.

                    plt.tight_layout()
                    heatmap_path = os.path.join(output_dir, f'{model_name}_shap_heatmap.png')
                    plt.savefig(heatmap_path)
                    plt.close()
                    output_files.append(heatmap_path)
                    print(f"SHAP热力图已保存: {heatmap_path}")
                except AttributeError as ae:
                    if "'TreeEnsemble' object has no attribute 'values'" in str(ae) or \
                       "'KernelExplainer' object has no attribute 'values'" in str(ae) : # newer shap versions
                        print(f"提示: SHAP热力图可能需要 explainer.shap_values(X) 的原始输出，而不是 Explanation.values。当前使用的 shap_values_for_heatmap_np 的形状是 {shap_values_for_heatmap_np.shape}")
                    print(f"错误: 生成SHAP热力图失败 (可能由于SHAP版本API不匹配): {ae}")
                    import traceback
                    traceback.print_exc()

                except Exception as e:
                    print(f"错误: 生成SHAP热力图失败: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            print("SHAP值或样本数据为空，跳过SHAP热力图。")

        # 9. SHAP 交互值摘要图 (Interaction Value Summary/Bar Plot)
        # This is typically for TreeExplainer models
        print(f"DEBUG: Before Interaction Plot:")
        if isinstance(explainer, shap.explainers.Tree) and hasattr(explainer, 'shap_interaction_values'):
            try:
                print("  Calculating SHAP interaction values...")
                shap_interaction_values = explainer.shap_interaction_values(X_sample_df)
                print(f"  SHAP interaction values calculated. Type: {type(shap_interaction_values)}, Shape: {(shap_interaction_values.shape if hasattr(shap_interaction_values, 'shape') else 'N/A')}")

                if shap_interaction_values is not None:
                    # For interaction summary plot, we often want a global summary (bar plot)
                    plt.figure(figsize=(12, 10))
                    with plt.rc_context({'font.family': 'sans-serif', 'font.sans-serif': ['SimHei', 'DejaVu Sans']}):
                        # The summary_plot for interaction values might expect a specific format or summarization first.
                        # Often, we sum absolute interaction values for each feature pair to get an overall importance.
                        # shap.summary_plot can take interaction_values directly if they are structured correctly (e.g., for beeswarm of interactions)
                        # For a bar plot of main interaction effects:
                        shap.summary_plot(shap_interaction_values, X_sample_df, plot_type="bar", max_display=20, show=False)
                    
                    current_ax = plt.gca()
                    if CHINESE_FONT_PROP and current_ax:
                        current_ax.set_xlabel("平均|SHAP交互值| (特征影响)", fontproperties=CHINESE_FONT_PROP)
                        for label in current_ax.get_xticklabels(): label.set_fontproperties(CHINESE_FONT_PROP)
                        for label in current_ax.get_yticklabels(): label.set_fontproperties(CHINESE_FONT_PROP)

                    plt.tight_layout()
                    interaction_summary_path = os.path.join(output_dir, f'{model_name}_shap_interaction_summary_bar.png')
                    plt.savefig(interaction_summary_path)
                    plt.close()
                    output_files.append(interaction_summary_path)
                    print(f"SHAP交互值摘要条形图已保存: {interaction_summary_path}")
                else:
                    print("  SHAP interaction values are None after calculation.")

            except Exception as e:
                print(f"错误: 生成SHAP交互值摘要图失败: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("模型类型不支持SHAP交互值 (需要TreeExplainer) 或 shap_interaction_values 方法不存在，跳过交互图。")


        return output_files
    finally:
        # Restore original rcParams
        plt.rcParams['font.sans-serif'] = original_font_sans_serif
        plt.rcParams['axes.unicode_minus'] = original_axes_unicode_minus
        print(f"DEBUG: Restored plt.rcParams['font.sans-serif'] to: {plt.rcParams['font.sans-serif']}")


    return output_files 