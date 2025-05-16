#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
风险预测主模块
该模块协调特征准备、模型训练、评估和可视化。
"""

import os
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split # Keep for main data split if needed elsewhere
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score
)

# 从新模块导入函数
from feature_engineering import create_derived_features, prepare_features_and_target
from model_core import train_risk_prediction_model
from model_visualization_utils import (
    visualize_model_performance, 
    explain_model_predictions, 
    visualize_shap_values,
    SHAP_AVAILABLE # Import SHAP_AVAILABLE to check if SHAP can be used
)

# 尝试从本地项目中导入必要的字体和可视化辅助函数
# (如果 run_risk_prediction 中直接使用这些)
# from .font_config import CHINESE_FONT_PROP, check_and_load_font
# from .data_visualization import apply_font_to_axis

warnings.filterwarnings('ignore')

# 全局变量或配置
target_definitions = [
    {
        'key': '高血压风险',
        'name': '高血压风险',
        'col': 'bp_category_code',
        'description': '预测高血压风险 (bp_category_code >= 2, 即1级高血压或更高)',
        'type': '回归->分类',
        'threshold': 2.0
    },
    {
        'key': 'PWV超标风险',
        'name': 'PWV超标风险',
        'col': 'cfpwv_speed',
        'description': '预测PWV超标风险（cfPWV≥10m/s）',
        'type': '回归->分类',
        'threshold': 10.0
    },
    {
        'key': '高综合风险',
        'name': '综合心血管风险',
        'col': 'high_comprehensive_risk_flag',
        'description': '预测高综合心血管风险 (基于综合风险等级的高风险或极高风险)',
        'type': '分类',
        'threshold': None
    }
]

def run_risk_prediction(df, target_key='高血压风险', model_type_override=None, output_dir="output"):
    """
    运行风险预测流程，包括特征准备、模型训练、评估和可视化
    
    参数:
        df: 输入的DataFrame
        target_key: 要预测的目标风险的键 (来自 target_definitions)
        model_type_override: 强制指定模型类型 (例如 'xgboost_classifier')
        output_dir: 输出目录的根路径
    
    返回:
        包含模型、指标和SHAP结果的字典
    """
    print(f"\n==== 开始风险预测: {target_key} ====")
    
    target_info = next((t for t in target_definitions if t['key'] == target_key), None)
    if not target_info:
        print(f"错误: 未找到目标 '{target_key}' 的定义")
        return None

    target_col = target_info['col']
    threshold = target_info.get('threshold')
    model_results = {'target_key': target_key, 'target_name': target_info['name']}
    
    X, y, is_binary_target = prepare_features_and_target(df.copy(), target_col, threshold)
    if X is None or y is None:
        print(f"特征/目标准备失败 for {target_key}")
        return None

    # Check class distribution in y
    if y is not None and hasattr(y, 'value_counts'):
        class_counts = y.value_counts()
        if (class_counts < 2).any():
            print(f"警告: 目标 '{target_key}' 的目标变量中存在样本数小于2的类别: \\n{class_counts}")
            print(f"跳过对 '{target_key}' 的模型训练，因为无法进行有效的训练/测试分割。")
            return None
            
    model_results['feature_names'] = list(X.columns) if hasattr(X, 'columns') else None

    if X.shape[0] > 50:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42,
            stratify=y if (hasattr(y, 'nunique') and y.nunique() < 10 and len(y) > 10) else None
        )
        print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    else:
        X_train, y_train = X, y
        X_test, y_test = None, None
        print(f"样本量过小 ({X.shape[0]}), 使用全部数据进行训练和评估。")

    if model_type_override:
        model_type = model_type_override
        print(f"使用覆盖的模型类型: {model_type}")
    else:
        if target_info['type'] == '回归' and threshold is None:
            model_type = 'xgboost_regressor'
        elif target_info['type'] == '分类' or threshold is not None or target_info['type'] == '回归->分类' or target_info['type'] == '综合分级': 
            model_type = 'xgboost_classifier'
        else:
            print(f"警告: 未明确的模型类型 for target '{target_key}' with type '{target_info['type']}' and threshold '{threshold}'. 默认为分类器.")
            model_type = 'xgboost_classifier'
    print(f"选择的模型类型: {model_type}")
    model_results['model_type'] = model_type

    model = train_risk_prediction_model(X_train, y_train, model_type=model_type)
    if model is None:
        print(f"模型训练失败 for {target_key}")
        return None
    model_results['model'] = model
    
    metrics = {}
    y_pred_test = None
    y_prob_test = None

    data_to_evaluate_on = X_test if X_test is not None else X_train
    true_labels_for_eval = y_test if y_test is not None else y_train

    if data_to_evaluate_on is not None and true_labels_for_eval is not None and not data_to_evaluate_on.empty:
        y_pred_eval = model.predict(data_to_evaluate_on)
        is_classification_task = model_type.endswith('_classifier')
        
        if is_classification_task:
            metrics['accuracy'] = accuracy_score(true_labels_for_eval, y_pred_eval)
            num_classes = len(np.unique(true_labels_for_eval)) if hasattr(true_labels_for_eval, 'unique') else len(set(true_labels_for_eval))
            avg_strat = 'weighted' if num_classes > 2 else 'binary'
            metrics['f1'] = f1_score(true_labels_for_eval, y_pred_eval, average=avg_strat, zero_division=0)
            metrics['precision'] = precision_score(true_labels_for_eval, y_pred_eval, average=avg_strat, zero_division=0)
            metrics['recall'] = recall_score(true_labels_for_eval, y_pred_eval, average=avg_strat, zero_division=0)
            if hasattr(model, "predict_proba"):
                y_prob_eval = model.predict_proba(data_to_evaluate_on)
                if y_prob_eval.shape[1] == 2:
                    metrics['roc_auc'] = roc_auc_score(true_labels_for_eval, y_prob_eval[:, 1])
                elif y_prob_eval.shape[1] > 2:
                    try:
                        metrics['roc_auc_ovr'] = roc_auc_score(true_labels_for_eval, y_prob_eval, multi_class='ovr', average='weighted')
                    except ValueError as e:
                        print(f"计算多分类ROC AUC (OVR)失败: {e}")
                        metrics['roc_auc_ovr'] = None 
            y_pred_test = y_pred_eval
            y_prob_test = y_prob_eval if 'y_prob_eval' in locals() else None
        else: 
            metrics['mse'] = mean_squared_error(true_labels_for_eval, y_pred_eval)
            metrics['r2'] = r2_score(true_labels_for_eval, y_pred_eval)
            y_pred_test = y_pred_eval
        
        print(f"在 {'测试集' if X_test is not None else '训练集'} 上的评估指标 ({target_key}): {metrics}")
    else:
        print("没有可用的测试集或训练集数据进行评估。")

    model_results['metrics'] = metrics
    
    task_type_for_viz = 'classification' if model_type.endswith('_classifier') else 'regression'
    viz_output_dir = os.path.join(output_dir, "image", "风险预测", target_key.replace(" ", "_").replace("/", "_"))
    os.makedirs(viz_output_dir, exist_ok=True)
    
    if y_pred_test is not None and true_labels_for_eval is not None:
        model_perf_plots = visualize_model_performance(
            true_labels_for_eval, 
            y_pred_test, 
            y_prob=y_prob_test, 
            task_type=task_type_for_viz, 
            output_dir=viz_output_dir, 
            model_name=target_key
        )
        model_results['performance_plots'] = model_perf_plots
    else:
        print("沒有預測結果或真實標籤可用於模型性能可視化。") # Typo: 没有

    if SHAP_AVAILABLE:
        data_for_shap = X_test if X_test is not None and not X_test.empty else X_train
        if data_for_shap is not None and not data_for_shap.empty:
            explainer, shap_values_calculated = explain_model_predictions(model, data_for_shap, feature_names=list(data_for_shap.columns))
            if explainer and shap_values_calculated is not None:
                model_results['shap_explainer'] = explainer
                model_results['shap_values'] = shap_values_calculated
                
                # Select a subset of X_test for SHAP visualization to avoid excessive computation/large plots
                num_shap_samples = min(100, data_for_shap.shape[0])
                X_shap_sample = data_for_shap.sample(n=num_shap_samples, random_state=42) if num_shap_samples > 0 else data_for_shap

                print(f"DEBUG: run_risk_prediction - About to call visualize_shap_values for {target_key}")
                print(f"  X_shap_sample.shape: {X_shap_sample.shape if hasattr(X_shap_sample, 'shape') else 'X_shap_sample no shape attribute'}")
                if isinstance(X_shap_sample, pd.DataFrame):
                    print(f"  X_shap_sample is DataFrame, empty: {X_shap_sample.empty}")

                shap_plots = visualize_shap_values(
                    explainer,
                    shap_values_calculated,
                    X_shap_sample, 
                    output_dir=viz_output_dir, 
                    model_name=target_key
                )
                model_results['shap_plots'] = shap_plots
            else:
                print(f"SHAP解释失败 for {target_key}")
        else:
            print("沒有可用於SHAP解釋的數據。") # Typo: 没有
    else:
        print("SHAP库不可用，跳过SHAP分析。")

    print(f"==== 完成风险预测: {target_key} ====\n")
    return model_results


if __name__ == '__main__':
    print("正在创建示例数据进行测试...")
    num_samples = 200
    data = pd.DataFrame({
        'age': np.random.randint(30, 80, num_samples),
        'gender': np.random.choice([0, 1], num_samples), 
        '收缩压': np.random.randint(90, 180, num_samples),
        '舒张压': np.random.randint(60, 110, num_samples),
        'bmi': np.random.uniform(18, 35, num_samples),
        'cfpwv_速度': np.random.uniform(5, 15, num_samples), 
        '身高': np.random.uniform(150, 190, num_samples),
        '体重': np.random.uniform(45, 100, num_samples),
        '综合风险得分': np.random.rand(num_samples)
    })
    conditions = [
        (data['收缩压'] < 120) & (data['舒张压'] < 80),
        (data['收缩压'] < 130) & (data['舒张压'] < 85),
        (data['收缩压'] < 140) | (data['舒张压'] < 90),
        (data['收缩压'] < 160) | (data['舒张压'] < 100),
        (data['收缩压'] < 180) | (data['舒张压'] < 110),
        (data['收缩压'] >= 180) | (data['舒张压'] >= 110)
    ]
    choices_numeric = [0, 0, 1, 2, 3, 4] 
    data['血压状态'] = np.select(conditions, choices_numeric, default=1)
    data['血压状态'] = data['血压状态'].astype('category')

    print("示例数据创建完毕:")
    print(data.head())
    print(data.info())

    print("\n--- 测试高血压风险预测 ---")
    results_hypertension = run_risk_prediction(data.copy(), target_key='高血压风险', output_dir='output_test/risk_prediction_test_refactored')
    if results_hypertension:
        print("高血压风险预测完成。结果摘要:")
        print(f"  模型类型: {results_hypertension.get('model_type')}")
        print(f"  评估指标: {results_hypertension.get('metrics')}")
        if results_hypertension.get('shap_plots'):
            print(f"  SHAP图已生成: {len(results_hypertension['shap_plots'])} 张")

    print("\n--- 测试PWV超标风险预测 ---")
    results_pwv = run_risk_prediction(data.copy(), target_key='PWV超标风险', output_dir='output_test/risk_prediction_test_refactored')
    if results_pwv:
        print("PWV超标风险预测完成。结果摘要:")
        print(f"  模型类型: {results_pwv.get('model_type')}")
        print(f"  评估指标: {results_pwv.get('metrics')}")
        if results_pwv.get('shap_plots'):
            print(f"  SHAP图已生成: {len(results_pwv['shap_plots'])} 张")

    print("\n--- 测试高综合风险预测 ---")
    results_综合 = run_risk_prediction(data.copy(), target_key='高综合风险', output_dir='output_test/risk_prediction_test_refactored')
    if results_综合:
        print("高综合风险预测完成。结果摘要:")
        print(f"  模型类型: {results_综合.get('model_type')}")
        print(f"  评估指标: {results_综合.get('metrics')}")
        if results_综合.get('shap_plots'):
            print(f"  SHAP图已生成: {len(results_综合['shap_plots'])} 张")

    print("\n风险预测模块测试完成。检查 output_test/risk_prediction_test_refactored 目录下的输出。")