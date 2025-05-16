#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模型核心模块：包含模型训练的核心逻辑
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

# (如果这些函数直接使用字体属性，则需要从本地项目中导入)
# from .data_visualization import CHINESE_FONT_PROP 

def train_risk_prediction_model(X, y, model_type='xgboost'):
    """
    训练风险预测模型
    
    参数:
        X: 特征矩阵
        y: 目标变量
        model_type: 模型类型
    
    返回:
        训练好的模型对象
    """
    if X is None or y is None:
        print("错误: 特征或目标为空")
        return None
    
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
    else:
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    if hasattr(y, 'dtype') and (pd.api.types.is_categorical_dtype(y) or pd.api.types.is_object_dtype(y)):
        y = y.astype('category').cat.codes
    
    if hasattr(X, 'isna'):
        if X.isna().any().any():
            print(f"检测到 {X.isna().sum().sum()} 个缺失值，将进行处理")
    
    if isinstance(X, pd.DataFrame) and not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
        print("警告: 特征包含非数值列，将尝试转换")
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                    X[col] = X[col].fillna(X[col].median())
                except:
                    print(f"警告: 无法将列 {col} 转换为数值类型，将使用标签编码")
                    if hasattr(X[col], 'astype'):
                        X[col] = X[col].astype('category').cat.codes
    
    if len(np.unique(y)) < 10 or model_type in ['xgboost_classifier', 'logistic_regression', 'random_forest_classifier']:
        task_type = '分类'
        n_classes = len(np.unique(y))
        is_binary = n_classes == 2
        is_multiclass = n_classes > 2
    else:
        task_type = '回归'
        is_binary = False
        is_multiclass = False
    
    task_type_str = "二分类" if is_binary else "多分类" if is_multiclass else "回归" # Renamed to avoid conflict with task_type variable name
    print(f"任务类型: {task_type_str}, 样本数: {X.shape[0]}")
    
    if X.shape[0] > 50:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if (is_binary or is_multiclass) else None
        )
        print(f"训练集: {X_train.shape[0]}样本, 验证集: {X_val.shape[0]}样本")
    else:
        X_train, y_train = X, y
        X_val, y_val = None, None
        print(f"样本量较小 ({X.shape[0]}), 使用全部数据训练")
    
    class_weights = None
    if is_binary:
        class_counts = np.bincount(y_train.astype(int))
        if len(class_counts) > 1 and (class_counts[0] / sum(class_counts) > 0.7 or class_counts[0] / sum(class_counts) < 0.3):
            print(f"检测到类别不平衡 (正例占比: {class_counts[1] / sum(class_counts):.2f})")
            class_weights = {
                0: 1.0 / class_counts[0] * sum(class_counts) / 2.0 if class_counts[0] > 0 else 1.0,
                1: 1.0 / class_counts[1] * sum(class_counts) / 2.0 if class_counts[1] > 0 else 1.0
            }
            print(f"应用类别权重: {class_weights}")
    
    pipeline = None
    try:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        model = None
        if model_type == 'xgboost_classifier':
            if is_binary:
                print("创建XGBoost二分类模型")
                model = xgb.XGBClassifier(
                    n_estimators=100, learning_rate=0.1, max_depth=4, min_child_weight=2,
                    subsample=0.8, colsample_bytree=0.8, gamma=0, reg_alpha=0.1, reg_lambda=1,
                    scale_pos_weight= (class_weights[1] / class_weights[0]) if class_weights and class_weights[0] > 0 else 1,
                    random_state=42, n_jobs=-1
                )
            else:
                print("创建XGBoost多分类模型")
                model = xgb.XGBClassifier(
                    n_estimators=100, learning_rate=0.1, max_depth=4, min_child_weight=2,
                    subsample=0.8, colsample_bytree=0.8, gamma=0, reg_alpha=0.1, reg_lambda=1,
                    random_state=42, n_jobs=-1
                )
        elif model_type == 'xgboost_regressor':
            print("创建XGBoost回归模型")
            model = xgb.XGBRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=4, min_child_weight=2,
                subsample=0.8, colsample_bytree=0.8, gamma=0, reg_alpha=0.1, reg_lambda=1,
                random_state=42, n_jobs=-1
            )
        elif model_type == 'random_forest':
            if is_binary or is_multiclass:
                print("创建随机森林分类模型")
                model = RandomForestClassifier(
                    n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                    random_state=42, n_jobs=-1,
                    class_weight='balanced' if is_binary and class_weights else None
                )
            else:
                print("创建随机森林回归模型")
                model = RandomForestRegressor(
                    n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                    random_state=42, n_jobs=-1
                )
        elif model_type == 'logistic_regression':
            if is_binary:
                print("创建逻辑回归模型")
                model = LogisticRegression(
                    C=1.0, penalty='l2', solver='liblinear', random_state=42, max_iter=1000,
                    class_weight='balanced' if class_weights else None
                )
            else:
                print("错误: 多分类问题请使用'xgboost_classifier'而非'logistic_regression'")
                return None
        elif model_type == 'linear_regression':
            print("创建线性回归模型")
            model = LinearRegression()
        else:
            print(f"错误: 未知的模型类型 '{model_type}'")
            return None
    
        pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', model)
        ])
    
        print("正在训练模型...")
        if X_val is not None and y_val is not None:
            pipeline.fit(X_train, y_train)
            if is_binary or is_multiclass:
                from sklearn.metrics import accuracy_score # Local import
                train_pred = pipeline.predict(X_train)
                val_pred = pipeline.predict(X_val)
                train_acc = accuracy_score(y_train, train_pred)
                val_acc = accuracy_score(y_val, val_pred)
                print(f"训练集准确率: {train_acc:.4f}, 验证集准确率: {val_acc:.4f}")
                if train_acc - val_acc > 0.2:
                    print("警告: 可能存在过拟合")
            else:
                from sklearn.metrics import mean_squared_error, r2_score # Local import
                train_pred = pipeline.predict(X_train)
                val_pred = pipeline.predict(X_val)
                train_mse = mean_squared_error(y_train, train_pred)
                train_r2 = r2_score(y_train, train_pred)
                val_mse = mean_squared_error(y_val, val_pred)
                val_r2 = r2_score(y_val, val_pred)
                print(f"训练集 - MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
                print(f"验证集 - MSE: {val_mse:.4f}, R²: {val_r2:.4f}")
                if train_r2 - val_r2 > 0.3:
                    print("警告: 可能存在过拟合")
        else:
            pipeline.fit(X_train, y_train)
            if is_binary or is_multiclass:
                from sklearn.metrics import accuracy_score # Local import
                train_pred = pipeline.predict(X_train)
                train_acc = accuracy_score(y_train, train_pred)
                print(f"训练集准确率: {train_acc:.4f}")
            else:
                from sklearn.metrics import mean_squared_error, r2_score # Local import
                train_pred = pipeline.predict(X_train)
                train_mse = mean_squared_error(y_train, train_pred)
                train_r2 = r2_score(y_train, train_pred)
                print(f"训练集 - MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
    
    except Exception as e:
        print(f"模型训练失败: {str(e)}")
        return None
    
    results_info = { # Renamed from results to avoid conflict
        'feature_names': feature_names,
        'task_type': task_type_str, # Use the renamed string variable
        'is_binary': is_binary,
        'is_multiclass': is_multiclass,
    }
    
    if hasattr(pipeline, "predict_proba") and (is_binary or is_multiclass):
        if X_val is not None:
            results_info['y_prob_val'] = pipeline.predict_proba(X_val)
    
    print("模型训练完成")
    # Return the pipeline and additional info if needed, or just the pipeline
    # For now, let's stick to returning the pipeline as before, info can be derived again if needed.
    return pipeline 