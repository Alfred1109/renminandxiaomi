#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模型训练模块：用于训练不同类型的机器学习模型
包含模型选择、参数配置和训练流程
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def train_risk_prediction_model(X, y, model_type='xgboost'):
    """
    训练风险预测模型
    
    参数:
        X: 特征DataFrame
        y: 目标Series
        model_type: 模型类型，可选 'xgboost', 'random_forest', 'logistic', 'gradient_boosting',
                   'ensemble', 'lightgbm', 'catboost'
    
    返回:
        训练好的模型和评估指标
    """
    if X is None or y is None:
        print("错误: 无法训练模型，特征或目标为空")
        return None, {}
    
    print(f"\n使用 {model_type} 模型训练风险预测模型...")
    
    # 检查特征和目标的有效性
    if X.shape[0] < 10:
        print(f"警告: 样本量太小 ({X.shape[0]} 行)，模型训练可能不稳定")
    if X.shape[1] < 2:
        print(f"警告: 特征太少 ({X.shape[1]} 列)，模型性能可能受限")
    
    # 尝试导入可选的高级模型库
    lightgbm_available = False
    catboost_available = False
    try:
        import lightgbm as lgb
        lightgbm_available = True
    except ImportError:
        if model_type == 'lightgbm':
            print("警告: 请先安装LightGBM库: pip install lightgbm")
            print("将使用XGBoost作为替代")
            model_type = 'xgboost'
    
    try:
        import catboost
        catboost_available = True
    except ImportError:
        if model_type == 'catboost':
            print("警告: 请先安装CatBoost库: pip install catboost")
            print("将使用XGBoost作为替代")
            model_type = 'xgboost'
    
    # 确定任务类型
    n_classes = len(np.unique(y))
    is_binary = n_classes == 2
    is_multiclass = n_classes > 2
    is_regression = not is_binary and not is_multiclass
    
    task_type = "二分类" if is_binary else "多分类" if is_multiclass else "回归"
    print(f"任务类型: {task_type} (类别数: {n_classes})")
    
    # 数据划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, 
        stratify=y if (is_binary or is_multiclass) else None
    )
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    
    # 创建模型
    model = create_model(model_type, is_binary, is_multiclass, n_classes, lightgbm_available, catboost_available)
    
    # 创建预处理管道
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    # 训练模型
    print("训练模型...")
    try:
        pipeline.fit(X_train, y_train)
        print("模型训练成功")
    except Exception as e:
        print(f"模型训练失败: {str(e)}")
        print("尝试使用基础XGBoost模型...")
        
        # 如果原模型失败，回退到基础XGBoost
        simple_model = xgb.XGBClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        ) if (is_binary or is_multiclass) else xgb.XGBRegressor(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', simple_model)
        ])
        
        try:
            pipeline.fit(X_train, y_train)
            print("基础XGBoost模型训练成功")
        except Exception as e2:
            print(f"基础模型训练也失败: {str(e2)}")
            return None, {}
    
    # 准备结果
    results = {
        'model': pipeline,
        'feature_names': X.columns.tolist() if hasattr(X, 'columns') else None,
        'is_binary': is_binary,
        'is_multiclass': is_multiclass,
        'is_regression': is_regression,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': pipeline.predict(X_test)
    }
    
    # 如果模型支持概率预测，添加预测概率
    if hasattr(pipeline, "predict_proba") and (is_binary or is_multiclass):
        results['y_prob'] = pipeline.predict_proba(X_test)
    
    print("模型训练完成")
    return pipeline, results

def create_model(model_type, is_binary, is_multiclass, n_classes, lightgbm_available=False, catboost_available=False):
    """
    根据指定类型创建机器学习模型
    
    参数:
        model_type: 模型类型
        is_binary: 是否为二分类任务
        is_multiclass: 是否为多分类任务
        n_classes: 类别数量
        lightgbm_available: LightGBM是否可用
        catboost_available: CatBoost是否可用
    
    返回:
        创建好的模型实例
    """
    if model_type == 'xgboost':
        if is_binary:
            model = xgb.XGBClassifier(
                n_estimators=100, 
                learning_rate=0.05,
                max_depth=4,
                min_child_weight=2,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0,
                objective='binary:logistic',
                random_state=42
            )
        elif is_multiclass:
            model = xgb.XGBClassifier(
                n_estimators=100, 
                learning_rate=0.05,
                max_depth=4,
                min_child_weight=2,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0,
                objective='multi:softprob',
                num_class=n_classes,
                random_state=42
            )
        else:  # 回归
            model = xgb.XGBRegressor(
                n_estimators=100, 
                learning_rate=0.05,
                max_depth=4,
                min_child_weight=2,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0,
                objective='reg:squarederror',
                random_state=42
            )
    
    elif model_type == 'random_forest':
        if is_binary or is_multiclass:
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                bootstrap=True,
                random_state=42,
                n_jobs=-1  # 使用所有CPU核心
            )
        else:  # 回归
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                bootstrap=True,
                random_state=42,
                n_jobs=-1  # 使用所有CPU核心
            )
            
    elif model_type == 'gradient_boosting':
        if is_binary:
            model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                min_samples_split=2,
                min_samples_leaf=1,
                subsample=1.0,
                max_features=None,
                random_state=42
            )
        elif is_multiclass:
            model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                min_samples_split=2,
                min_samples_leaf=1,
                subsample=1.0,
                max_features=None,
                random_state=42
            )
        else:  # 回归
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                min_samples_split=2,
                min_samples_leaf=1,
                subsample=1.0,
                max_features=None,
                random_state=42
            )
            
    elif model_type == 'logistic':
        if is_binary:
            model = LogisticRegression(
                C=1.0,
                penalty='l2',
                solver='lbfgs',
                max_iter=1000,
                random_state=42,
                n_jobs=-1  # 使用所有CPU核心
            )
        elif is_multiclass:
            model = LogisticRegression(
                C=1.0,
                penalty='l2',
                solver='lbfgs',
                multi_class='multinomial',
                max_iter=1000,
                random_state=42,
                n_jobs=-1  # 使用所有CPU核心
            )
        else:  # 回归任务不适用逻辑回归
            print("警告: 逻辑回归不适用于回归任务，将使用线性回归")
            model = LinearRegression(n_jobs=-1)
    
    elif model_type == 'lightgbm' and lightgbm_available:
        import lightgbm as lgb
        if is_binary:
            model = lgb.LGBMClassifier(
                boosting_type='gbdt',
                num_leaves=31,
                max_depth=-1,
                learning_rate=0.05,
                n_estimators=100,
                objective='binary',
                random_state=42,
                n_jobs=-1
            )
        elif is_multiclass:
            model = lgb.LGBMClassifier(
                boosting_type='gbdt',
                num_leaves=31,
                max_depth=-1,
                learning_rate=0.05,
                n_estimators=100,
                objective='multiclass',
                num_class=n_classes,
                random_state=42,
                n_jobs=-1
            )
        else:  # 回归
            model = lgb.LGBMRegressor(
                boosting_type='gbdt',
                num_leaves=31,
                max_depth=-1,
                learning_rate=0.05,
                n_estimators=100,
                objective='regression',
                random_state=42,
                n_jobs=-1
            )
    
    elif model_type == 'catboost' and catboost_available:
        import catboost
        if is_binary:
            model = catboost.CatBoostClassifier(
                iterations=100,
                learning_rate=0.05,
                depth=6,
                loss_function='Logloss',
                random_seed=42,
                thread_count=-1,
                verbose=0
            )
        elif is_multiclass:
            model = catboost.CatBoostClassifier(
                iterations=100,
                learning_rate=0.05,
                depth=6,
                loss_function='MultiClass',
                random_seed=42,
                thread_count=-1,
                verbose=0
            )
        else:  # 回归
            model = catboost.CatBoostRegressor(
                iterations=100,
                learning_rate=0.05,
                depth=6,
                loss_function='RMSE',
                random_seed=42,
                thread_count=-1,
                verbose=0
            )
    
    elif model_type == 'ensemble':
        # 集成多个模型以提高性能
        print("构建模型集成...")
        
        from sklearn.ensemble import VotingClassifier, VotingRegressor
        
        if is_binary or is_multiclass:
            # 创建多个基础分类器
            estimators = []
            
            # 随机森林
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            estimators.append(('rf', rf))
            
            # 梯度提升
            gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
            estimators.append(('gb', gb))
            
            # XGBoost
            xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
            estimators.append(('xgb', xgb_model))
            
            # LightGBM（如果可用）
            if lightgbm_available:
                import lightgbm as lgb
                lgbm = lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                estimators.append(('lgbm', lgbm))
            
            # 对于二分类，使用概率进行软投票
            if is_binary:
                model = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
            else:
                # 对于多分类，可能也使用软投票
                model = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
        
        else:  # 回归
            # 创建多个基础回归器
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            
            estimators = []
            
            # 随机森林
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            estimators.append(('rf', rf))
            
            # 梯度提升
            gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
            estimators.append(('gb', gb))
            
            # XGBoost
            xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
            estimators.append(('xgb', xgb_model))
            
            # LightGBM（如果可用）
            if lightgbm_available:
                import lightgbm as lgb
                lgbm = lgb.LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                estimators.append(('lgbm', lgbm))
            
            model = VotingRegressor(estimators=estimators, n_jobs=-1)
    
    else:
        print(f"警告: 未知模型类型 '{model_type}'，使用XGBoost")
        model = xgb.XGBClassifier(n_estimators=100, random_state=42) if (is_binary or is_multiclass) else xgb.XGBRegressor(n_estimators=100, random_state=42)
    
    return model 