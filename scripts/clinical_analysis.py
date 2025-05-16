#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
临床分析模块：提供PWV数据的临床解释和健康风险评估
包含临床阈值定义、风险分层和预后评估
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# PWV临床阈值
# 来源：相关研究文献和临床指南
PWV_THRESHOLDS = {
    '<30': {'正常': 8.0, '边缘': 9.0, '异常': 10.0},
    '30-39': {'正常': 8.5, '边缘': 9.5, '异常': 10.5},
    '40-49': {'正常': 9.0, '边缘': 10.0, '异常': 11.0},
    '50-59': {'正常': 10.0, '边缘': 11.0, '异常': 12.0},
    '60-69': {'正常': 11.5, '边缘': 12.5, '异常': 13.5},
    '70+': {'正常': 13.0, '边缘': 14.0, '异常': 15.0}
}

def classify_pwv_risk(df):
    """
    根据年龄和临床阈值对PWV值进行风险分类
    
    参数:
        df: 数据DataFrame
    
    返回:
        添加了PWV风险分类的DataFrame
    """
    print("\n执行PWV风险分类...")
    
    # 确保数据中有PWV值
    if 'pwv' not in df.columns:
        print("错误: 数据中没有PWV值")
        return df
    
    # 确保数据中有年龄组列
    if '年龄组' not in df.columns:
        # 如果没有，尝试创建
        if 'age' in df.columns:
            age_bins = [0, 30, 40, 50, 60, 70, 120]
            age_labels = ['<30', '30-39', '40-49', '50-59', '60-69', '70+']
            df['年龄组'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)
        else:
            print("错误: 数据中没有年龄信息，无法进行风险分类")
            return df
    
    # 创建PWV风险分类列
    df['PWV风险'] = '未知'
    
    for age_group, thresholds in PWV_THRESHOLDS.items():
        mask = df['年龄组'] == age_group
        
        # 正常PWV值
        df.loc[mask & (df['pwv'] <= thresholds['正常']), 'PWV风险'] = '正常'
        
        # 边缘PWV值
        df.loc[mask & (df['pwv'] > thresholds['正常']) & 
               (df['pwv'] <= thresholds['边缘']), 'PWV风险'] = '边缘'
        
        # 异常但非显著风险
        df.loc[mask & (df['pwv'] > thresholds['边缘']) & 
               (df['pwv'] <= thresholds['异常']), 'PWV风险'] = '轻度风险'
        
        # 显著风险
        df.loc[mask & (df['pwv'] > thresholds['异常']), 'PWV风险'] = '显著风险'
    
    # 显示风险分类统计
    risk_counts = df['PWV风险'].value_counts()
    risk_percent = df['PWV风险'].value_counts(normalize=True) * 100
    
    print("\nPWV风险分类统计:")
    print(risk_counts)
    print("\nPWV风险分类百分比:")
    print(risk_percent)
    
    return df

def create_composite_risk_score(df):
    """
    创建综合风险评分（结合PWV、血压、BMI等因素）
    
    参数:
        df: 数据DataFrame
    
    返回:
        添加了综合风险评分的DataFrame
    """
    print("\n计算综合风险评分...")
    
    # 检查必要的列是否存在
    required_columns = ['pwv', 'sbp', 'dbp', 'bmi', 'age']
    for col in required_columns:
        if col not in df.columns:
            print(f"错误: 数据中缺少 {col} 列")
            return df
    
    # 初始化风险评分
    df['综合风险评分'] = 0
    
    # 1. PWV风险评分
    if 'PWV风险' not in df.columns:
        df = classify_pwv_risk(df)
    
    # 根据PWV风险分类添加分数
    pwv_risk_scores = {'正常': 0, '边缘': 1, '轻度风险': 2, '显著风险': 3, '未知': 1.5}
    for risk, score in pwv_risk_scores.items():
        df.loc[df['PWV风险'] == risk, '综合风险评分'] += score
    
    # 2. 血压风险评分
    # 收缩压评分
    df.loc[df['sbp'] < 120, '综合风险评分'] += 0
    df.loc[(df['sbp'] >= 120) & (df['sbp'] < 140), '综合风险评分'] += 1
    df.loc[(df['sbp'] >= 140) & (df['sbp'] < 160), '综合风险评分'] += 2
    df.loc[df['sbp'] >= 160, '综合风险评分'] += 3
    
    # 舒张压评分
    df.loc[df['dbp'] < 80, '综合风险评分'] += 0
    df.loc[(df['dbp'] >= 80) & (df['dbp'] < 90), '综合风险评分'] += 1
    df.loc[(df['dbp'] >= 90) & (df['dbp'] < 100), '综合风险评分'] += 2
    df.loc[df['dbp'] >= 100, '综合风险评分'] += 3
    
    # 3. BMI风险评分
    df.loc[df['bmi'] < 18.5, '综合风险评分'] += 0.5  # 偏瘦
    df.loc[(df['bmi'] >= 18.5) & (df['bmi'] < 24), '综合风险评分'] += 0  # 正常
    df.loc[(df['bmi'] >= 24) & (df['bmi'] < 28), '综合风险评分'] += 1  # 超重
    df.loc[df['bmi'] >= 28, '综合风险评分'] += 2  # 肥胖
    
    # 4. 年龄风险评分
    df.loc[df['age'] < 40, '综合风险评分'] += 0
    df.loc[(df['age'] >= 40) & (df['age'] < 50), '综合风险评分'] += 0.5
    df.loc[(df['age'] >= 50) & (df['age'] < 60), '综合风险评分'] += 1
    df.loc[(df['age'] >= 60) & (df['age'] < 70), '综合风险评分'] += 1.5
    df.loc[df['age'] >= 70, '综合风险评分'] += 2
    
    # 5. 创建风险等级
    df['综合风险等级'] = pd.cut(
        df['综合风险评分'],
        bins=[0, 2, 4, 6, float('inf')],
        labels=['低风险', '中等风险', '高风险', '极高风险'],
        right=False
    )

    # Create a binary flag for high/very high comprehensive risk
    df['high_comprehensive_risk_flag'] = df['综合风险等级'].apply(lambda x: 1 if x in ['高风险', '极高风险'] else 0)
    print(f"已创建'high_comprehensive_risk_flag'特征。分布:\n{df['high_comprehensive_risk_flag'].value_counts(normalize=True)}")
    
    # 显示风险评分统计
    print("\n综合风险评分统计:")
    print(df['综合风险评分'].describe())
    
    # 显示风险等级统计
    risk_level_counts = df['综合风险等级'].value_counts()
    risk_level_percent = df['综合风险等级'].value_counts(normalize=True) * 100
    
    print("\n综合风险等级统计:")
    print(risk_level_counts)
    print("\n综合风险等级百分比:")
    print(risk_level_percent)
    
    return df

def calculate_cv_risk_score(df):
    """
    计算10年心血管疾病风险评分
    基于PWV和其他传统风险因素的简化模型
    
    参数:
        df: 数据DataFrame
    
    返回:
        添加了10年CVD风险的DataFrame
    """
    print("\n计算10年心血管疾病风险评分...")
    
    # 检查必要的列是否存在
    if 'pwv' not in df.columns or 'age' not in df.columns or 'gender' not in df.columns:
        print("错误: 数据中缺少必要的列")
        return df
    
    # 初始化基础风险分数
    df['基础风险分数'] = 0
    
    # 1. 年龄评分
    for gender_val in [0, 1]:  # 0-女性, 1-男性
        gender_mask = df['gender'] == gender_val
        # 如果当前性别的子集为空，则跳过，避免对空Series操作
        if gender_mask.sum() == 0:
            continue
        age_series_subset = df.loc[gender_mask, 'age'] # 提取对应性别的年龄数据
        if gender_val == 0:  # 女性
            scores = np.where(
                age_series_subset < 35, 0,
                np.where(age_series_subset < 45, 2,
                np.where(age_series_subset < 55, 4,
                np.where(age_series_subset < 65, 6,
                np.where(age_series_subset < 75, 8, 10))))
            )
            df.loc[gender_mask, '基础风险分数'] = scores
        else:  # 男性
            scores = np.where(
                age_series_subset < 35, 0,
                np.where(age_series_subset < 45, 3,
                np.where(age_series_subset < 55, 6,
                np.where(age_series_subset < 65, 8,
                np.where(age_series_subset < 75, 10, 12))))
            )
            df.loc[gender_mask, '基础风险分数'] = scores
    
    # 2. 血压评分
    if 'sbp' in df.columns:
        df['血压评分'] = np.where(
            df['sbp'] < 120, 0,
            np.where(df['sbp'] < 140, 1,
            np.where(df['sbp'] < 160, 2,
            np.where(df['sbp'] < 180, 3, 4)))
        )
        df['基础风险分数'] += df['血压评分']
    
    # 3. PWV评分 (基于PWV与年龄组的关系)
    df['PWV评分'] = 0
    
    # 确保数据中有年龄组列
    if '年龄组' not in df.columns:
        # 如果没有，创建
        age_bins = [0, 30, 40, 50, 60, 70, 120]
        age_labels = ['<30', '30-39', '40-49', '50-59', '60-69', '70+']
        df['年龄组'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)
    
    # 对每个年龄组应用相应的PWV评分
    for age_group, thresholds in PWV_THRESHOLDS.items():
        mask = df['年龄组'] == age_group
        
        # PWV在正常范围
        df.loc[mask & (df['pwv'] <= thresholds['正常']), 'PWV评分'] = 0
        
        # PWV在边缘范围
        df.loc[mask & (df['pwv'] > thresholds['正常']) & 
               (df['pwv'] <= thresholds['边缘']), 'PWV评分'] = 1
        
        # PWV轻度异常
        df.loc[mask & (df['pwv'] > thresholds['边缘']) & 
               (df['pwv'] <= thresholds['异常']), 'PWV评分'] = 2
        
        # PWV显著异常
        df.loc[mask & (df['pwv'] > thresholds['异常']), 'PWV评分'] = 3
    
    # 加入PWV评分到基础风险
    df['基础风险分数'] += df['PWV评分']
    
    # 4. 计算10年CVD风险
    # 确保风险分数在有效范围内
    df['基础风险分数'] = df['基础风险分数'].clip(0, 25)
    
    # 初始化10年CVD风险列
    df['10年CVD风险(%)'] = 0.0
    
    # 逐行处理风险映射，使用if-elif序列而非字典
    for idx in df.index:
        score = df.at[idx, '基础风险分数']
        # 跳过NaN值
        if pd.isna(score):
            continue
        
        # 四舍五入转换为整数
        score_int = int(round(score))
        
        # 确保在有效范围内
        if score_int < 0:
            score_int = 0
        elif score_int > 25:
            score_int = 25
        
        # 使用if-elif序列映射风险值
        if score_int == 0:
            df.at[idx, '10年CVD风险(%)'] = 1.0
        elif score_int == 1:
            df.at[idx, '10年CVD风险(%)'] = 1.2
        elif score_int == 2:
            df.at[idx, '10年CVD风险(%)'] = 1.4
        elif score_int == 3:
            df.at[idx, '10年CVD风险(%)'] = 1.6
        elif score_int == 4:
            df.at[idx, '10年CVD风险(%)'] = 1.9
        elif score_int == 5:
            df.at[idx, '10年CVD风险(%)'] = 2.3
        elif score_int == 6:
            df.at[idx, '10年CVD风险(%)'] = 2.8
        elif score_int == 7:
            df.at[idx, '10年CVD风险(%)'] = 3.3
        elif score_int == 8:
            df.at[idx, '10年CVD风险(%)'] = 3.9
        elif score_int == 9:
            df.at[idx, '10年CVD风险(%)'] = 4.7
        elif score_int == 10:
            df.at[idx, '10年CVD风险(%)'] = 5.6
        elif score_int == 11:
            df.at[idx, '10年CVD风险(%)'] = 6.7
        elif score_int == 12:
            df.at[idx, '10年CVD风险(%)'] = 7.9
        elif score_int == 13:
            df.at[idx, '10年CVD风险(%)'] = 9.4
        elif score_int == 14:
            df.at[idx, '10年CVD风险(%)'] = 11.2
        elif score_int == 15:
            df.at[idx, '10年CVD风险(%)'] = 13.3
        elif score_int == 16:
            df.at[idx, '10年CVD风险(%)'] = 15.6
        elif score_int == 17:
            df.at[idx, '10年CVD风险(%)'] = 18.4
        elif score_int == 18:
            df.at[idx, '10年CVD风险(%)'] = 21.6
        elif score_int == 19:
            df.at[idx, '10年CVD风险(%)'] = 25.3
        elif score_int == 20:
            df.at[idx, '10年CVD风险(%)'] = 29.4
        elif score_int == 21:
            df.at[idx, '10年CVD风险(%)'] = 34.0
        elif score_int == 22:
            df.at[idx, '10年CVD风险(%)'] = 39.2
        elif score_int == 23:
            df.at[idx, '10年CVD风险(%)'] = 45.1
        elif score_int >= 24:
            df.at[idx, '10年CVD风险(%)'] = 50.0
    
    # 5. 定义风险等级
    df['CVD风险等级'] = pd.cut(
        df['10年CVD风险(%)'],
        bins=[0, 5, 10, 20, float('inf')],
        labels=['低风险', '中等风险', '高风险', '极高风险']
    )
    
    # 显示CVD风险统计
    print("\n10年CVD风险统计:")
    print(df['10年CVD风险(%)'].describe())
    
    # 显示风险等级统计
    risk_level_counts = df['CVD风险等级'].value_counts()
    risk_level_percent = df['CVD风险等级'].value_counts(normalize=True) * 100
    
    print("\nCVD风险等级统计:")
    print(risk_level_counts)
    print("\nCVD风险等级百分比:")
    print(risk_level_percent)
    
    return df

def run_clinical_analysis(df):
    """
    运行所有临床分析
    
    参数:
        df: 数据DataFrame
    
    返回:
        包含临床分析结果的DataFrame和结果字典
    """
    print("\n开始执行临床分析...")
    
    # 初始化结果字典
    results = {}
    
    # 1. PWV风险分类
    df = classify_pwv_risk(df)
    results['pwv_risk_stats'] = {
        'counts': df['PWV风险'].value_counts(),
        'percentages': df['PWV风险'].value_counts(normalize=True) * 100
    }
    
    # 2. 综合风险评分
    df = create_composite_risk_score(df)
    results['composite_risk_stats'] = {
        'score_stats': df['综合风险评分'].describe(),
        'level_counts': df['综合风险等级'].value_counts(),
        'level_percentages': df['综合风险等级'].value_counts(normalize=True) * 100
    }
    
    # 3. 10年心血管疾病风险
    try:
        import traceback
        df = calculate_cv_risk_score(df)
        
        # 检查10年CVD风险列是否添加成功
        if '10年CVD风险(%)' in df.columns and 'CVD风险等级' in df.columns:
            results['cvd_risk_stats'] = {
                'risk_stats': df['10年CVD风险(%)'].describe(),
                'level_counts': df['CVD风险等级'].value_counts(),
                'level_percentages': df['CVD风险等级'].value_counts(normalize=True) * 100
            }
        else:
            print("⚠️ 10年CVD风险列创建失败，跳过相关统计")
    except Exception as e:
        error_message = f"❌ 计算10年CVD风险时出现错误: {e}"
        print(error_message)
        print("详细错误信息:")
        traceback.print_exc() # 打印详细堆栈信息
        print("错误位置信息结束")
        
        # 添加默认值，确保分析可继续
        if '10年CVD风险(%)' not in df.columns:
            df['10年CVD风险(%)'] = np.nan
        if 'CVD风险等级' not in df.columns:
            df['CVD风险等级'] = np.nan
    
    # 4. 年龄组与风险等级交叉分析
    age_risk_cross = pd.crosstab(
        df['年龄组'], 
        df['CVD风险等级'], 
        normalize='index'
    ) * 100
    
    results['age_risk_cross'] = age_risk_cross
    print("\n年龄组与CVD风险等级交叉分析 (%):")
    print(age_risk_cross)
    
    # 5. 高风险人群的特征分析
    high_risk = df[(df['CVD风险等级'] == '高风险') | (df['CVD风险等级'] == '极高风险')]
    if len(high_risk) > 0:
        high_risk_features = high_risk[['age', 'gender', 'bmi', 'sbp', 'dbp', 'pwv']].describe()
        results['high_risk_features'] = high_risk_features
        print("\n高风险人群特征分析:")
        print(high_risk_features)
    
    print("\n临床分析完成！")
    
    return df, results

if __name__ == '__main__':
    # 导入必要的模块
    from data_processing import load_and_prepare_data
    
    # 加载和预处理数据
    df = load_and_prepare_data()
    
    # 执行临床分析
    df_with_risk, clinical_results = run_clinical_analysis(df) 