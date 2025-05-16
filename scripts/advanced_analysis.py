#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
高级分析模块：提供PWV数据的扩展分析功能
包含子群体分析、时间序列分析和高级统计方法
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal, friedmanchisquare
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.power import TTestPower, TTestIndPower
from statsmodels.stats.proportion import proportion_confint
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

def subgroup_analysis(df, groupby_column, target_columns=None):
    """
    对数据进行子群体分析，按指定列分组并分析目标列的统计特性
    
    参数:
        df: 数据DataFrame
        groupby_column: 用于分组的列名（如'bmi分类'或'血压状态'）
        target_columns: 要分析的目标列列表
    
    返回:
        包含分析结果的字典
    """
    print(f"\n按 {groupby_column} 进行子群体分析...")
    
    # 如果未指定目标列，使用默认列
    if target_columns is None:
        target_columns = ['pwv', 'sbp', 'dbp', 'age']
    
    # 初始化结果字典
    results = {
        'summary_stats': {},
        'anova_results': {},
        'multiple_comparisons': {}
    }
    
    # 1. 计算每个子群体的描述性统计量
    subgroup_stats = df.groupby(groupby_column)[target_columns].agg(['mean', 'std', 'count', 'min', 'max'])
    results['summary_stats'] = subgroup_stats
    print(f"\n{groupby_column}子群体统计描述:")
    print(subgroup_stats)
    
    # 2. 子群体间差异的ANOVA分析
    anova_results = []
    for column in target_columns:
        # 删除缺失值
        column_data = df[[groupby_column, column]].dropna()
        if len(column_data) > 0:
            # 获取所有组
            groups = column_data.groupby(groupby_column)[column].apply(list)
            if len(groups) >= 2:  # 至少需要两个组才能进行ANOVA
                # 尝试进行常规ANOVA（参数检验）
                try:
                    f_stat, p_value = stats.f_oneway(*groups)
                    test_type = 'ANOVA'
                except:
                    # 如果ANOVA失败，尝试非参数Kruskal-Wallis检验
                    try:
                        if len(groups) == 2:
                            # 对于两个组，使用Mann-Whitney U检验
                            stat, p_value = mannwhitneyu(groups.iloc[0], groups.iloc[1])
                            test_type = 'Mann-Whitney U'
                        else:
                            # 对于多个组，使用Kruskal-Wallis H检验
                            stat, p_value = kruskal(*groups)
                            test_type = 'Kruskal-Wallis'
                        f_stat = stat  # 保存统计量
                    except:
                        f_stat = None
                        p_value = None
                        test_type = '检验失败'
                
                anova_results.append({
                    '变量': column,
                    '检验类型': test_type,
                    '统计量': f_stat,
                    'P值': p_value
                })
                
                # 3. 如果ANOVA显著，进行事后多重比较（Tukey HSD）
                if p_value is not None and p_value < 0.05 and len(groups) > 2:
                    try:
                        # 创建长格式的数据
                        data_list = []
                        for group_name, values in groups.items():
                            for value in values:
                                data_list.append([group_name, value])
                        long_data = pd.DataFrame(data_list, columns=[groupby_column, column])
                        
                        # 进行Tukey HSD事后检验
                        tukey = pairwise_tukeyhsd(endog=long_data[column], 
                                                 groups=long_data[groupby_column], 
                                                 alpha=0.05)
                        
                        # 保存多重比较结果
                        tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], 
                                               columns=tukey._results_table.data[0])
                        
                        results['multiple_comparisons'][column] = tukey_df
                        print(f"\n{column} 的事后多重比较 (Tukey HSD):")
                        print(tukey_df)
                    except Exception as e:
                        print(f"进行 {column} 的多重比较时出错: {str(e)}")
    
    # 保存ANOVA结果
    results['anova_results'] = pd.DataFrame(anova_results)
    print("\n组间差异检验结果:")
    print(results['anova_results'])
    
    return results

def create_bmi_categories(df):
    """
    根据BMI值创建BMI分类
    
    参数:
        df: 数据DataFrame
    
    返回:
        添加了'bmi分类'列的DataFrame
    """
    # 确保存在bmi列
    if 'bmi' not in df.columns:
        print("错误: 数据中没有BMI列")
        return df
    
    # 创建BMI分类
    conditions = [
        (df['bmi'] < 18.5),
        (df['bmi'] >= 18.5) & (df['bmi'] < 24.0),
        (df['bmi'] >= 24.0) & (df['bmi'] < 28.0),
        (df['bmi'] >= 28.0)
    ]
    categories = ['偏瘦', '正常', '超重', '肥胖']
    
    df['bmi分类'] = pd.cut(df['bmi'], bins=[0, 18.5, 24.0, 28.0, float('inf')], labels=categories)
    
    # 显示分类统计
    bmi_counts = df['bmi分类'].value_counts().sort_index()
    print("\nBMI分类统计:")
    print(bmi_counts)
    
    # 计算各分类的百分比
    bmi_percent = df['bmi分类'].value_counts(normalize=True).sort_index() * 100
    print("\nBMI分类百分比:")
    print(bmi_percent)
    
    return df

def create_bp_categories(df):
    """
    根据收缩压和舒张压创建血压状态分类
    
    参数:
        df: 数据DataFrame
    
    返回:
        添加了'血压状态'列的DataFrame
    """
    # 确保存在必要的列
    if 'sbp' not in df.columns or 'dbp' not in df.columns:
        print("错误: 数据中缺少血压列 (sbp or dbp)")
        return df
    
    # 创建血压分类
    conditions = [
        # 正常血压
        (df['sbp'] < 120) & (df['dbp'] < 80),
        # 正常高值
        ((df['sbp'] >= 120) & (df['sbp'] < 140)) | ((df['dbp'] >= 80) & (df['dbp'] < 90)),
        # 高血压1级
        ((df['sbp'] >= 140) & (df['sbp'] < 160)) | ((df['dbp'] >= 90) & (df['dbp'] < 100)),
        # 高血压2级
        ((df['sbp'] >= 160) & (df['sbp'] < 180)) | ((df['dbp'] >= 100) & (df['dbp'] < 110)),
        # 高血压3级
        (df['sbp'] >= 180) | (df['dbp'] >= 110)
    ]
    categories = ['正常血压', '正常高值', '高血压1级', '高血压2级', '高血压3级']
    
    df['血压状态'] = np.select(conditions, categories, default='未知')
    
    # 显示分类统计
    bp_counts = df['血压状态'].value_counts().sort_index()
    print("\n血压状态分类统计:")
    print(bp_counts)
    
    # 计算各分类的百分比
    bp_percent = df['血压状态'].value_counts(normalize=True).sort_index() * 100
    print("\n血压状态分类百分比:")
    print(bp_percent)
    
    return df

def statistical_power_analysis(df, target_column='pwv', group_column='gender'):
    """
    执行统计功效分析
    
    参数:
        df: 数据DataFrame
        target_column: 目标分析列
        group_column: 分组列
    
    返回:
        包含功效分析结果的DataFrame
    """
    print(f"\n执行统计功效分析: {target_column} by {group_column}...")
    
    # 准备数据
    data = df[[target_column, group_column]].dropna()
    if len(data) == 0:
        print("错误: 没有有效数据用于功效分析")
        return None
    
    unique_groups = data[group_column].unique()
    if len(unique_groups) < 2:
        print("错误: 需要至少两个组进行功效分析")
        return None
    
    results = []
    
    # 对两组执行功效分析
    if len(unique_groups) == 2:
        group1_data = data[data[group_column] == unique_groups[0]][target_column]
        group2_data = data[data[group_column] == unique_groups[1]][target_column]
        
        # 计算效应量 (Cohen's d)
        mean1, mean2 = group1_data.mean(), group2_data.mean()
        pooled_std = np.sqrt(((group1_data.std() ** 2) + (group2_data.std() ** 2)) / 2)
        effect_size = abs(mean1 - mean2) / pooled_std
        
        # 计算当前样本量的功效
        n1, n2 = len(group1_data), len(group2_data)
        power_analysis = TTestIndPower()
        achieved_power = power_analysis.power(effect_size=effect_size, nobs1=n1, alpha=0.05, ratio=n2/n1)
        
        # 计算达到不同功效水平所需的样本量
        for target_power in [0.8, 0.85, 0.9, 0.95]:
            required_n = power_analysis.solve_power(effect_size=effect_size, 
                                                  power=target_power, 
                                                  alpha=0.05, 
                                                  ratio=n2/n1)
            
            results.append({
                '目标功效': target_power,
                '效应量 (Cohen\'s d)': effect_size,
                '当前样本量': n1 + n2,
                '达到目标功效所需样本量': int(np.ceil(required_n * (1 + n2/n1))),
                '当前统计功效': achieved_power
            })
        
        power_df = pd.DataFrame(results)
        print("\n功效分析结果:")
        print(power_df)
        return power_df
    else:
        print("注意: 功效分析当前仅支持两组比较")
        return None

def cluster_analysis(df, features=None, n_clusters=3):
    """
    执行聚类分析以发现数据中的自然分组
    
    参数:
        df: 数据DataFrame
        features: 用于聚类的特征列表
        n_clusters: 聚类数量
    
    返回:
        包含聚类结果的DataFrame
    """
    print("\n执行聚类分析...")
    
    # 如果未指定特征，使用默认特征
    if features is None:
        features = ['age', 'bmi', '收缩压', '舒张压', 'pwv']
    
    # 准备数据
    data = df[features].dropna()
    if len(data) == 0:
        print("错误: 没有有效数据用于聚类分析")
        return None
    
    # 标准化数据
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # 执行K-means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)
    
    # 添加聚类标签到原始数据
    clustered_df = data.copy()
    clustered_df['聚类'] = clusters
    
    # 计算每个聚类的统计信息
    cluster_stats = clustered_df.groupby('聚类')[features].agg(['mean', 'std', 'count'])
    print("\n聚类分析结果:")
    print(cluster_stats)
    
    # 计算每个聚类的占比
    cluster_proportions = clustered_df['聚类'].value_counts(normalize=True) * 100
    print("\n聚类占比(%):")
    print(cluster_proportions)
    
    # 定义聚类特征
    cluster_features = []
    for cluster in range(n_clusters):
        # 获取该聚类的平均值
        cluster_mean = cluster_stats.xs('mean', level=1, axis=1)[clustered_df['聚类'] == cluster].mean()
        
        # 找出该聚类最显著的特征
        # 计算与总体平均值的差异，按绝对差异排序
        overall_mean = data.mean()
        diff = cluster_mean - overall_mean
        abs_diff = abs(diff)
        top_features = abs_diff.sort_values(ascending=False).head(3).index.tolist()
        
        # 确定每个顶级特征是高于还是低于平均值
        feature_desc = []
        for feat in top_features:
            if diff[feat] > 0:
                direction = "高"
            else:
                direction = "低"
            feature_desc.append(f"{feat}{direction}")
        
        cluster_features.append({
            '聚类': cluster,
            '样本数': len(clustered_df[clustered_df['聚类'] == cluster]),
            '占比(%)': cluster_proportions[cluster],
            '显著特征': ', '.join(feature_desc)
        })
    
    # 创建聚类特征描述DataFrame
    cluster_desc_df = pd.DataFrame(cluster_features)
    print("\n聚类特征描述:")
    print(cluster_desc_df)
    
    return {
        'clustered_data': clustered_df,
        'cluster_stats': cluster_stats,
        'cluster_description': cluster_desc_df,
        'kmeans_model': kmeans
    }

def pwv_reference_comparison(df):
    """
    将PWV值与参考范围进行比较
    
    参数:
        df: 数据DataFrame
    
    返回:
        包含参考比较结果的DataFrame
    """
    print("\n执行PWV参考值比较分析...")
    
    # PWV参考值表（按年龄段）
    # 来源: 参考文献和元分析数据
    pwv_reference = pd.DataFrame({
        '年龄组': ['<30', '30-39', '40-49', '50-59', '60-69', '70+'],
        '参考下限': [6.0, 6.5, 7.0, 8.0, 9.0, 10.0],
        '参考上限': [8.0, 8.5, 9.0, 10.0, 11.5, 13.0]
    })
    
    # 确保数据中有'年龄组'列
    if '年龄组' not in df.columns:
        # 如果没有，尝试创建
        age_bins = [0, 30, 40, 50, 60, 70, 120]
        age_labels = ['<30', '30-39', '40-49', '50-59', '60-69', '70+']
        df['年龄组'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)
    
    # 初始化结果列表
    comparison_results = []
    
    # 按年龄组分析实际PWV值与参考范围的对比
    for age_group in pwv_reference['年龄组']:
        # 当前年龄组的数据
        group_data = df[df['年龄组'] == age_group]
        if len(group_data) == 0:
            continue
            
        # 获取参考范围
        ref_row = pwv_reference[pwv_reference['年龄组'] == age_group].iloc[0]
        ref_lower = ref_row['参考下限']
        ref_upper = ref_row['参考上限']
        
        # 计算统计值
        pwv_mean = group_data['pwv'].mean()
        pwv_std = group_data['pwv'].std()
        
        # 计算超出参考范围的比例
        below_ref = (group_data['pwv'] < ref_lower).sum() / len(group_data) * 100
        above_ref = (group_data['pwv'] > ref_upper).sum() / len(group_data) * 100
        within_ref = 100 - below_ref - above_ref
        
        # 计算95%置信区间
        n = len(group_data)
        ci_lower = pwv_mean - 1.96 * pwv_std / np.sqrt(n)
        ci_upper = pwv_mean + 1.96 * pwv_std / np.sqrt(n)
        
        # 添加到结果
        comparison_results.append({
            '年龄组': age_group,
            '样本数': n,
            'PWV均值': pwv_mean,
            'PWV标准差': pwv_std,
            '95%置信区间下限': ci_lower,
            '95%置信区间上限': ci_upper,
            '参考下限': ref_lower,
            '参考上限': ref_upper,
            '低于参考范围(%)': below_ref,
            '在参考范围内(%)': within_ref,
            '高于参考范围(%)': above_ref
        })
    
    # 创建结果DataFrame
    results_df = pd.DataFrame(comparison_results)
    print("\nPWV与参考值比较结果:")
    print(results_df)
    
    return results_df

def run_advanced_analysis(df):
    """
    运行所有高级分析
    
    参数:
        df: 数据DataFrame
    
    返回:
        包含所有高级分析结果的字典
    """
    print("\n开始执行高级分析...")
    
    # 初始化结果字典
    results = {}
    
    # 1. 创建BMI分类
    df = create_bmi_categories(df)
    
    # 2. 创建血压状态分类
    df = create_bp_categories(df)
    
    # 3. 按BMI分类进行子群体分析
    results['bmi_subgroup'] = subgroup_analysis(df, 'bmi分类')
    
    # 4. 按血压状态进行子群体分析
    results['bp_subgroup'] = subgroup_analysis(df, '血压状态')
    
    # 5. PWV统计功效分析
    results['power_analysis'] = statistical_power_analysis(df)
    
    # 6. 聚类分析
    results['cluster_analysis'] = cluster_analysis(df)
    
    # 7. PWV参考值比较
    results['pwv_reference'] = pwv_reference_comparison(df)
    
    print("\n高级分析完成！")
    
    return results, df

if __name__ == '__main__':
    # 导入必要的模块
    from data_processing import load_and_prepare_data
    
    # 加载和预处理数据
    df = load_and_prepare_data()
    
    # 执行高级分析
    advanced_results, enhanced_df = run_advanced_analysis(df) 