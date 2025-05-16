#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据分析模块：提供PWV数据的基本统计分析功能
包含描述性统计、相关性分析、回归分析等功能
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.formula.api import ols
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from datetime import datetime
import warnings
from statsmodels.stats.multicomp import pairwise_tukeyhsd # Import for Tukey's HSD
warnings.filterwarnings('ignore')

def calculate_basic_statistics(df):
    """
    计算基本统计指标
    
    参数:
        df: 数据DataFrame
    
    返回:
        包含基本统计结果的DataFrame
    """
    print("\n计算基本统计指标...")
    
    # 选择需要统计的数值列 - updated and expanded with standardized names
    numeric_cols = [
        'pwv', 'age', 'sbp', 'dbp', 'bmi', 'height', 'weight',
        'cfpwv_speed', 'bapwv_right_speed', 'bapwv_left_speed',
        'cfpwv_carotid_si', 'cfpwv_carotid_ri', 'hrv_index',
        'creatinine_umol_l', 'urea_mmol_l', 'crp_mg_l', 
        'ef_percent', 'abi_right_pt_index', '脉压差', # Added 脉压差 if created
        # New additions for this round
        'cfpwv_carotid_daix', 'cfpwv_interval_ms', 'cfpwv_distance_cm',
        'bapwv_right_distance_cm', 'bapwv_left_interval_ms',
        'abi_right_brachial_index', 'bfv_carotid_mean_speed',
        'bnp_pg_ml', 'wbc_10_9', 'hb_g_l'
    ]
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    if not available_cols:
        print("警告: 未找到可用于统计的数值列")
        return None
    
    # 计算基本统计量
    stats_df = df[available_cols].describe().T
    
    # 添加其他统计量
    stats_df['变异系数'] = stats_df['std'] / stats_df['mean'] * 100  # 变异系数CV
    stats_df['中位数'] = df[available_cols].median()
    stats_df['偏度'] = df[available_cols].skew()
    stats_df['峰度'] = df[available_cols].kurtosis()
    
    # 整理列顺序
    stats_df = stats_df[['count', 'mean', 'std', '变异系数', 'min', '25%', '50%', '75%', 'max', '中位数', '偏度', '峰度']]
    
    # 重置索引，确保列名明确
    stats_df = stats_df.reset_index().rename(columns={'index': '指标'})
    
    print("基本统计指标计算完成")
    return stats_df

def analyze_gender_differences(df):
    """
    分析性别差异
    
    参数:
        df: 数据DataFrame
    
    返回:
        包含性别差异分析结果的DataFrame
    """
    print("\n分析性别差异...")
    
    # 检查是否存在性别列和足够的数据
    if 'gender' not in df.columns:
        print("警告: 数据中没有性别列")
        return None
    
    # 数值列 - updated and expanded
    numeric_cols = [
        'pwv', 'age', 'sbp', 'dbp', 'bmi', 'height', 'weight',
        'cfpwv_speed', 'bapwv_right_speed', 'bapwv_left_speed',
        'cfpwv_carotid_si', 'cfpwv_carotid_ri', 'hrv_index',
        'creatinine_umol_l', 'urea_mmol_l', 'crp_mg_l', 
        'ef_percent', 'abi_right_pt_index', '脉压差',
        # New additions for this round
        'cfpwv_carotid_daix', 'cfpwv_interval_ms', 'cfpwv_distance_cm',
        'bapwv_right_distance_cm', 'bapwv_left_interval_ms',
        'abi_right_brachial_index', 'bfv_carotid_mean_speed',
        'bnp_pg_ml', 'wbc_10_9', 'hb_g_l'
    ]
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    if not available_cols:
        print("警告: 未找到可用于性别差异分析的数值列")
        return None
    
    # 分组
    male_df = df[df['gender'] == 1]
    female_df = df[df['gender'] == 0]
    
    if len(male_df) < 5 or len(female_df) < 5:
        print(f"警告: 性别分组样本量太小 (男: {len(male_df)}, 女: {len(female_df)})")
        return None
    
    # 初始化结果列表
    results = []
    
    for col in available_cols:
        # 分离数据并移除NaN
        male_col_data = male_df[col].dropna()
        female_col_data = female_df[col].dropna()

        # 计算基本统计量
        male_mean = male_col_data.mean()
        male_std = male_col_data.std()
        male_n = len(male_col_data)
        female_mean = female_col_data.mean()
        female_std = female_col_data.std()
        female_n = len(female_col_data)
        
        # 初始化测试相关变量
        test_used = "N/A"
        t_stat, p_value = np.nan, np.nan
        normality_male_p, normality_female_p = np.nan, np.nan
        levene_p = np.nan
        cohen_d = np.nan
        assumptions_met_notes = []

        # 检查样本量是否足够进行检验
        if male_n < 5 or female_n < 5:
            assumptions_met_notes.append("样本量不足 (<5)")
            test_used = "样本量不足"
        else:
            # 1. 正态性检验 (Shapiro-Wilk)
            if male_n >= 3: # Shapiro-Wilk needs at least 3 samples
                _, normality_male_p = stats.shapiro(male_col_data)
            else:
                normality_male_p = np.nan # Not enough samples for Shapiro
                
            if female_n >= 3:
                _, normality_female_p = stats.shapiro(female_col_data)
            else:
                normality_female_p = np.nan

            is_male_normal = normality_male_p > 0.05 if pd.notna(normality_male_p) else False
            is_female_normal = normality_female_p > 0.05 if pd.notna(normality_female_p) else False

            if not is_male_normal: assumptions_met_notes.append("男性数据非正态")
            if not is_female_normal: assumptions_met_notes.append("女性数据非正态")

            # 2. 方差齐性检验 (Levene's test)
            # Levene's test requires at least 2 samples per group, already handled by male_n/female_n check
            try:
                _, levene_p = stats.levene(male_col_data, female_col_data)
            except ValueError: # Can happen if one group has zero variance
                levene_p = np.nan 
                assumptions_met_notes.append("Levene检验无法执行(可能某组方差为0)")


            variances_equal = levene_p > 0.05 if pd.notna(levene_p) else False
            if pd.notna(levene_p) and not variances_equal: assumptions_met_notes.append("方差不齐")
            
            # 3. 选择并执行检验
            if is_male_normal and is_female_normal:
                if variances_equal:
                    test_used = "Student's t-test"
                    t_stat, p_value = stats.ttest_ind(male_col_data, female_col_data, equal_var=True)
                else:
                    test_used = "Welch's t-test"
                    t_stat, p_value = stats.ttest_ind(male_col_data, female_col_data, equal_var=False)
            else: # 至少一组非正态
                test_used = "Mann-Whitney U"
                try:
                    t_stat, p_value = stats.mannwhitneyu(male_col_data, female_col_data, alternative='two-sided')
                    # Mann-Whitney U t_stat is U-statistic, not directly comparable to t-statistic in magnitude
                except ValueError as e: # Can occur if all values are identical in one group, etc.
                    assumptions_met_notes.append(f"Mann-Whitney U 执行错误: {e}")
                    p_value = np.nan


            # 4. 计算效应量 (Cohen's d for t-tests)
            if test_used in ["Student's t-test", "Welch's t-test"] and male_n > 0 and female_n > 0:
                # Pooled standard deviation for Student's t, or use individual for Welch's approx.
                # For simplicity, use a common formula for Cohen's d that works for both
                # s_pooled = np.sqrt(((male_n - 1) * male_std**2 + (female_n - 1) * female_std**2) / (male_n + female_n - 2))
                # if s_pooled == 0:
                #     cohen_d = np.nan
                # else:
                #     cohen_d = (male_mean - female_mean) / s_pooled
                # A more robust way for Cohen's d, especially with unequal variances:
                s_pooled_welch = np.sqrt((male_std**2 + female_std**2) / 2)
                if s_pooled_welch == 0 or pd.isna(s_pooled_welch): # check for NaN as well
                    cohen_d = np.nan
                else:
                    cohen_d = (male_mean - female_mean) / s_pooled_welch
        
        # 添加到结果列表
        results.append({
            '指标': col,
            '男性例数': male_n,
            '男性均值': male_mean,
            '男性标准差': male_std,
            '女性例数': female_n,
            '女性均值': female_mean,
            '女性标准差': female_std,
            '均值差值': male_mean - female_mean if pd.notna(male_mean) and pd.notna(female_mean) else np.nan,
            '检验方法': test_used,
            '统计量': t_stat, # Name is generic, could be t or U
            'P值': p_value,
            'Cohen_d': cohen_d if pd.notna(cohen_d) else "N/A",
            '是否显著差异': '是' if pd.notna(p_value) and p_value < 0.05 else ('否' if pd.notna(p_value) else 'N/A'),
            '男性正态P值': normality_male_p,
            '女性正态P值': normality_female_p,
            'Levene方差齐性P值': levene_p,
            '备注': "; ".join(assumptions_met_notes) if assumptions_met_notes else "前提条件基本满足"
        })
    
    # 创建DataFrame
    results_df = pd.DataFrame(results)
    print("性别差异分析完成")
    return results_df

def analyze_age_groups(df):
    """
    分析各年龄组特征
    
    参数:
        df: 数据DataFrame
    
    返回:
        包含年龄组分析结果的DataFrame
    """
    print("\n分析年龄组差异...")
    
    # 检查是否存在年龄组列
    if '年龄组' not in df.columns:
        if 'age' in df.columns:
            print("注意: 未找到'年龄组'列，尝试创建...")
            # 创建年龄组
            age_bins = [0, 30, 40, 50, 60, 70, 120]
            age_labels = ['<30', '30-39', '40-49', '50-59', '60-69', '70+']
            df['年龄组'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)
        else:
            print("警告: 数据中没有年龄相关列")
            return None
    
    # 数值列 - updated and expanded
    numeric_cols = [
        'pwv', 'age', 'sbp', 'dbp', 'bmi', 'height', 'weight',
        'cfpwv_speed', 'bapwv_right_speed', 'bapwv_left_speed',
        'cfpwv_carotid_si', 'cfpwv_carotid_ri', 'hrv_index',
        'creatinine_umol_l', 'urea_mmol_l', 'crp_mg_l', 
        'ef_percent', 'abi_right_pt_index', '脉压差',
        # New additions for this round
        'cfpwv_carotid_daix', 'cfpwv_interval_ms', 'cfpwv_distance_cm',
        'bapwv_right_distance_cm', 'bapwv_left_interval_ms',
        'abi_right_brachial_index', 'bfv_carotid_mean_speed',
        'bnp_pg_ml', 'wbc_10_9', 'hb_g_l'
    ]
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    if not available_cols:
        print("警告: 未找到可用于年龄组分析的数值列")
        return None
    
    # 计算各年龄组的统计值
    group_stats = df.groupby('年龄组')[available_cols].agg(['count', 'mean', 'std'])
    
    # 格式化多级索引
    group_stats.columns = [f'{col}_{stat}' for col, stat in group_stats.columns]
    
    # 重置索引，使年龄组成为一列
    group_stats = group_stats.reset_index()
    
    # 进行方差分析（ANOVA），比较各年龄组之间的差异
    anova_results = []
    tukey_results_dict = {} # To store tukey results if ANOVA is used

    for col in available_cols:
        # 获取各组数据
        groups_data = [df[df['年龄组'] == group_name][col].dropna() for group_name in df['年龄组'].unique() if pd.notna(group_name)]
        group_names_for_test = [str(group_name) for group_name in df['年龄组'].unique() if pd.notna(group_name) and len(df[df['年龄组'] == group_name][col].dropna()) >= 5] # Ensure group names are strings for Tukey
        groups_for_test = [g for g in groups_data if len(g) >= 5] # Minimum 5 samples per group for tests

        test_used = "N/A"
        stat_val, p_value, effect_size = np.nan, np.nan, np.nan
        normality_p_values = []
        levene_p = np.nan
        assumptions_met_notes = []

        if len(groups_for_test) < 2:  # 至少需要2个组进行比较
            assumptions_met_notes.append("有效比较组不足2个")
            test_used = "组别不足"
        else:
            # 1. 正态性检验 (Shapiro-Wilk for each group)
            all_groups_normal = True
            for i, group_data in enumerate(groups_for_test):
                if len(group_data) >= 3: # Shapiro needs at least 3 samples
                    _, shapiro_p = stats.shapiro(group_data)
                    normality_p_values.append(round(shapiro_p, 4))
                    if shapiro_p <= 0.05:
                        all_groups_normal = False
                        assumptions_met_notes.append(f"组 {group_names_for_test[i]} 非正态 (p={shapiro_p:.4f})")
                else:
                    normality_p_values.append(np.nan)
                    all_groups_normal = False # Treat as non-normal if too small for test
                    assumptions_met_notes.append(f"组 {group_names_for_test[i]} 样本过小无法验正态性")
            
            # 2. 方差齐性检验 (Levene's test)
            try:
                _, levene_p = stats.levene(*groups_for_test)
            except ValueError:
                levene_p = np.nan
                assumptions_met_notes.append("Levene检验无法执行")
            
            variances_equal = levene_p > 0.05 if pd.notna(levene_p) else False
            if pd.notna(levene_p) and not variances_equal: assumptions_met_notes.append(f"方差不齐 (Levene p={levene_p:.4f})")

            # 3. 选择并执行检验
            if all_groups_normal and variances_equal:
                test_used = "ANOVA"
                try:
                    stat_val, p_value = stats.f_oneway(*groups_for_test)
                    # Calculate Eta-squared for ANOVA effect size
                    # Sum of squares between groups (SSB)
                    grand_mean = np.concatenate(groups_for_test).mean()
                    ssb = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups_for_test)
                    # Sum of squares total (SST)
                    sst = sum((x - grand_mean)**2 for g in groups_for_test for x in g)
                    if sst == 0: effect_size = np.nan
                    else: effect_size = ssb / sst # Eta-squared

                    if p_value < 0.05:
                        # Perform Tukey's HSD post-hoc test
                        all_data_for_tukey = []
                        all_groups_for_tukey = []
                        for i, group_data in enumerate(groups_for_test):
                            all_data_for_tukey.extend(group_data)
                            all_groups_for_tukey.extend([group_names_for_test[i]] * len(group_data))
                        
                        if len(all_data_for_tukey) > 0 and len(all_groups_for_tukey) > 0:
                            tukey_result = pairwise_tukeyhsd(np.array(all_data_for_tukey), np.array(all_groups_for_tukey), alpha=0.05)
                            tukey_df = pd.DataFrame(data=tukey_result._results_table.data[1:], columns=tukey_result._results_table.data[0])
                            tukey_results_dict[col] = tukey_df
                            assumptions_met_notes.append("Tukey HSD已执行")
                        else:
                            assumptions_met_notes.append("Tukey HSD数据不足")

                except Exception as e:
                    assumptions_met_notes.append(f"ANOVA 执行错误: {e}")
                    stat_val, p_value = np.nan, np.nan
            else:
                test_used = "Kruskal-Wallis"
                try:
                    stat_val, p_value = stats.kruskal(*groups_for_test)
                    # Effect size for Kruskal-Wallis (Epsilon-squared or Freeman's Theta)
                    # Epsilon-squared = H / ( (n^2 - 1) / (n + 1) ) -> H / (n-1)
                    # For simplicity, we might just report H and p-value for now, or set effect_size to N/A
                    n_total = sum(len(g) for g in groups_for_test)
                    if n_total > 1 and len(groups_for_test) > 1: # H is stat_val
                         # Calculate Epsilon-squared for Kruskal-Wallis
                        epsilon_squared = (stat_val - (len(groups_for_test) -1)) / (n_total - len(groups_for_test))
                        # Another common one: eta_squared_H = (H - k + 1) / (n - k)
                        # H = stat_val, k = len(groups_for_test), n = n_total
                        # Let's use H / (n-1) as a simpler approximation or eta_squared_H for now.
                        # eta_H_squared = (stat_val - len(groups_for_test) + 1) / (n_total - len(groups_for_test))
                        # Or a simpler interpretation: (stat_val) / (n_total -1) if n_total > 1 else np.nan
                        # Let's use a common formula for eta-squared based on H: (H - k + 1) / (n - k)
                        # k = number of groups, n = total number of observations
                        # H = Kruskal-Wallis H statistic (stat_val here)
                        k_groups = len(groups_for_test)
                        if n_total - k_groups != 0:
                             effect_size = (stat_val - k_groups + 1) / (n_total - k_groups) 
                        else:
                             effect_size = np.nan
                    else:
                        effect_size = np.nan

                except Exception as e:
                    assumptions_met_notes.append(f"Kruskal-Wallis 执行错误: {e}")
                    stat_val, p_value = np.nan, np.nan

        anova_results.append({
            '变量': col,
            '检验方法': test_used,
            '统计量': stat_val,
            'P值': p_value,
            '效应量': effect_size if pd.notna(effect_size) else "N/A",
            '是否显著差异': '是' if pd.notna(p_value) and p_value < 0.05 else ('否' if pd.notna(p_value) else 'N/A'),
            '各组正态P值(Shapiro)': ", ".join(map(str, normality_p_values)) if normality_p_values else "N/A",
            'Levene方差齐性P值': levene_p if pd.notna(levene_p) else "N/A",
            '备注': "; ".join(assumptions_met_notes) if assumptions_met_notes else "前提条件基本满足"
        })
    
    # 创建ANOVA结果DataFrame
    anova_df = pd.DataFrame(anova_results)
    
    # 添加ANOVA结果到主结果字典
    results = {
        'group_stats': group_stats, # This is the descriptive stats per group
        'age_group_tests_summary': anova_df, # This is the summary of ANOVA/Kruskal tests
        'age_group_tukey_results': tukey_results_dict # This will contain DFs for each var if Tukey was run
    }
    
    print("年龄组分析完成")
    return results

def calculate_correlations(df):
    """
    计算变量之间的相关性
    
    参数:
        df: 数据DataFrame
    
    返回:
        包含相关性分析结果的DataFrame
    """
    print("\n计算变量相关性...")
    
    # 选择数值列进行相关性分析 - updated and expanded
    numeric_cols = [
        'pwv', 'age', 'sbp', 'dbp', 'bmi', 'height', 'weight',
        'cfpwv_speed', 'bapwv_right_speed', 'bapwv_left_speed',
        'cfpwv_carotid_si', 'cfpwv_carotid_ri', 'hrv_index',
        'creatinine_umol_l', 'urea_mmol_l', 'crp_mg_l', 
        'ef_percent', 'abi_right_pt_index', '脉压差',
        # New additions for this round
        'cfpwv_carotid_daix', 'cfpwv_interval_ms', 'cfpwv_distance_cm',
        'bapwv_right_distance_cm', 'bapwv_left_interval_ms',
        'abi_right_brachial_index', 'bfv_carotid_mean_speed',
        'bnp_pg_ml', 'wbc_10_9', 'hb_g_l'
    ]
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    if len(available_cols) < 2:
        print("警告: 数值列不足，无法进行相关性分析")
        return None
    
    # 计算相关系数矩阵
    corr_matrix = df[available_cols].corr()
    
    # 转换为长格式，便于报告
    corr_results = []
    
    for i, col1 in enumerate(available_cols):
        for j, col2 in enumerate(available_cols):
            if i < j:  # 仅使用矩阵的上三角部分，避免重复
                r = corr_matrix.loc[col1, col2]
                
                # 计算p值 (使用pearsonr)
                try:
                    _, p_value = stats.pearsonr(
                        df[col1].dropna(),
                        df[col2].dropna()
                    )
                except Exception as e:
                    print(f"计算'{col1}'和'{col2}'的p值时出错: {e}")
                    p_value = np.nan
                
                # 添加到结果列表
                corr_results.append({
                    '变量1': col1,
                    '变量2': col2,
                    '相关系数': r,
                    'P值': p_value,
                    '是否显著相关': '是' if p_value < 0.05 else '否'
                })
    
    # 创建结果DataFrame
    corr_df = pd.DataFrame(corr_results)
    
    # 按相关系数绝对值大小排序
    corr_df = corr_df.sort_values(by='相关系数', key=abs, ascending=False)
    
    print("相关性分析完成")
    return corr_df

def perform_pwv_regression(df):
    """
    执行PWV多元回归分析
    
    参数:
        df: 数据DataFrame
    
    返回:
        包含回归分析结果的DataFrame
    """
    print("\n执行PWV回归分析...")
    
    # 检查是否存在PWV列
    if 'pwv' not in df.columns:
        print("警告: 数据中没有pwv列")
        return None
    
    # 确保关键列存在
    required_cols = ['pwv', 'age', 'sbp', 'dbp', 'bmi', 'gender']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        print(f"警告: 回归分析所需列缺失: {missing}，跳过PWV回归分析")
        return pd.DataFrame()

    # 构建基本回归模型
    try:
        model_formula = 'pwv ~ age + sbp + dbp + bmi + C(gender)'
        # Add more predictors based on availability and relevance
        extended_predictors = [
            'hrv_index', 'crp_mg_l', 'ef_percent', 'abi_right_pt_index', 
            'creatinine_umol_l', 'urea_mmol_l',
            # New additions for this round
            'cfpwv_carotid_daix', 'cfpwv_interval_ms', # Note: distance might be collinear with interval and speed
            'bnp_pg_ml', 'wbc_10_9', 'hb_g_l', 'abi_right_brachial_index', 'bfv_carotid_mean_speed'
            # 'cfpwv_distance_cm', 'bapwv_right_distance_cm', 'bapwv_left_interval_ms' - these might be less direct predictors of PWV itself
            # or could cause multicollinearity if speed is already in a model implicitly or explicitly.
        ]
        for predictor in extended_predictors:
            if predictor in df.columns:
                # Check if predictor is numeric and has variance
                if pd.api.types.is_numeric_dtype(df[predictor]) and df[predictor].nunique() > 1:
                     # Handle potential NaN values in predictor by ensuring they don't break the formula string
                    if df[predictor].isnull().sum() < len(df) * 0.5: # Example: allow if less than 50% NaN
                         model_formula += f' + {predictor}'
                else:
                    print(f"Skipping predictor {predictor} due to non-numeric type or no variance.")

        model = ols(model_formula, data=df).fit()
        results_summary = model.summary()
        print("PWV回归分析完成 (线性模型):")
        
        # 提取回归系数和显著性
        results = []
        conf_int_df = model.conf_int() # Get confidence intervals
        conf_int_df.columns = ['CI Lower', 'CI Upper']

        for variable in model.params.index:
            results.append({
                '变量': variable,
                '系数': model.params[variable],
                '标准误': model.bse[variable],
                't值': model.tvalues[variable],
                'P值': model.pvalues[variable],
                'CI Lower': conf_int_df.loc[variable, 'CI Lower'] if variable in conf_int_df.index else np.nan,
                'CI Upper': conf_int_df.loc[variable, 'CI Upper'] if variable in conf_int_df.index else np.nan,
                '是否显著': '是' if model.pvalues[variable] < 0.05 else '否'
            })
        
        # 添加模型评估指标
        r_squared = model.rsquared
        adj_r_squared = model.rsquared_adj
        f_pvalue = model.f_pvalue
        
        # 创建结果DataFrame
        results_df = pd.DataFrame(results)
        
        # 添加模型总体评价
        model_stats = pd.DataFrame([{
            '模型': 'PWV多元回归',
            'R²': r_squared,
            '调整R²': adj_r_squared,
            '样本量': len(df),
            'F检验P值': f_pvalue,
            '方程': f"PWV = {model.params['Intercept']:.4f} + " + 
                  " + ".join([f"{model.params[v]:.4f}*{v}" for v in model.params.index if v != 'Intercept' and v not in ['C(gender)[T.1.0]', 'C(gender)[T.1]'] ]) # Dynamically build equation string
        }])
        
        regression_results = {
            'coefficients': results_df,
            'model_stats': model_stats,
            'model': model
        }
        
        print("PWV回归分析完成")
        return regression_results
    
    except Exception as e:
        print(f"执行回归分析时出错: {e}")
        return None

def analyze_categorical_associations(df, analysis_results_dict):
    """
    使用卡方检验分析分类变量之间的关联性。

    参数:
        df (pd.DataFrame): 包含数据的DataFrame。
        analysis_results_dict (dict): 用于存储分析结果的字典。

    返回:
        pd.DataFrame: 包含卡方检验结果的DataFrame。
    """
    print("\n分析分类变量间的关联性 (卡方检验)...")
    
    # 定义要分析的分类变量对
    # 格式: (col1, col2, description)
    # 确保这些列存在于 create_derived_features 中
    categorical_pairs = [
        ('gender', 'bp_category', '性别与血压分级'),
        ('gender', 'pwv_category', '性别与PWV分级'),
        ('gender', 'abi_risk_category', '性别与ABI风险分级'),
        ('bp_category', 'pwv_category', '血压分级与PWV分级'),
        ('bp_category', 'abi_risk_category', '血压分级与ABI风险分级'),
        ('pwv_category', 'abi_risk_category', 'PWV分级与ABI风险分级'),
        ('age_group', 'bp_category', '年龄组与血压分级'),
        ('age_group', 'pwv_category', '年龄组与PWV分级'),
        ('age_group', 'abi_risk_category', '年龄组与ABI风险分级'),
        ('risk_level', 'bp_category', '综合风险等级与血压分级'),
        ('risk_level', 'pwv_category', '综合风险等级与PWV分级'),
    ]

    results = []

    for col1_name, col2_name, description in categorical_pairs:
        if col1_name in df.columns and col2_name in df.columns:
            # 创建列联表
            contingency_table = pd.crosstab(df[col1_name], df[col2_name])
            
            # 卡方检验要求每个单元格的期望频数至少为5，或至少80%的单元格期望频数>=5
            # 此处简化，仅检查表格大小
            if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
                print(f"  跳过: '{description}' ({col1_name} vs {col2_name}) - 其中一个或两个变量的类别不足2个。")
                results.append({
                    '比较对': description,
                    '变量1': col1_name,
                    '变量2': col2_name,
                    '卡方统计量': np.nan,
                    'P值': np.nan,
                    '自由度': np.nan,
                    '备注': '类别不足，无法检验'
                })
                continue

            if contingency_table.empty or contingency_table.sum().sum() < 10: # 样本量过小
                print(f"  跳过: '{description}' ({col1_name} vs {col2_name}) - 样本量过小。")
                results.append({
                    '比较对': description,
                    '变量1': col1_name,
                    '变量2': col2_name,
                    '卡方统计量': np.nan,
                    'P值': np.nan,
                    '自由度': np.nan,
                    '备注': '样本量过小'
                })
                continue

            print(f"\n  检验: {description} ({col1_name} vs {col2_name})")
            print("  列联表:")
            print(contingency_table)

            try:
                chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
                significance = "是" if p < 0.05 else "否"
                
                # 检查期望频数
                small_expected_freq = (expected < 5).sum()
                total_cells = expected.size
                percent_small_expected = (small_expected_freq / total_cells) * 100
                
                remark = ""
                if percent_small_expected > 20: # 如果超过20%的单元格期望频数小于5
                    remark = f"警告: {percent_small_expected:.1f}% 的单元格期望频数 < 5。卡方检验结果可能不可靠。"
                    print(f"    {remark}")
                
                print(f"    卡方统计量: {chi2:.4f}, P值: {p:.4f}, 自由度: {dof}")
                print(f"    关联性是否显著 (P < 0.05): {significance}")

                results.append({
                    '比较对': description,
                    '变量1': col1_name,
                    '变量2': col2_name,
                    '卡方统计量': chi2,
                    'P值': p,
                    '自由度': dof,
                    '是否显著': significance,
                    '备注': remark
                })

            except ValueError as ve: # 通常是由于列联表不符合要求 (例如，全零行/列)
                print(f"    执行卡方检验时出错 for {description}: {ve}")
                results.append({
                    '比较对': description,
                    '变量1': col1_name,
                    '变量2': col2_name,
                    '卡方统计量': np.nan,
                    'P值': np.nan,
                    '自由度': np.nan,
                    '备注': f'卡方检验ValueError: {ve}'
                })
            except Exception as e:
                print(f"    执行卡方检验时发生未知错误 for {description}: {e}")
                results.append({
                    '比较对': description,
                    '变量1': col1_name,
                    '变量2': col2_name,
                    '卡方统计量': np.nan,
                    'P值': np.nan,
                    '自由度': np.nan,
                    '备注': f'卡方检验未知错误: {e}'
                })
        else:
            print(f"  跳过: '{description}' ({col1_name} vs {col2_name}) - 一个或两个变量列不存在。")
            results.append({
                '比较对': description,
                '变量1': col1_name,
                '变量2': col2_name,
                '卡方统计量': np.nan,
                'P值': np.nan,
                '自由度': np.nan,
                '备注': '列不存在'
            })

    results_df = pd.DataFrame(results)
    analysis_results_dict['categorical_associations'] = results_df
    print("\n分类变量关联性分析完成。")
    return results_df

def analyze_paired_measurements(df, analysis_results_dict):
    """
    分析成对测量数据 (例如，左侧 vs 右侧)

    参数:
        df: 数据DataFrame
        analysis_results_dict: 包含所有分析结果的字典

    返回:
        包含成对t检验结果的DataFrame或None
    """
    print("\n分析成对测量数据...")
    paired_results = []
    
    # 定义要比较的成对列
    # 格式: (col1, col2, description_prefix)
    measurement_pairs = [
        ('bapwv_left_speed', 'bapwv_right_speed', 'baPWV速度'),
        # 未来可以添加更多成对测量，例如：
        # ('bapwv_left_distance_cm', 'bapwv_right_distance_cm', 'baPWV距离'),
        # ('bapwv_left_interval_ms', 'bapwv_right_interval_ms', 'baPWV时间间隔'),
        # ('abi_left_pt_index', 'abi_right_pt_index', '胫后ABI') # 假设这些列已标准化
    ]

    for col1_name, col2_name, desc_prefix in measurement_pairs:
        if col1_name in df.columns and col2_name in df.columns:
            # 提取成对数据并移除任何一对中存在NaN的行
            paired_data = df[[col1_name, col2_name]].dropna()
            
            if len(paired_data) < 5: # 需要足够的样本量进行检验
                print(f"警告: {desc_prefix} 的成对数据样本量不足 ({len(paired_data)}对)，跳过检验")
                continue

            val1 = paired_data[col1_name]
            val2 = paired_data[col2_name]

            try:
                t_stat, p_value = stats.ttest_rel(val1, val2)
                paired_results.append({
                    '测量指标': desc_prefix,
                    '侧别1': col1_name,
                    '均值1': val1.mean(),
                    '标准差1': val1.std(),
                    '侧别2': col2_name,
                    '均值2': val2.mean(),
                    '标准差2': val2.std(),
                    '样本数(对)': len(paired_data),
                    '均值差值 (1-2)': (val1 - val2).mean(),
                    't统计量': t_stat,
                    'P值': p_value,
                    '是否显著差异': '是' if p_value < 0.05 else '否'
                })
            except Exception as e:
                print(f"执行 {desc_prefix} ({col1_name} vs {col2_name}) 的成对t检验时出错: {e}")
        else:
            print(f"警告: {desc_prefix} 的成对列缺失 ({col1_name} 或 {col2_name})，跳过检验")

    if not paired_results:
        print("未找到可进行成对分析的数据或样本量不足。")
        return None
    
    results_df = pd.DataFrame(paired_results)
    analysis_results_dict['paired_measurements'] = results_df  # Store in dict
    print("成对测量数据分析完成")
    return results_df

def analyze_abi_risk_groups(df, analysis_results_dict):
    """
    分析各ABI风险组的特征

    参数:
        df: 数据DataFrame
        analysis_results_dict: 包含所有分析结果的字典

    返回:
        包含ABI风险组分析结果的字典 (group_stats, anova_results)
    """
    print("\n分析ABI风险组差异...")

    if 'abi_risk_category' not in df.columns:
        print("警告: 数据中没有 'abi_risk_category' 列，无法进行ABI风险组分析。")
        return None

    # 使用与之前分析一致的数值列列表
    numeric_cols = [
        'pwv', 'age', 'sbp', 'dbp', 'bmi', 'height', 'weight',
        'cfpwv_speed', 'bapwv_right_speed', 'bapwv_left_speed',
        'cfpwv_carotid_si', 'cfpwv_carotid_ri', 'hrv_index',
        'creatinine_umol_l', 'urea_mmol_l', 'crp_mg_l', 
        'ef_percent', 'abi_right_pt_index', '脉压差',
        'cfpwv_carotid_daix', 'cfpwv_interval_ms', 'cfpwv_distance_cm',
        'bapwv_right_distance_cm', 'bapwv_left_interval_ms',
        'abi_right_brachial_index', 'bfv_carotid_mean_speed',
        'bnp_pg_ml', 'wbc_10_9', 'hb_g_l'
    ]
    available_cols = [col for col in numeric_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

    if not available_cols:
        print("警告: 未找到可用于ABI风险组分析的有效数值列。")
        return None

    # 计算各ABI风险组的统计值
    # Ensure abi_risk_category is treated as categorical for grouping and ordering if necessary
    df['abi_risk_category'] = pd.Categorical(df['abi_risk_category'], 
                                             categories=['正常/临界', '轻度PAD', '中度PAD', '重度PAD', '血管不可压缩'], 
                                             ordered=True)
    group_stats = df.groupby('abi_risk_category', observed=False)[available_cols].agg(['count', 'mean', 'std'])
    group_stats.columns = [f'{col}_{stat}' for col, stat in group_stats.columns]
    group_stats = group_stats.reset_index()

    # ANOVA比较各ABI风险组之间的差异
    anova_results = []
    for col in available_cols:
        groups = [df[df['abi_risk_category'] == group][col].dropna() for group in df['abi_risk_category'].unique() if pd.notna(group)]
        groups = [g for g in groups if len(g) >= 2] # Ensure groups have enough data for ANOVA

        if len(groups) >= 2:
            try:
                f_stat, p_value = stats.f_oneway(*groups)
                anova_results.append({
                    '变量': col,
                    'F统计量': f_stat,
                    'P值': p_value,
                    '是否显著差异': '是' if p_value < 0.05 else '否'
                })
            except Exception as e:
                print(f"执行 '{col}' 的ANOVA (ABI风险组) 时出错: {e}")
        else:
            print(f"变量 '{col}' 在ABI风险组中不足两个有效分组进行ANOVA分析。")

    anova_df = pd.DataFrame(anova_results)
    results = {
        'group_stats': group_stats,
        'anova_results': anova_df
    }
    analysis_results_dict['abi_risk_group_analysis'] = results
    print("ABI风险组分析完成。")
    return results

def detect_outliers(df, method='iqr', columns=None, threshold=1.5):
    """
    检测异常值
    
    参数:
        df: 数据DataFrame
        method: 检测方法，'iqr'（四分位距法）或'zscore'（z分数法）
        columns: 需要检测的列，默认为None（检测所有数值列）
        threshold: 阈值，IQR法的默认值为1.5，Z分数法的默认值为3.0
    
    返回:
        包含异常值检测结果的字典
    """
    print("\n检测异常值...")
    
    if df is None or df.empty:
        print("数据为空，无法检测异常值")
        return {'summary': pd.DataFrame(), 'outliers': {}}

    if columns is None:
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) == 0:
            print("未找到数值类型的列进行异常值检测")
            return {'summary': pd.DataFrame(), 'outliers': {}}
        columns = numeric_cols
    else:
        columns = [col for col in columns if col in df.columns and 
                  pd.api.types.is_numeric_dtype(df[col])]  # 使用pandas API判断数值类型
    
    # 初始化结果
    outliers_dict = {}
    outlier_counts = []
    z_threshold = 3.0  # Z分数法的默认阈值
    
    if len(columns) == 0:  # 使用len()明确判断列表是否为空
        print("警告: 未找到可用于异常值检测的有效数值列")
        return {'summary': pd.DataFrame(columns=['指标', '异常值数量', '总数据量', '异常值比例(%)', '检测方法']), 'outliers': outliers_dict}

    # 使用有效的列进行异常值检测
    for col in columns: # Now col is guaranteed to be a name of a Series type column
        col_data = df[col].dropna()
        
        current_col_outlier_values = pd.Series(dtype=col_data.dtype) 

        if col_data.empty:
            outlier_counts.append({
                '指标': col, '异常值数量': 0, '总数据量': 0,
                '异常值比例(%)': 0, '检测方法': f"{method.upper()} (阈值={threshold if method == 'iqr' else z_threshold}) - 无有效数据"
            })
            outliers_dict[col] = current_col_outlier_values
            continue

        if method == 'iqr':
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR == 0:
                current_col_outlier_values = pd.Series(dtype=col_data.dtype) if col_data.nunique() <= 1 else col_data[~col_data.isin([Q1])] 
            else: # IQR != 0
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                # 明确使用element-wise的方式创建布尔掩码
                is_lower_outlier = col_data < lower_bound
                is_upper_outlier = col_data > upper_bound
                
                # 使用pandas的element-wise或运算符
                outlier_mask = is_lower_outlier | is_upper_outlier
                
                # 对Series进行索引操作，确保outlier_mask是Series类型
                try:
                    # 如果outlier_mask是Series，直接用于索引
                    if isinstance(outlier_mask, pd.Series):
                        current_col_outlier_values = col_data.loc[outlier_mask]
                    else:
                        # 如果不是Series，创建一个临时的空Series
                        print(f"警告: '{col}'的outlier_mask不是Series类型")
                        current_col_outlier_values = pd.Series(dtype=col_data.dtype)
                except Exception as e:
                    print(f"索引异常值出错: {e}")
                    current_col_outlier_values = pd.Series(dtype=col_data.dtype)
            
            outliers_dict[col] = current_col_outlier_values

        elif method == 'zscore':
            if len(col_data) < 2 or col_data.std() == 0: # Check for std == 0
                current_col_outlier_values = pd.Series(dtype=col_data.dtype)
            else:
                z_scores = np.abs(stats.zscore(col_data))
                # Filter out NaNs from z_scores if any (though col_data is dropna'd)
                valid_z_scores = z_scores[~np.isnan(z_scores)]
                if len(valid_z_scores) > 0: # Check if z_scores could be computed
                     # Align z_scores with col_data's index if z_scores is a numpy array
                    z_series = pd.Series(z_scores, index=col_data.index)
                    current_col_outlier_values = col_data[z_series > z_threshold]
                else: # Should not happen if std !=0 and len > 1
                    current_col_outlier_values = pd.Series(dtype=col_data.dtype)
            
            outliers_dict[col] = current_col_outlier_values
        else:
            print(f"不支持的方法: {method}")
            outliers_dict[col] = pd.Series(dtype=col_data.dtype) # Empty series for unsupported method
            # Skip appending to outlier_counts for unsupported methods or handle as 0 outliers
            outlier_counts.append({
                '指标': col, '异常值数量': 0, '总数据量': len(col_data),
                '异常值比例(%)': 0, '检测方法': f"不支持的方法: {method}"
            })
            continue
        
        # Common append to outlier_counts
        final_outliers_for_col = outliers_dict.get(col, pd.Series(dtype=col_data.dtype))
        outlier_counts.append({
            '指标': col,
            '异常值数量': len(final_outliers_for_col),
            '总数据量': len(col_data),
            '异常值比例(%)': (len(final_outliers_for_col) / len(col_data) * 100) if len(col_data) > 0 else 0,
            '检测方法': f"{method.upper()} (阈值={threshold if method == 'iqr' else z_threshold})"
        })
    
    summary_df = pd.DataFrame(outlier_counts)
    # Ensure unique rows in summary, in case a column was processed in a strange way (though loop is per col)
    summary_df = summary_df.drop_duplicates(subset=['指标', '检测方法'], keep='last')

    results = {
        'summary': summary_df,
        'outliers': outliers_dict 
    }
    
    print("异常值检测完成")
    return results

def analyze_pwv_data(df):
    """
    分析PWV数据的主函数
    
    参数:
        df: 预处理后的数据DataFrame
    
    返回:
        包含所有分析结果的字典
    """
    print("\n======== 开始PWV数据分析 ========")
    
    # 检查输入数据
    if df is None or df.empty:
        print("错误: 无有效数据可分析")
        return None
    
    # 初始化结果字典
    results = {}
    
    # 1. 基本统计分析
    basic_stats = calculate_basic_statistics(df)
    if basic_stats is not None:
        results['basic_stats'] = basic_stats
    
    # 2. 性别差异分析
    gender_comparison = analyze_gender_differences(df)
    if gender_comparison is not None:
        results['gender_comparison'] = gender_comparison
    
    # 3. 年龄组分析
    age_group_results = analyze_age_groups(df)
    if age_group_results is not None:
        results['age_group_stats'] = age_group_results['group_stats']
        results['age_group_tests_summary'] = age_group_results['age_group_tests_summary']
        if 'age_group_tukey_results' in age_group_results and age_group_results['age_group_tukey_results']:
            # Tukey results are a dict of DataFrames. Need to save them appropriately.
            # For now, let's ensure the dict is passed. Saving logic will handle it.
            results['age_group_tukey_results'] = age_group_results['age_group_tukey_results']
    
    # 4. 相关性分析
    correlation_results = calculate_correlations(df)
    if correlation_results is not None:
        results['correlation_analysis'] = correlation_results
        results['age_group_tukey_results'] = age_group_results['age_group_tukey_results']
    
    # 5. PWV回归分析
    pwv_regression = perform_pwv_regression(df)
    if pwv_regression is not None:
        results['pwv_regression'] = pwv_regression['coefficients']
        results['pwv_model_stats'] = pwv_regression['model_stats']
        results['pwv_model'] = pwv_regression['model']  # 保存模型对象，以便后续使用
    
    # 6. 异常值检测
    outliers_results = detect_outliers(df, method='iqr')
    if outliers_results is not None:
        results['outliers'] = outliers_results
    
    # 7. 成对测量分析
    paired_analysis_results = analyze_paired_measurements(df, results)
    if paired_analysis_results is not None:
        results['paired_analysis'] = paired_analysis_results

    # 8. ABI风险组分析
    abi_risk_group_analysis_results = analyze_abi_risk_groups(df, results)
    if abi_risk_group_analysis_results is not None:
        results['abi_risk_group_stats'] = abi_risk_group_analysis_results['group_stats']
        results['abi_risk_group_anova'] = abi_risk_group_analysis_results['anova_results']

    # 9. 分类变量关联性分析
    categorical_association_results = analyze_categorical_associations(df, results)
    if categorical_association_results is not None:
        results['categorical_associations'] = categorical_association_results

    print("\n======== PWV数据分析完成 ========")
    
    return results

def save_analysis_results(results, output_path="output/tables/PWV_Analysis_Results.xlsx"):
    """
    保存分析结果到Excel文件
    
    参数:
        results: 分析结果字典
        output_path: 输出文件路径
    
    返回:
        保存是否成功的布尔值
    """
    print(f"\n保存分析结果到 {output_path}...")
    
    if not results:
        print("错误: 没有结果可保存")
        return False
    
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 创建ExcelWriter对象
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 保存各个分析结果到不同的工作表
            for key, value in results.items():
                # 跳过不是DataFrame的值或模型对象
                if key.endswith('_model'):
                    continue
                
                if isinstance(value, pd.DataFrame):
                    # 写入工作表
                    sheet_name = key[:31]  # Excel工作表名称长度限制
                    value.to_excel(writer, sheet_name=sheet_name, index=False)
                    print(f"  已保存 '{key}' 到工作表 '{sheet_name}'")
                elif isinstance(value, dict) and key == 'age_group_tukey_results':
                    # Handle dict of DataFrames (Tukey results)
                    for var_name, tukey_df in value.items():
                        if isinstance(tukey_df, pd.DataFrame):
                            sheet_name = f"Tukey_{var_name}"[:31]
                            tukey_df.to_excel(writer, sheet_name=sheet_name, index=False)
                            print(f"  已保存 Tukey结果 for '{var_name}' 到工作表 '{sheet_name}'")
        
        print(f"✅ 分析结果已保存到: {output_path}")
        return True
    
    except Exception as e:
        print(f"❌ 保存分析结果时出错: {e}")
        return False

if __name__ == "__main__":
    # 如果作为独立脚本运行，从data_processing导入load_and_prepare_data
    from data_processing import load_and_prepare_data
    
    # 加载和预处理数据
    df = load_and_prepare_data()
    
    if df is not None:
        # 分析数据
        analysis_results = analyze_pwv_data(df)
        
        if analysis_results:
            # 保存分析结果
            output_path = os.path.join("output", "tables", f"PWV_Analysis_Results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
            save_analysis_results(analysis_results, output_path) 