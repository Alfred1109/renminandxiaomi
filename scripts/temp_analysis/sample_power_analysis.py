#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
样本量与统计功效分析整合脚本
基于真实效应量计算不同统计功效(80%和90%)下的最小样本量要求
并分析当前数据样本是否满足统计学检验的要求
可视化样本量需求并导出详细结果到Excel文件
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.stats.power as smp
import statsmodels.stats.api as sms
from matplotlib.font_manager import FontProperties
import warnings
warnings.filterwarnings('ignore')

# 确保能访问项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入项目的数据处理模块
try:
    from scripts.data_processing import load_and_prepare_data
    from scripts.fix_chinese_in_matplotlib import setup, apply_to_figure
    chinese_font = setup()
except ImportError as e:
    print(f"导入项目模块时出错: {e}")
    print("尝试使用替代方法...")
    # 修复字体设置
    chinese_font = FontProperties()
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

# 加载数据
def load_data():
    """加载并准备PWV分析数据"""
    try:
        df = load_and_prepare_data()
        print(f"成功加载数据：{df.shape[0]}行 x {df.shape[1]}列")
        return df
    except Exception as e:
        print(f"加载数据时出错: {e}")
        print("尝试直接加载数据文件...")
        try:
            # 尝试直接加载Excel文件
            data_paths = [
                os.path.join(project_root, "docs", "excel", "pwv数据采集表 -去公式.xlsx"),
                os.path.join(project_root, "docs", "excel", "原始数据", "pwv数据采集表 -去公式.xlsx"),
                "docs/excel/pwv数据采集表 -去公式.xlsx"
            ]
            
            for path in data_paths:
                if os.path.exists(path):
                    df = pd.read_excel(path)
                    print(f"成功从 {path} 加载数据：{df.shape[0]}行 x {df.shape[1]}列")
                    
                    # 按照原始数据结构重命名必要的列
                    rename_dict = {
                        '基础信息-年龄': 'age',
                        '受试者-性别': 'gender',
                        '竞品信息-脉搏波传导速度': 'pwv',
                        'cfPWV-速度m/s': 'cfpwv_速度',
                        'baPWV-右侧-速度m/s': 'bapwv_右侧_速度',
                        'baPWV-左侧-速度m/s': 'bapwv_左侧_速度'
                    }
                    
                    # 只尝试重命名存在的列
                    rename_cols = {old: new for old, new in rename_dict.items() if old in df.columns}
                    if rename_cols:
                        df = df.rename(columns=rename_cols)
                        print(f"已重命名列: {rename_cols}")
                        
                    # 将关键列转换为数值类型
                    numeric_cols = ['age', 'pwv', 'cfpwv_速度', 'bapwv_右侧_速度', 'bapwv_左侧_速度', '收缩压', '舒张压']
                    for col in numeric_cols:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # 创建年龄组分类
                    if 'age' in df.columns:
                        df['age_group'] = pd.cut(df['age'], bins=[0, 40, 60, 100], labels=['<40', '40-60', '>60'])
                    
                    return df
            
            print("未找到数据文件，使用模拟数据...")
            # 创建模拟数据
            n = 221  # 与实际数据相近的样本量
            mock_data = {
                'age': np.random.normal(58, 10, n),
                'pwv': np.random.normal(9.5, 1.8, n),
                '收缩压': np.random.normal(130, 15, n),
                '舒张压': np.random.normal(80, 10, n),
                'cfpwv_速度': np.random.normal(9.25, 1.5, n),
                'bapwv_右侧_速度': np.random.normal(11.1, 2.0, n),
                'bapwv_左侧_速度': np.random.normal(11.3, 2.0, n),
                'gender': np.random.choice([0, 1], n)
            }
            df = pd.DataFrame(mock_data)
            
            # 创建年龄组分类
            df['age_group'] = pd.cut(df['age'], bins=[0, 40, 60, 100], labels=['<40', '40-60', '>60'])
            
            return df
        except Exception as inner_e:
            print(f"直接加载数据文件也失败: {inner_e}")
            return None
    return None

def missing_data_analysis(df):
    """分析数据缺失情况"""
    if df is None:
        print("无数据可分析")
        return None
    
    # 计算每列缺失值比例
    missing_data = df.isnull().sum().sort_values(ascending=False)
    missing_percent = (df.isnull().sum() / df.isnull().count() * 100).sort_values(ascending=False)
    missing_analysis = pd.concat([missing_data, missing_percent], axis=1, keys=['缺失数量', '缺失比例(%)'])
    
    # 只展示有缺失值的列
    missing_analysis = missing_analysis[missing_analysis['缺失数量'] > 0]
    
    if not missing_analysis.empty:
        print("\n数据缺失情况分析:")
        print(missing_analysis)
    else:
        print("数据完整，无缺失值")
    
    return missing_analysis

def calculate_effect_sizes(df):
    """
    计算各变量的真实效应量
    
    参数:
        df: 数据DataFrame
        
    返回:
        包含各变量效应量的字典
    """
    if df is None:
        print("无数据可计算效应量")
        return None
    
    print("\n计算各变量的真实效应量...")
    
    # 要分析的主要变量（连续型）
    continuous_vars = ['age', 'pwv', '收缩压', '舒张压', 'cfpwv_速度', 'bapwv_右侧_速度', 'bapwv_左侧_速度']
    
    # 确保所有变量都在DataFrame中
    available_vars = [v for v in continuous_vars if v in df.columns]
    if len(available_vars) < len(continuous_vars):
        missing_vars = set(continuous_vars) - set(available_vars)
        print(f"警告: 以下变量未在数据集中找到: {missing_vars}")
    
    # 计算效应量
    effect_sizes = {}
    
    print("\n各指标效应量计算结果:")
    print("-" * 70)
    print(f"{'变量名':<20} {'样本数':<10} {'均值':<10} {'标准差':<10} {'标准化效应量':<15} {'真实效应量'}")
    print("-" * 70)
    
    for var in available_vars:
        # 获取非缺失数据
        data = df[var].dropna()
        n = len(data)
        
        if n > 5:
            mean = data.mean()
            std = data.std()
            
            # 标准化效应量 (Cohen's d) - 假设中等效应量0.3
            std_effect_size = 0.3
            
            # 真实效应量 - 基于数据的实际均值和标准差
            # 假设检验的差异为5%的均值变化
            real_effect_size = 0.05 * abs(mean) / std if std > 0 else 0.3
            
            effect_sizes[var] = {
                'n': n,
                'mean': mean,
                'std': std,
                'std_effect': std_effect_size,
                'real_effect': real_effect_size
            }
            
            print(f"{var:<20} {n:<10d} {mean:<10.2f} {std:<10.2f} {std_effect_size:<15.3f} {real_effect_size:.4f}")
    
    # 计算组间比较的效应量
    # 1. 性别间PWV差异
    if 'gender' in df.columns and 'pwv' in df.columns:
        male_pwv = df[df['gender'] == 1]['pwv'].dropna()
        female_pwv = df[df['gender'] == 0]['pwv'].dropna()
        
        if len(male_pwv) > 5 and len(female_pwv) > 5:
            # 计算Cohen's d效应量
            gender_effect = np.abs(male_pwv.mean() - female_pwv.mean()) / np.sqrt(
                (male_pwv.var() + female_pwv.var()) / 2
            )
            
            effect_sizes['gender_pwv_diff'] = {
                'n': len(male_pwv) + len(female_pwv),
                'group1_mean': male_pwv.mean(),
                'group2_mean': female_pwv.mean(),
                'group1_std': male_pwv.std(),
                'group2_std': female_pwv.std(),
                'effect': gender_effect
            }
            
            print(f"{'性别PWV差异':<20} {len(male_pwv) + len(female_pwv):<10d} {'':<10} {'':<10} {'':<15} {gender_effect:.4f}")
    
    # 2. 年龄组PWV差异
    if 'age_group' in df.columns and 'pwv' in df.columns:
        age_groups_pwv = df.groupby('age_group')['pwv'].apply(lambda x: x.dropna().values)
        k_groups = len(age_groups_pwv)
        
        if k_groups >= 2:
            # 计算ANOVA的效应量f
            group_means = [np.mean(g) for g in age_groups_pwv if len(g) > 0]
            grand_mean = np.mean([m for m in group_means])
            group_sizes = [len(g) for g in age_groups_pwv if len(g) > 0]
            
            # 组间平方和
            ss_between = sum([n * (m - grand_mean)**2 for n, m in zip(group_sizes, group_means)])
            
            # 组内平方和
            ss_within = sum([sum((x - m)**2) for x, m in zip(age_groups_pwv, group_means) if len(x) > 0])
            
            # 计算效应量f
            if ss_within > 0:
                age_effect = np.sqrt(ss_between / ss_within)
            else:
                age_effect = 0.25  # 默认中等效应量
            
            effect_sizes['age_group_pwv_diff'] = {
                'n': sum(group_sizes),
                'group_means': group_means,
                'group_sizes': group_sizes,
                'effect': age_effect
            }
            
            print(f"{'年龄组PWV差异':<20} {sum(group_sizes):<10d} {'':<10} {'':<10} {'':<15} {age_effect:.4f}")
    
    print("-" * 70)
    return effect_sizes

def calculate_required_sample_size(effect_sizes, powers=[0.8, 0.9], alpha=0.05):
    """
    基于效应量计算不同统计功效下的最小样本量要求
    
    参数:
        effect_sizes: 效应量字典
        powers: 统计功效列表，默认为[0.8, 0.9]
        alpha: 显著性水平，默认0.05
        
    返回:
        包含各变量在不同功效下所需样本量的字典
    """
    if effect_sizes is None:
        print("无效应量数据可计算样本量")
        return None
    
    sample_sizes = {}
    
    for power in powers:
        power_key = f"power_{int(power*100)}"
        print(f"\n计算在统计功效{power*100:.0f}%、显著性水平{alpha}下的最小样本量要求...")
        
        print("\n各指标实际样本量与最小样本量需求:")
        print("-" * 100)
        print(f"{'变量名':<20} {'实际样本量':<15} {'效应量类型':<15} {'效应量值':<15} {'最小所需样本量':<15} {'是否满足要求'}")
        print("-" * 100)
        
        for var, effect_data in effect_sizes.items():
            if var not in ['gender_pwv_diff', 'age_group_pwv_diff']:
                # 单变量样本量计算
                # 使用标准化效应量
                analysis_std = smp.TTestIndPower()
                required_n_std = analysis_std.solve_power(
                    effect_size=effect_data['std_effect'], 
                    power=power, 
                    alpha=alpha, 
                    ratio=1.0
                )
                required_n_std = int(np.ceil(required_n_std))
                
                # 使用真实效应量
                analysis_real = smp.TTestIndPower()
                required_n_real = analysis_real.solve_power(
                    effect_size=effect_data['real_effect'], 
                    power=power, 
                    alpha=alpha, 
                    ratio=1.0
                )
                required_n_real = int(np.ceil(required_n_real))
                
                if power_key not in sample_sizes:
                    sample_sizes[power_key] = {}
                
                sample_sizes[power_key][var] = {
                    'std_effect': {
                        'effect': effect_data['std_effect'],
                        'required': required_n_std,
                        'actual': effect_data['n'],
                        'sufficient': effect_data['n'] >= required_n_std
                    },
                    'real_effect': {
                        'effect': effect_data['real_effect'],
                        'required': required_n_real,
                        'actual': effect_data['n'],
                        'sufficient': effect_data['n'] >= required_n_real
                    }
                }
                
                # 输出结果
                status_std = "✓" if effect_data['n'] >= required_n_std else "✗"
                status_real = "✓" if effect_data['n'] >= required_n_real else "✗"
                
                print(f"{var:<20} {effect_data['n']:<15d} {'标准化':<15} {effect_data['std_effect']:<15.3f} {required_n_std:<15d} {status_std}")
                print(f"{'':<20} {'':<15} {'真实':<15} {effect_data['real_effect']:<15.4f} {required_n_real:<15d} {status_real}")
            
            elif var == 'gender_pwv_diff':
                # 性别组间比较样本量
                analysis = smp.TTestIndPower()
                required_n = analysis.solve_power(
                    effect_size=effect_data['effect'], 
                    power=power, 
                    alpha=alpha, 
                    ratio=1.0
                )
                required_n = int(np.ceil(required_n * 2))  # 两组总样本量
                
                if power_key not in sample_sizes:
                    sample_sizes[power_key] = {}
                
                sample_sizes[power_key][var] = {
                    'effect': effect_data['effect'],
                    'required': required_n,
                    'actual': effect_data['n'],
                    'sufficient': effect_data['n'] >= required_n
                }
                
                status = "✓" if effect_data['n'] >= required_n else "✗"
                print(f"{'性别PWV差异':<20} {effect_data['n']:<15d} {'组间差异':<15} {effect_data['effect']:<15.4f} {required_n:<15d} {status}")
            
            elif var == 'age_group_pwv_diff':
                # 年龄组ANOVA样本量
                k_groups = len(effect_data['group_means'])
                analysis = smp.FTestAnovaPower()
                required_n = analysis.solve_power(
                    effect_size=effect_data['effect'],
                    power=power,
                    alpha=alpha,
                    k_groups=k_groups
                )
                required_n = int(np.ceil(required_n * k_groups))  # 所有组总样本量
                
                if power_key not in sample_sizes:
                    sample_sizes[power_key] = {}
                
                sample_sizes[power_key][var] = {
                    'effect': effect_data['effect'],
                    'required': required_n,
                    'actual': effect_data['n'],
                    'sufficient': effect_data['n'] >= required_n,
                    'k_groups': k_groups
                }
                
                status = "✓" if effect_data['n'] >= required_n else "✗"
                print(f"{'年龄组PWV差异':<20} {effect_data['n']:<15d} {'组间差异':<15} {effect_data['effect']:<15.4f} {required_n:<15d} {status}")
        
        print("-" * 100)
    
    return sample_sizes

def plot_sample_size_comparison(sample_sizes, output_dir="output/figures/other"):
    """
    可视化不同效应量计算方法下的样本量需求比较
    
    参数:
        sample_sizes: 样本量需求字典
        output_dir: 输出目录
    """
    if sample_sizes is None:
        print("无样本量数据可绘图")
        return
    
    # 创建输出目录
    os.makedirs(os.path.join(project_root, output_dir), exist_ok=True)
    
    # 提取80%和90%功效下的数据
    power_80 = sample_sizes.get('power_80', {})
    power_90 = sample_sizes.get('power_90', {})
    
    if not power_80 or not power_90:
        print("样本量数据不完整，无法绘制比较图")
        return
    
    # 创建基本变量的样本量比较图
    basic_vars = [var for var in power_80.keys() if var not in ['gender_pwv_diff', 'age_group_pwv_diff']]
    
    if basic_vars:
        plt.figure(figsize=(14, 8))
        
        # 数据准备
        var_names = []
        std_80_samples = []
        real_80_samples = []
        std_90_samples = []
        real_90_samples = []
        actual_samples = []
        
        for var in basic_vars:
            if var in power_80 and var in power_90:
                var_names.append(var)
                std_80_samples.append(power_80[var]['std_effect']['required'])
                real_80_samples.append(power_80[var]['real_effect']['required'])
                std_90_samples.append(power_90[var]['std_effect']['required'])
                real_90_samples.append(power_90[var]['real_effect']['required'])
                actual_samples.append(power_80[var]['std_effect']['actual'])  # 实际样本量相同
        
        # 设置柱状图
        x = np.arange(len(var_names))
        width = 0.18
        
        fig, ax = plt.subplots(figsize=(14, 8))
        rects1 = ax.bar(x - width*1.5, std_80_samples, width, label='标准化效应量 (80%功效)')
        rects2 = ax.bar(x - width*0.5, real_80_samples, width, label='真实效应量 (80%功效)')
        rects3 = ax.bar(x + width*0.5, std_90_samples, width, label='标准化效应量 (90%功效)')
        rects4 = ax.bar(x + width*1.5, real_90_samples, width, label='真实效应量 (90%功效)')
        
        # 添加实际样本量线
        for i, actual in enumerate(actual_samples):
            ax.plot([x[i] - width*2, x[i] + width*2], [actual, actual], 'r--', linewidth=1.5)
        
        # 添加实际样本量标记
        ax.plot([], [], 'r--', label='实际样本量')
        
        # 设置图表属性
        ax.set_xlabel('变量')
        ax.set_ylabel('所需样本量')
        ax.set_title('不同效应量计算方法和统计功效下的样本量需求比较')
        ax.set_xticks(x)
        ax.set_xticklabels(var_names, rotation=45, ha='right')
        ax.legend()
        
        # 添加数值标签
        def add_labels(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height}',
                            xy=(rect.get_x() + rect.get_width()/2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=8)
        
        add_labels(rects1)
        add_labels(rects2)
        add_labels(rects3)
        add_labels(rects4)
        
        fig.tight_layout()
        
        # 应用中文字体设置
        try:
            apply_to_figure(fig)
        except:
            pass
        
        # 保存图表
        output_path = os.path.join(project_root, output_dir, "sample_size_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"样本量比较图已保存至: {output_path}")
        plt.close()
    
    # 创建分组比较样本量图
    group_vars = ['gender_pwv_diff', 'age_group_pwv_diff']
    group_vars_present = [var for var in group_vars if var in power_80 and var in power_90]
    
    if group_vars_present:
        plt.figure(figsize=(10, 6))
        
        # 数据准备
        var_names = []
        samples_80 = []
        samples_90 = []
        actual_samples = []
        
        for var in group_vars_present:
            var_names.append('性别PWV差异' if var == 'gender_pwv_diff' else '年龄组PWV差异')
            samples_80.append(power_80[var]['required'])
            samples_90.append(power_90[var]['required'])
            actual_samples.append(power_80[var]['actual'])
        
        # 设置柱状图
        x = np.arange(len(var_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, samples_80, width, label='80%功效')
        rects2 = ax.bar(x + width/2, samples_90, width, label='90%功效')
        
        # 添加实际样本量线
        for i, actual in enumerate(actual_samples):
            ax.plot([x[i] - width, x[i] + width], [actual, actual], 'r--', linewidth=1.5)
        
        # 添加实际样本量标记
        ax.plot([], [], 'r--', label='实际样本量')
        
        # 设置图表属性
        ax.set_xlabel('比较类型')
        ax.set_ylabel('所需样本量')
        ax.set_title('不同统计功效下组间比较所需的样本量')
        ax.set_xticks(x)
        ax.set_xticklabels(var_names)
        ax.legend()
        
        # 添加数值标签
        def add_labels(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height}',
                            xy=(rect.get_x() + rect.get_width()/2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')
        
        add_labels(rects1)
        add_labels(rects2)
        
        fig.tight_layout()
        
        # 应用中文字体设置
        try:
            apply_to_figure(fig)
        except:
            pass
        
        # 保存图表
        output_path = os.path.join(project_root, output_dir, "group_comparison_sample_size.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"组间比较样本量图已保存至: {output_path}")
        plt.close()

def power_analysis_curve(effect_sizes, output_dir="output/figures/other"):
    """
    绘制不同效应量下的功效分析曲线
    
    参数:
        effect_sizes: 效应量字典
        output_dir: 输出目录
    """
    if effect_sizes is None:
        print("无效应量数据可绘制功效曲线")
        return
    
    # 创建输出目录
    os.makedirs(os.path.join(project_root, output_dir), exist_ok=True)
    
    # 要分析的主要变量（连续型）
    continuous_vars = [var for var in effect_sizes.keys() if var not in ['gender_pwv_diff', 'age_group_pwv_diff']]
    
    for var in continuous_vars:
        # 提取效应量数据
        data = effect_sizes[var]
        actual_n = data['n']
        std_effect = data['std_effect']
        real_effect = data['real_effect']
        
        # 创建样本量范围
        sample_sizes = np.arange(10, 400, 10)
        
        # 计算不同样本量下的功效
        power_std = np.array([
            smp.TTestIndPower().power(effect_size=std_effect, nobs1=n, alpha=0.05)
            for n in sample_sizes
        ])
        
        power_real = np.array([
            smp.TTestIndPower().power(effect_size=real_effect, nobs1=n, alpha=0.05)
            for n in sample_sizes
        ])
        
        # 绘制功效曲线
        plt.figure(figsize=(10, 6))
        plt.plot(sample_sizes, power_std, 'b-', label=f'标准化效应量 ({std_effect:.3f})')
        plt.plot(sample_sizes, power_real, 'g-', label=f'真实效应量 ({real_effect:.4f})')
        
        # 标记80%和90%功效线
        plt.axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='80%功效')
        plt.axhline(y=0.9, color='m', linestyle='--', alpha=0.7, label='90%功效')
        
        # 标记实际样本量
        plt.axvline(x=actual_n, color='k', linestyle='--', alpha=0.7, label=f'实际样本量 ({actual_n})')
        
        # 找到80%和90%功效对应的样本量
        std_80_idx = np.where(power_std >= 0.8)[0]
        std_90_idx = np.where(power_std >= 0.9)[0]
        real_80_idx = np.where(power_real >= 0.8)[0]
        real_90_idx = np.where(power_real >= 0.9)[0]
        
        n_std_80 = sample_sizes[std_80_idx[0]] if len(std_80_idx) > 0 else float('inf')
        n_std_90 = sample_sizes[std_90_idx[0]] if len(std_90_idx) > 0 else float('inf')
        n_real_80 = sample_sizes[real_80_idx[0]] if len(real_80_idx) > 0 else float('inf')
        n_real_90 = sample_sizes[real_90_idx[0]] if len(real_90_idx) > 0 else float('inf')
        
        # 设置图表属性
        plt.xlabel('样本量')
        plt.ylabel('统计功效')
        plt.title(f'{var}的功效分析曲线')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 添加注释
        annotation_text = (
            f"标准化效应量所需样本量:\n"
            f"  - 80%功效: {n_std_80 if n_std_80 != float('inf') else '> 400'}\n"
            f"  - 90%功效: {n_std_90 if n_std_90 != float('inf') else '> 400'}\n\n"
            f"真实效应量所需样本量:\n"
            f"  - 80%功效: {n_real_80 if n_real_80 != float('inf') else '> 400'}\n"
            f"  - 90%功效: {n_real_90 if n_real_90 != float('inf') else '> 400'}\n\n"
            f"实际样本量: {actual_n}"
        )
        
        plt.annotate(annotation_text, xy=(0.02, 0.02), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                     fontsize=9, verticalalignment='bottom')
        
        # 应用中文字体设置
        try:
            apply_to_figure(plt.gcf())
        except:
            pass
        
        # 保存图表
        output_path = os.path.join(project_root, output_dir, f"{var}_power_curve.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"{var}的功效曲线已保存至: {output_path}")
        plt.close()
    
    # 绘制组间比较功效曲线
    group_vars = ['gender_pwv_diff', 'age_group_pwv_diff']
    for var in group_vars:
        if var in effect_sizes:
            # 提取效应量数据
            data = effect_sizes[var]
            actual_n = data['n']
            effect = data['effect']
            
            # 创建样本量范围
            sample_sizes = np.arange(10, 400, 10)
            
            # 计算不同样本量下的功效
            if var == 'gender_pwv_diff':
                # t检验功效
                power_values = np.array([
                    smp.TTestIndPower().power(effect_size=effect, nobs1=n, alpha=0.05)  # 修改为nobs1
                    for n in sample_sizes
                ])
            else:  # age_group_pwv_diff
                # ANOVA功效
                k_groups = len(data.get('group_means', []))
                power_values = np.array([
                    smp.FTestAnovaPower().power(effect_size=effect, nobs=n*k_groups, alpha=0.05, k_groups=k_groups)
                    for n in sample_sizes
                ])
            
            # 绘制功效曲线
            plt.figure(figsize=(10, 6))
            plt.plot(sample_sizes, power_values, 'b-', label=f'效应量 ({effect:.4f})')
            
            # 标记80%和90%功效线
            plt.axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='80%功效')
            plt.axhline(y=0.9, color='m', linestyle='--', alpha=0.7, label='90%功效')
            
            # 标记实际样本量
            plt.axvline(x=actual_n, color='k', linestyle='--', alpha=0.7, label=f'实际样本量 ({actual_n})')
            
            # 找到80%和90%功效对应的样本量
            power_80_idx = np.where(power_values >= 0.8)[0]
            power_90_idx = np.where(power_values >= 0.9)[0]
            
            n_80 = sample_sizes[power_80_idx[0]] if len(power_80_idx) > 0 else float('inf')
            n_90 = sample_sizes[power_90_idx[0]] if len(power_90_idx) > 0 else float('inf')
            
            # 设置图表属性
            plt.xlabel('样本量')
            plt.ylabel('统计功效')
            var_name = '性别PWV差异' if var == 'gender_pwv_diff' else '年龄组PWV差异'
            plt.title(f'{var_name}的功效分析曲线')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # 添加注释
            annotation_text = (
                f"所需样本量:\n"
                f"  - 80%功效: {n_80 if n_80 != float('inf') else '> 400'}\n"
                f"  - 90%功效: {n_90 if n_90 != float('inf') else '> 400'}\n\n"
                f"实际样本量: {actual_n}"
            )
            
            plt.annotate(annotation_text, xy=(0.02, 0.02), xycoords='axes fraction',
                         bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                         fontsize=9, verticalalignment='bottom')
            
            # 应用中文字体设置
            try:
                apply_to_figure(plt.gcf())
            except:
                pass
            
            # 保存图表
            output_path = os.path.join(project_root, output_dir, f"{var}_power_curve.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"{var_name}的功效曲线已保存至: {output_path}")
            plt.close()

def export_results_to_excel(effect_sizes, sample_sizes, output_dir="output/tables"):
    """
    将效应量和样本量结果导出到Excel文件
    
    参数:
        effect_sizes: 效应量字典
        sample_sizes: 样本量需求字典
        output_dir: 输出目录
    """
    if effect_sizes is None or sample_sizes is None:
        print("无数据可导出到Excel")
        return
    
    # 创建输出目录
    os.makedirs(os.path.join(project_root, output_dir), exist_ok=True)
    
    # 1. 创建效应量表
    effect_data = []
    
    # 处理单变量效应量
    continuous_vars = [var for var in effect_sizes.keys() if var not in ['gender_pwv_diff', 'age_group_pwv_diff']]
    for var in continuous_vars:
        data = effect_sizes[var]
        effect_data.append({
            '变量名': var,
            '样本量': data['n'],
            '均值': data['mean'],
            '标准差': data['std'],
            '标准化效应量': data['std_effect'],
            '真实效应量': data['real_effect']
        })
    
    # 处理组间比较效应量
    if 'gender_pwv_diff' in effect_sizes:
        data = effect_sizes['gender_pwv_diff']
        effect_data.append({
            '变量名': '性别PWV差异',
            '样本量': data['n'],
            '均值': f"男:{data['group1_mean']:.2f}, 女:{data['group2_mean']:.2f}",
            '标准差': f"男:{data['group1_std']:.2f}, 女:{data['group2_std']:.2f}",
            '效应量': data['effect']
        })
    
    if 'age_group_pwv_diff' in effect_sizes:
        data = effect_sizes['age_group_pwv_diff']
        effect_data.append({
            '变量名': '年龄组PWV差异',
            '样本量': data['n'],
            '分组均值': ", ".join([f"{m:.2f}" for m in data['group_means']]),
            '分组大小': ", ".join([f"{s}" for s in data['group_sizes']]),
            '效应量': data['effect']
        })
    
    effect_df = pd.DataFrame(effect_data)
    
    # 2. 创建样本量需求表
    sample_data = []
    
    # 80%功效数据
    power_80 = sample_sizes.get('power_80', {})
    # 90%功效数据
    power_90 = sample_sizes.get('power_90', {})
    
    if power_80 and power_90:
        # 处理单变量样本量
        for var in continuous_vars:
            if var in power_80 and var in power_90:
                var_data_80 = power_80[var]
                var_data_90 = power_90[var]
                
                sample_data.append({
                    '变量名': var,
                    '实际样本量': var_data_80['std_effect']['actual'],
                    '标准化效应量': var_data_80['std_effect']['effect'],
                    '80%功效所需样本量(标准化)': var_data_80['std_effect']['required'],
                    '90%功效所需样本量(标准化)': var_data_90['std_effect']['required'],
                    '是否满足80%功效(标准化)': var_data_80['std_effect']['sufficient'],
                    '是否满足90%功效(标准化)': var_data_90['std_effect']['sufficient'],
                    '真实效应量': var_data_80['real_effect']['effect'],
                    '80%功效所需样本量(真实)': var_data_80['real_effect']['required'],
                    '90%功效所需样本量(真实)': var_data_90['real_effect']['required'],
                    '是否满足80%功效(真实)': var_data_80['real_effect']['sufficient'],
                    '是否满足90%功效(真实)': var_data_90['real_effect']['sufficient']
                })
        
        # 处理组间比较样本量
        group_vars = ['gender_pwv_diff', 'age_group_pwv_diff']
        for var in group_vars:
            if var in power_80 and var in power_90:
                var_data_80 = power_80[var]
                var_data_90 = power_90[var]
                
                var_name = '性别PWV差异' if var == 'gender_pwv_diff' else '年龄组PWV差异'
                sample_data.append({
                    '变量名': var_name,
                    '实际样本量': var_data_80['actual'],
                    '效应量': var_data_80['effect'],
                    '80%功效所需样本量': var_data_80['required'],
                    '90%功效所需样本量': var_data_90['required'],
                    '是否满足80%功效': var_data_80['sufficient'],
                    '是否满足90%功效': var_data_90['sufficient']
                })
    
    sample_df = pd.DataFrame(sample_data)
    
    # 创建Excel写入器
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    excel_path = os.path.join(project_root, output_dir, f"样本量分析结果_{timestamp}.xlsx")
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        effect_df.to_excel(writer, sheet_name='效应量计算结果', index=False)
        sample_df.to_excel(writer, sheet_name='样本量需求分析', index=False)
        
        # 自动调整列宽
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            for i, col in enumerate(effect_df.columns if sheet_name == '效应量计算结果' else sample_df.columns):
                max_width = max(
                    len(str(col)),
                    effect_df[col].astype(str).map(len).max() if sheet_name == '效应量计算结果' else sample_df[col].astype(str).map(len).max()
                )
                worksheet.column_dimensions[chr(65 + i)].width = max_width + 2
    
    print(f"分析结果已导出到Excel: {excel_path}")
    
    return excel_path

def main():
    """主函数"""
    # 1. 加载数据
    print("="*80)
    print("PWV样本量与统计功效分析程序")
    print("="*80)
    print("\n加载数据...")
    df = load_data()
    
    if df is None:
        print("无法加载数据，程序退出")
        return
    
    # 2. 分析数据缺失情况
    missing_analysis = missing_data_analysis(df)
    
    # 3. 计算效应量
    effect_sizes = calculate_effect_sizes(df)
    
    # 4. 计算样本量需求
    sample_sizes = calculate_required_sample_size(effect_sizes, powers=[0.8, 0.9])
    
    # 5. 绘制样本量比较图
    plot_sample_size_comparison(sample_sizes)
    
    # 6. 绘制功效分析曲线
    power_analysis_curve(effect_sizes)
    
    # 7. 导出结果到Excel
    excel_path = export_results_to_excel(effect_sizes, sample_sizes)
    
    print("\n"+"="*80)
    print("分析完成！")
    print(f"结果已保存到 {excel_path}")
    print("="*80)

if __name__ == "__main__":
    main() 