#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
血管年龄分析与拟合脚本
基于PWV数据拟合血管年龄，并与实际年龄进行比较分析
生成比较分布图和统计数据
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
except ImportError as e:
    print(f"导入项目模块时出错: {e}")
    print("尝试使用替代方法...")

# 设置中文字体
try:
    chinese_font = setup()
except (ImportError, NameError) as e:
    print(f"设置中文字体时出错: {e}")
    # 修复字体设置
    chinese_font = FontProperties()
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

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
                    numeric_cols = ['age', 'pwv', 'cfpwv_速度', 'bapwv_右侧_速度', 'bapwv_左侧_速度']
                    for col in numeric_cols:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    return df
            
            print("未找到数据文件，使用模拟数据...")
            # 创建模拟数据
            n = 221  # 与实际数据相近的样本量
            mock_data = {
                'age': np.random.normal(58, 10, n),
                'pwv': np.random.normal(9.5, 1.8, n),
                'gender': np.random.choice([0, 1], n),
                'cfpwv_速度': np.random.normal(9.25, 1.5, n),
                'bapwv_右侧_速度': np.random.normal(11.1, 2.0, n),
                'bapwv_左侧_速度': np.random.normal(11.3, 2.0, n),
                '收缩压': np.random.normal(130, 15, n),
                '舒张压': np.random.normal(80, 10, n)
            }
            # 添加一些与年龄相关的相关性
            for i in range(n):
                age_effect = mock_data['age'][i] * 0.03  # 年龄对PWV的影响
                mock_data['pwv'][i] += age_effect
                mock_data['cfpwv_速度'][i] += age_effect
                mock_data['bapwv_右侧_速度'][i] += age_effect
                mock_data['bapwv_左侧_速度'][i] += age_effect
            
            df = pd.DataFrame(mock_data)
            return df
        except Exception as inner_e:
            print(f"直接加载数据文件也失败: {inner_e}")
            return None
    return None

def fit_vascular_age_model(df, pwv_col='pwv', output_dir="output/figures/other"):
    """
    拟合血管年龄模型
    
    参数:
        df: 数据DataFrame
        pwv_col: 用于拟合的PWV列名
        output_dir: 输出图表的目录
    
    返回:
        拟合好的模型和特征重要性
    """
    print("\n拟合血管年龄模型...")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查必要的列是否存在
    required_cols = ['age', pwv_col]
    for col in required_cols:
        if col not in df.columns:
            print(f"错误: 列 '{col}' 在数据集中未找到，无法进行建模")
            return None, None
    
    # 数据准备
    # 1. 先尝试使用多个特征进行拟合
    # PWV和基本人口学特征作为自变量
    potential_features = [
        pwv_col, 'gender', '收缩压', '舒张压', 'height', 'weight', 'bmi', 
        'cfpwv_速度', 'bapwv_右侧_速度', 'bapwv_左侧_速度'
    ]
    
    # 选择存在的特征列
    features = [col for col in potential_features if col in df.columns]
    
    # 确保至少有一个主要特征存在
    if pwv_col not in features:
        print(f"错误: 主要特征 '{pwv_col}' 不存在或全为NaN")
        return None, None
    
    # 准备训练数据，删除包含缺失值的行
    model_data = df[features + ['age']].dropna()
    
    if len(model_data) < 10:
        print("错误: 有效数据不足，无法进行建模")
        return None, None
    
    print(f"建模使用 {len(model_data)}/{len(df)} 行有效数据")
    X = model_data[features]
    y = model_data['age']
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # 尝试不同模型并选择最佳模型
    models = {
        "线性回归": LinearRegression(),
        "随机森林": RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    best_model = None
    best_r2 = -float('inf')
    model_results = {}
    
    print("\n测试不同模型的性能:")
    print("-" * 60)
    print(f"{'模型名称':<15} {'R²':<10} {'MAE':<10} {'RMSE':<10} {'交叉验证 R²':<15}")
    print("-" * 60)
    
    for name, model in models.items():
        # 训练模型
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 评估性能
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # 交叉验证
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        cv_r2 = np.mean(cv_scores)
        
        model_results[name] = {
            'model': model,
            'r2': r2,
            'cv_r2': cv_r2,
            'mae': mae,
            'rmse': rmse
        }
        
        print(f"{name:<15} {r2:<10.3f} {mae:<10.3f} {rmse:<10.3f} {cv_r2:<15.3f}")
        
        # 选择最佳模型（基于交叉验证R²）
        if cv_r2 > best_r2:
            best_r2 = cv_r2
            best_model = model
    
    print("-" * 60)
    best_model_name = next(name for name, results in model_results.items() if results['model'] == best_model)
    print(f"选择 {best_model_name} 作为最终模型，交叉验证 R² = {best_r2:.3f}")
    
    # 获取特征重要性（如果适用）
    feature_importance = None
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = dict(zip(features, best_model.feature_importances_))
        importance_df = pd.DataFrame({
            '特征': features,
            '重要性': best_model.feature_importances_
        }).sort_values('重要性', ascending=False)
        
        print("\n特征重要性:")
        print(importance_df)
        
        # 绘制特征重要性图
        fig = plt.figure(figsize=(10, 6))
        sns.barplot(x='重要性', y='特征', data=importance_df)
        plt.title('血管年龄模型特征重要性', fontproperties=chinese_font)
        
        # 应用字体到整个图表
        try:
            apply_to_figure(fig, chinese_font)
        except Exception as e:
            print(f"应用中文字体到特征重要性图表时出错: {e}")
            
        plt.tight_layout()
        importance_file = os.path.join(output_dir, "vascular_age_feature_importance.png")
        plt.savefig(importance_file, dpi=300)
        plt.close()
        print(f"特征重要性图已保存至: {importance_file}")
    elif hasattr(best_model, 'coef_'):
        coefs = best_model.coef_
        feature_importance = dict(zip(features, coefs))
        importance_df = pd.DataFrame({
            '特征': features,
            '系数': coefs
        }).sort_values('系数', ascending=False)
        
        print("\n线性模型系数:")
        print(importance_df)
        
        # 绘制系数图
        fig = plt.figure(figsize=(10, 6))
        sns.barplot(x='系数', y='特征', data=importance_df)
        plt.title('血管年龄线性模型系数', fontproperties=chinese_font)
        
        # 应用字体到整个图表
        try:
            apply_to_figure(fig, chinese_font)
        except Exception as e:
            print(f"应用中文字体到模型系数图表时出错: {e}")
            
        plt.tight_layout()
        coef_file = os.path.join(output_dir, "vascular_age_model_coefficients.png")
        plt.savefig(coef_file, dpi=300)
        plt.close()
        print(f"模型系数图已保存至: {coef_file}")
    
    return best_model, feature_importance, features

def calculate_vascular_age(df, model, features, pwv_col='pwv'):
    """
    计算所有样本的血管年龄
    
    参数:
        df: 包含PWV和年龄数据的DataFrame
        model: 训练好的血管年龄模型
        features: 用于预测的特征列表
        pwv_col: PWV列名
    
    返回:
        添加了血管年龄和差异列的DataFrame
    """
    print("\n计算血管年龄...")
    
    # 创建副本以避免修改原始数据
    result_df = df.copy()
    
    # 对缺失值进行处理（使用中位数填充）
    X_pred = result_df[features].copy()
    for col in features:
        if X_pred[col].isnull().any():
            median_val = X_pred[col].median()
            X_pred[col].fillna(median_val, inplace=True)
            print(f"列 '{col}' 的缺失值已用中位数 {median_val:.2f} 填充")
    
    # 预测血管年龄
    result_df['vascular_age'] = model.predict(X_pred)
    
    # 计算血管年龄与实际年龄的差异
    result_df['age_difference'] = result_df['vascular_age'] - result_df['age']
    
    # 分类血管年龄状态
    def categorize_vascular_age(row):
        diff = row['age_difference']
        if diff < -5:
            return "年轻"
        elif diff > 5:
            return "衰老"
        else:
            return "正常"
    
    result_df['vascular_status'] = result_df.apply(categorize_vascular_age, axis=1)
    
    # 打印血管年龄统计结果
    print("\n血管年龄统计:")
    print(result_df[['age', 'vascular_age', 'age_difference']].describe())
    
    # 统计血管年龄状态分布
    status_counts = result_df['vascular_status'].value_counts()
    print("\n血管年龄状态分布:")
    status_pct = status_counts / status_counts.sum() * 100
    status_df = pd.DataFrame({
        '数量': status_counts,
        '百分比(%)': status_pct.round(2)
    })
    print(status_df)
    
    return result_df

def analyze_vascular_age(df, output_dir="output/figures/other"):
    """
    分析血管年龄与实际年龄的关系并生成图表
    
    参数:
        df: 包含血管年龄和实际年龄的DataFrame
        output_dir: 输出图表的目录
    """
    print("\n分析血管年龄与实际年龄的关系...")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查必要的列是否存在
    required_cols = ['age', 'vascular_age', 'age_difference', 'vascular_status']
    for col in required_cols:
        if col not in df.columns:
            print(f"错误: 列 '{col}' 在数据集中未找到，无法进行分析")
            return
    
    # 获取血管状态分布数据用于保存
    status_counts = df['vascular_status'].value_counts()
    status_pct = status_counts / status_counts.sum() * 100
    status_df = pd.DataFrame({
        '数量': status_counts,
        '百分比(%)': status_pct.round(2)
    })
    
    # 1. 血管年龄 vs 实际年龄散点图
    fig = plt.figure(figsize=(10, 8))
    scatter = plt.scatter(df['age'], df['vascular_age'], c=df['age_difference'], 
                         cmap='coolwarm', alpha=0.7, s=50)
    
    # 添加对角线（理想情况下，血管年龄=实际年龄）
    min_age = min(df['age'].min(), df['vascular_age'].min())
    max_age = max(df['age'].max(), df['vascular_age'].max())
    plt.plot([min_age, max_age], [min_age, max_age], 'k--', alpha=0.5)
    
    # 添加线性回归线
    z = np.polyfit(df['age'], df['vascular_age'], 1)
    p = np.poly1d(z)
    plt.plot(np.sort(df['age']), p(np.sort(df['age'])), "r-", alpha=0.7)
    corr = df['age'].corr(df['vascular_age'])
    plt.annotate(f'相关系数: {corr:.3f}', xy=(0.05, 0.95), xycoords='axes fraction', 
                fontsize=12, fontproperties=chinese_font)
    
    # 添加标签和标题
    plt.xlabel('实际年龄 (岁)', fontproperties=chinese_font)
    plt.ylabel('血管年龄 (岁)', fontproperties=chinese_font)
    plt.title('血管年龄与实际年龄的关系', fontproperties=chinese_font)
    plt.colorbar(scatter, label='年龄差异 (岁)')
    plt.grid(alpha=0.3)
    
    # 应用字体到整个图表
    try:
        apply_to_figure(fig, chinese_font)
    except Exception as e:
        print(f"应用中文字体到散点图时出错: {e}")
    
    # 保存图表
    scatter_file = os.path.join(output_dir, "vascular_age_vs_real_age_scatter.png")
    plt.tight_layout()
    plt.savefig(scatter_file, dpi=300)
    plt.close()
    print(f"血管年龄散点图已保存至: {scatter_file}")
    
    # 2. 年龄差异（血管年龄-实际年龄）的分布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 直方图
    sns.histplot(df['age_difference'], kde=True, ax=ax1)
    ax1.axvline(x=0, color='r', linestyle='--', alpha=0.7)
    ax1.set_xlabel('血管年龄 - 实际年龄 (岁)', fontproperties=chinese_font)
    ax1.set_ylabel('频数', fontproperties=chinese_font)
    ax1.set_title('年龄差异分布', fontproperties=chinese_font)
    
    # 标记正态分布参数
    mean_diff = df['age_difference'].mean()
    std_diff = df['age_difference'].std()
    ax1.text(0.05, 0.95, f'均值: {mean_diff:.2f}\n标准差: {std_diff:.2f}', 
             transform=ax1.transAxes, fontproperties=chinese_font, 
             verticalalignment='top')
    
    # 箱线图，按性别分组（如果有性别列）
    if 'gender' in df.columns:
        gender_labels = {0: '女性', 1: '男性'}
        df['gender_label'] = df['gender'].map(gender_labels)
        sns.boxplot(x='gender_label', y='age_difference', data=df, ax=ax2)
        ax2.set_xlabel('性别', fontproperties=chinese_font)
        ax2.set_ylabel('血管年龄 - 实际年龄 (岁)', fontproperties=chinese_font)
        ax2.set_title('不同性别的年龄差异比较', fontproperties=chinese_font)
        
        # 进行t检验，判断性别差异是否显著
        male_diff = df[df['gender'] == 1]['age_difference']
        female_diff = df[df['gender'] == 0]['age_difference']
        t_stat, p_val = stats.ttest_ind(male_diff, female_diff, equal_var=False)
        sig_text = f"t检验: t={t_stat:.2f}, p={p_val:.4f}"
        if p_val < 0.05:
            sig_text += "\n差异显著 (p<0.05)"
        ax2.text(0.05, 0.95, sig_text, transform=ax2.transAxes, 
                fontproperties=chinese_font, verticalalignment='top')
    else:
        # 如果没有性别列，就用血管状态作为分组变量
        sns.boxplot(x='vascular_status', y='age_difference', data=df, ax=ax2)
        ax2.set_xlabel('血管状态', fontproperties=chinese_font)
        ax2.set_ylabel('血管年龄 - 实际年龄 (岁)', fontproperties=chinese_font)
        ax2.set_title('不同血管状态的年龄差异比较', fontproperties=chinese_font)
    
    # 应用字体到整个图表
    try:
        for ax in [ax1, ax2]:
            for label in ax.get_xticklabels():
                label.set_fontproperties(chinese_font)
            for label in ax.get_yticklabels():
                label.set_fontproperties(chinese_font)
    except Exception as e:
        print(f"应用中文字体到年龄差异分布图时出错: {e}")
    
    plt.tight_layout()
    dist_file = os.path.join(output_dir, "vascular_age_difference_distribution.png")
    plt.savefig(dist_file, dpi=300)
    plt.close()
    print(f"年龄差异分布图已保存至: {dist_file}")
    
    # 3. 年龄组分析
    # 创建年龄组
    age_bins = [0, 40, 50, 60, 70, 100]
    age_labels = ['<40', '40-49', '50-59', '60-69', '≥70']
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 年龄组的血管年龄差异箱线图
    sns.boxplot(x='age_group', y='age_difference', data=df, ax=ax1)
    ax1.set_xlabel('年龄组', fontproperties=chinese_font)
    ax1.set_ylabel('血管年龄 - 实际年龄 (岁)', fontproperties=chinese_font)
    ax1.set_title('不同年龄组的血管年龄差异', fontproperties=chinese_font)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # 年龄组的血管状态百分比堆叠条形图
    status_by_age = pd.crosstab(df['age_group'], df['vascular_status'], normalize='index') * 100
    status_by_age.plot(kind='bar', stacked=True, ax=ax2, colormap='viridis')
    ax2.set_xlabel('年龄组', fontproperties=chinese_font)
    ax2.set_ylabel('比例 (%)', fontproperties=chinese_font)
    ax2.set_title('不同年龄组的血管状态分布', fontproperties=chinese_font)
    ax2.legend(title='血管状态', prop=chinese_font)
    
    # 应用字体到整个图表
    try:
        for ax in [ax1, ax2]:
            for label in ax.get_xticklabels():
                label.set_fontproperties(chinese_font)
            for label in ax.get_yticklabels():
                label.set_fontproperties(chinese_font)
            
            # 设置图例文本
            legend = ax.get_legend()
            if legend:
                for text in legend.get_texts():
                    text.set_fontproperties(chinese_font)
                if legend.get_title():
                    legend.get_title().set_fontproperties(chinese_font)
    except Exception as e:
        print(f"应用中文字体到年龄组分析图时出错: {e}")
    
    plt.tight_layout()
    age_group_file = os.path.join(output_dir, "vascular_age_by_age_group.png")
    plt.savefig(age_group_file, dpi=300)
    plt.close()
    print(f"年龄组分析图已保存至: {age_group_file}")
    
    # 统计每个年龄组的血管年龄情况
    age_group_stats = df.groupby('age_group').agg({
        'age': ['count', 'mean'],
        'vascular_age': ['mean', 'std'],
        'age_difference': ['mean', 'std', lambda x: (x > 5).mean() * 100]
    })
    age_group_stats.columns = [f'{col[0]}_{col[1]}' if col[1] != '<lambda_0>' else '衰老比例(%)' 
                              for col in age_group_stats.columns]
    
    print("\n各年龄组血管年龄统计:")
    print(age_group_stats)
    
    # 保存统计结果到Excel
    stats_file = os.path.join(output_dir, "vascular_age_statistics.xlsx")
    with pd.ExcelWriter(stats_file) as writer:
        age_group_stats.to_excel(writer, sheet_name='年龄组统计')
        status_df.to_excel(writer, sheet_name='血管状态分布')
        
        # 如果有性别列，添加性别分组统计
        if 'gender' in df.columns:
            gender_stats = df.groupby('gender_label').agg({
                'age': ['count', 'mean'],
                'vascular_age': ['mean', 'std'],
                'age_difference': ['mean', 'std', lambda x: (x > 5).mean() * 100]
            })
            gender_stats.columns = [f'{col[0]}_{col[1]}' if col[1] != '<lambda_0>' else '衰老比例(%)' 
                                  for col in gender_stats.columns]
            gender_stats.to_excel(writer, sheet_name='性别分组统计')
    
    print(f"统计结果已保存至: {stats_file}")
    
    return age_group_stats, status_df

def main():
    """主函数"""
    print("=" * 80)
    print("PWV血管年龄拟合与分析")
    print("=" * 80)
    
    # 加载数据
    df = load_data()
    if df is None:
        print("数据加载失败，无法继续分析。")
        return
    
    # 拟合血管年龄模型
    model, feature_importance, features = fit_vascular_age_model(df)
    if model is None:
        print("血管年龄模型拟合失败，无法继续分析。")
        return
    
    # 计算血管年龄
    df_with_vascular_age = calculate_vascular_age(df, model, features)
    
    # 分析血管年龄
    analyze_vascular_age(df_with_vascular_age)
    
    print("\n血管年龄分析完成！")

if __name__ == "__main__":
    main() 