#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
增强可视化模块：提供更丰富的PWV数据可视化功能
包含高级图表类型、临床风险可视化和多维数据展示功能
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

# 导入字体配置模块
import sys
scripts_dir = os.path.abspath(os.path.dirname(__file__))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)
from font_config import configure_chinese_fonts, enhance_plot_style, apply_color_scheme, create_custom_legend

# 导入data_visualization模块中的字体应用函数
from data_visualization import fix_chinese_font_display, apply_font_to_axis, apply_font_to_figure, apply_font_to_legend, CHINESE_FONT_PROP

# 配置matplotlib支持中文显示
configure_chinese_fonts()
enhance_plot_style()
apply_color_scheme("default")  # 使用默认配色方案
fix_chinese_font_display()  # 确保中文字体加载正确

def create_advanced_heatmap(df, x_var='age', y_var='bmi', z_var='pwv', save_dir='output/image'):
    """
    创建高级热力图，展示三个变量之间的关系
    
    参数:
        df: 数据DataFrame
        x_var: x轴变量
        y_var: y轴变量
        z_var: 用颜色表示的变量
        save_dir: 保存图表的目录
    
    返回:
        保存的图表文件名列表
    """
    print(f"\n创建高级热力图: {z_var} by {x_var} & {y_var}")
    
    # 确保目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 准备数据
    data = df[[x_var, y_var, z_var]].dropna()
    if len(data) == 0:
        print(f"错误: {x_var}, {y_var}, {z_var} 组合没有有效数据")
        return []
    
    output_files = []
    
    # 1. 创建2D热力图
    plt.figure(figsize=(12, 9))
    
    # 使用pivot_table创建热力图数据
    # 对x和y变量进行分箱
    x_bins = pd.cut(data[x_var], bins=10)
    y_bins = pd.cut(data[y_var], bins=10)
    
    # 创建热力图数据
    heatmap_data = data.assign(x_bin=x_bins, y_bin=y_bins).pivot_table(
        values=z_var, 
        index='y_bin', 
        columns='x_bin',
        aggfunc='mean'
    )
    
    # 绘制热力图
    ax = sns.heatmap(
        heatmap_data,
        cmap='viridis',
        annot=True,
        fmt='.1f',
        linewidths=0.5
    )
    
    plt.title(f'{z_var}与{x_var}、{y_var}的关系热力图')
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    
    # 应用中文字体
    apply_font_to_axis(ax)
    
    # 修复colorbar上的标签文本
    cbar = ax.collections[0].colorbar
    if CHINESE_FONT_PROP:
        current_label_text = cbar.ax.get_ylabel()
        if current_label_text: # 只有当标签文本存在时才设置
            cbar.ax.set_ylabel(current_label_text, fontproperties=CHINESE_FONT_PROP)
        for tick_label in cbar.ax.get_yticklabels():
            tick_label.set_fontproperties(CHINESE_FONT_PROP)
    
    plt.tight_layout()
    
    # 保存图表
    heatmap_file = f'{z_var}_by_{x_var}_{y_var}_heatmap.png'
    plt.savefig(f'{save_dir}/{heatmap_file}')
    plt.close()
    output_files.append(heatmap_file)
    
    # 2. 创建等高线图
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # 使用KDE创建平滑的等高线图
    x = data[x_var]
    y = data[y_var]
    z = data[z_var]
    
    # 创建网格
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    xi, yi = np.meshgrid(xi, yi)
    
    # 整合数据点
    positions = np.vstack([xi.ravel(), yi.ravel()])
    values = np.vstack([x, y])
    kernel = gaussian_kde(values)
    density = kernel(positions).reshape(xi.shape)
    
    # 绘制等高线图
    contour = plt.contourf(xi, yi, density, levels=15, cmap='viridis', alpha=0.7)
    cbar1 = plt.colorbar(contour, label='密度')
    
    # 添加散点图，按z值着色
    scatter = plt.scatter(x, y, c=z, cmap='coolwarm', alpha=0.6, s=50, edgecolor='k')
    cbar2 = plt.colorbar(scatter, label=z_var)
    
    # 设置标题和标签
    ax.set_title(f'{z_var}的{x_var}-{y_var}等高线图')
    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)
    
    # 应用中文字体
    apply_font_to_axis(ax)
    
    # 修复colorbar上的标签文本
    if CHINESE_FONT_PROP:
        # 安全修改 cbar1 的 ylabel
        current_label_text1 = cbar1.ax.get_ylabel()
        if current_label_text1:
            cbar1.ax.set_ylabel(current_label_text1, fontproperties=CHINESE_FONT_PROP)
        for tick_label in cbar1.ax.get_yticklabels():
            tick_label.set_fontproperties(CHINESE_FONT_PROP)
        
        # 安全修改 cbar2 的 ylabel
        current_label_text2 = cbar2.ax.get_ylabel()
        if current_label_text2:
            cbar2.ax.set_ylabel(current_label_text2, fontproperties=CHINESE_FONT_PROP)
        for label in cbar2.ax.get_yticklabels():
            label.set_fontproperties(CHINESE_FONT_PROP)
    
    plt.tight_layout()
    
    # 保存图表
    contour_file = f'{z_var}_by_{x_var}_{y_var}_contour.png'
    plt.savefig(f'{save_dir}/{contour_file}')
    plt.close()
    output_files.append(contour_file)
    
    # 3. 创建3D表面图
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 使用x, y, z直接绘制散点图
    scatter = ax.scatter(x, y, z, c=z, cmap='coolwarm', s=50, alpha=0.6)
    cbar = fig.colorbar(scatter, ax=ax, label=z_var)
    
    # 尝试拟合曲面
    try:
        # 创建meshgrid
        xi = np.linspace(x.min(), x.max(), 30)
        yi = np.linspace(y.min(), y.max(), 30)
        xi, yi = np.meshgrid(xi, yi)
        
        # 使用简单的线性回归预测z值
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(data[[x_var, y_var]], data[z_var])
        
        # 预测zi值
        grid_points = np.c_[xi.ravel(), yi.ravel()]
        zi = model.predict(grid_points).reshape(xi.shape)
        
        # 绘制预测曲面
        surf = ax.plot_surface(xi, yi, zi, cmap='viridis', alpha=0.5)
        cbar_surf = fig.colorbar(surf, ax=ax, label=f'预测{z_var}')
        
        # 设置colorbar标签字体
        if CHINESE_FONT_PROP:
            # 安全修改 cbar_surf 的 ylabel
            current_label_text_surf = cbar_surf.ax.get_ylabel()
            if current_label_text_surf:
                 cbar_surf.ax.set_ylabel(current_label_text_surf, fontproperties=CHINESE_FONT_PROP)
            for label_tick in cbar_surf.ax.get_yticklabels(): # Renamed to avoid conflict
                label_tick.set_fontproperties(CHINESE_FONT_PROP)
    except Exception as e:
        print(f"拟合3D曲面时出错: {str(e)}")
    
    # 设置坐标轴标签和标题
    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)
    ax.set_zlabel(z_var)
    ax.set_title(f'{x_var}, {y_var}与{z_var}的3D关系')
    
    # 应用中文字体到坐标轴标签
    if CHINESE_FONT_PROP:
        ax.set_title(f'{x_var}, {y_var}与{z_var}的3D关系', fontproperties=CHINESE_FONT_PROP)
        ax.set_xlabel(x_var, fontproperties=CHINESE_FONT_PROP)
        ax.set_ylabel(y_var, fontproperties=CHINESE_FONT_PROP)
        ax.set_zlabel(z_var, fontproperties=CHINESE_FONT_PROP)
        
        # 安全修改 cbar 的 ylabel
        current_label_text_main_3d = cbar.ax.get_ylabel()
        if CHINESE_FONT_PROP and current_label_text_main_3d:
            cbar.ax.set_ylabel(current_label_text_main_3d, fontproperties=CHINESE_FONT_PROP)
        for tick_label in cbar.ax.get_yticklabels(): # Ensure this is tick_label
            if CHINESE_FONT_PROP:
                 tick_label.set_fontproperties(CHINESE_FONT_PROP)
    
    plt.tight_layout()
    
    # 保存图表
    surface_file = f'{z_var}_by_{x_var}_{y_var}_3d.png'
    plt.savefig(f'{save_dir}/{surface_file}')
    plt.close()
    output_files.append(surface_file)
    
    return output_files

def create_risk_visualizations(df, save_dir='output/image'):
    """
    创建风险评估相关的可视化图表
    
    参数:
        df: 带有风险评估结果的DataFrame
        save_dir: 保存图表的目录
    
    返回:
        保存的图表文件名列表
    """
    print("\n创建风险评估可视化...")
    
    # 确保目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    output_files = []
    
    # 检查必要的风险列
    risk_columns = ['PWV风险', '综合风险等级', 'CVD风险等级', '10年CVD风险(%)']
    missing_columns = [col for col in risk_columns if col not in df.columns]
    
    if missing_columns:
        print(f"警告: 缺少风险列: {', '.join(missing_columns)}")
        # 继续处理可用的列
    
    # 1. PWV风险分布饼图
    if 'PWV风险' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 8))
        risk_counts = df['PWV风险'].value_counts()
        
        # 定义颜色映射
        colors = ['#2ca02c', '#d4c72c', '#ffa500', '#d62728']
        
        # 确保explode长度与数据匹配
        unique_risks = risk_counts.index.tolist()
        # 修复explode参数长度问题
        if len(risk_counts) > 0:
            explode = [0.1 if i == 0 else 0 for i in range(len(risk_counts))]  # 突出显示第一个类别
        else:
            explode = [0]
        
        # 如果颜色不够，复制颜色
        if len(unique_risks) > len(colors):
            colors = colors * (len(unique_risks) // len(colors) + 1)
        colors = colors[:len(unique_risks)]
        
        # 确保explode长度与risk_counts匹配
        if len(explode) != len(risk_counts):
            explode = explode[:len(risk_counts)] if len(explode) > len(risk_counts) else explode + [0] * (len(risk_counts) - len(explode))
            
        wedges, texts, autotexts = ax.pie(
            risk_counts,
            labels=risk_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            explode=explode,
            colors=colors,
            shadow=True
        )
        
        # 修复标签文本的字体
        if CHINESE_FONT_PROP:
            for text in texts:
                text.set_fontproperties(CHINESE_FONT_PROP)
            for autotext in autotexts:
                autotext.set_fontproperties(CHINESE_FONT_PROP)
        
        ax.axis('equal')
        ax.set_title('PWV风险分布')
        
        # 应用中文字体到标题
        if CHINESE_FONT_PROP:
            ax.set_title('PWV风险分布', fontproperties=CHINESE_FONT_PROP)
        
        # 保存图表
        pwv_risk_file = 'pwv_risk_distribution_pie.png'
        plt.savefig(f'{save_dir}/{pwv_risk_file}')
        plt.close()
        output_files.append(pwv_risk_file)
    
    # 2. 年龄组与风险等级的关系图
    if 'CVD风险等级' in df.columns and '年龄组' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 计算每个年龄组中各风险等级的占比
        risk_by_age = pd.crosstab(
            df['年龄组'], 
            df['CVD风险等级'], 
            normalize='index'
        ) * 100
        
        # 绘制堆叠条形图
        risk_by_age.plot(
            kind='bar',
            stacked=True,
            colormap='RdYlGn_r',
            figsize=(12, 8),
            ax=ax
        )
        
        ax.set_title('不同年龄组的CVD风险分布')
        ax.set_xlabel('年龄组')
        ax.set_ylabel('百分比 (%)')
        legend = ax.legend(title='CVD风险等级')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 应用中文字体
        apply_font_to_axis(ax)
        apply_font_to_legend(legend)
        
        plt.tight_layout()
        
        # 保存图表
        risk_by_age_file = 'cvd_risk_by_age_group.png'
        plt.savefig(f'{save_dir}/{risk_by_age_file}')
        plt.close()
        output_files.append(risk_by_age_file)
    
    # 3. 风险散点图（PWV vs 10年CVD风险）
    if 'pwv' in df.columns and '10年CVD风险(%)' in df.columns and 'age' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 确保有足够有效的数据点
        valid_data = df[['pwv', '10年CVD风险(%)', 'age']].dropna()
        
        if len(valid_data) > 10:  # 确保有足够的点
            scatter = ax.scatter(
                valid_data['pwv'],
                valid_data['10年CVD风险(%)'],
                c=valid_data['age'],
                cmap='viridis',
                alpha=0.7,
                s=70,
                edgecolor='k'
            )
            
            cbar = fig.colorbar(scatter, ax=ax, label='年龄')
            ax.set_xlabel('PWV (m/s)')
            ax.set_ylabel('10年CVD风险 (%)')
            ax.set_title('PWV与10年CVD风险的关系')
            
            # 应用中文字体到colorbar
            if CHINESE_FONT_PROP:
                current_label_text_cvd = cbar.ax.get_ylabel()
                if current_label_text_cvd:
                    cbar.ax.set_ylabel(current_label_text_cvd, fontproperties=CHINESE_FONT_PROP)
                for tick_label in cbar.ax.get_yticklabels():
                    tick_label.set_fontproperties(CHINESE_FONT_PROP)
            
            # 添加趋势线
            z = np.polyfit(valid_data['pwv'], valid_data['10年CVD风险(%)'], 1)
            p = np.poly1d(z)
            ax.plot(
                valid_data['pwv'],
                p(valid_data['pwv']),
                "r--",
                alpha=0.8,
                label=f'趋势线: y={z[0]:.2f}x+{z[1]:.2f}'
            )
            
            ax.grid(True, alpha=0.3)
            legend = ax.legend()
            
            # 应用中文字体
            apply_font_to_axis(ax)
            apply_font_to_legend(legend)
            
            plt.tight_layout()
            
            # 保存图表
            risk_scatter_file = 'pwv_vs_cvd_risk_scatter.png'
            plt.savefig(f'{save_dir}/{risk_scatter_file}')
            plt.close()
            output_files.append(risk_scatter_file)
    
    # 4. 风险热力图（血压相关）
    if '收缩压' in df.columns and '舒张压' in df.columns and 'PWV风险' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 定义风险类别颜色映射
        risk_colors = {
            '正常': '#2ca02c',
            '边缘': '#d4c72c',
            '轻度风险': '#ffa500',
            '显著风险': '#d62728',
            '未知': '#999999'
        }
        
        # 根据风险类别设置颜色
        colors = [risk_colors.get(risk, '#999999') for risk in df['PWV风险']]
        
        ax.scatter(
            df['收缩压'],
            df['舒张压'],
            c=colors,
            alpha=0.7,
            s=80,
            edgecolor='k'
        )
        
        # 添加血压分类区域
        # 正常血压
        ax.axvspan(0, 120, 0, 80/180, alpha=0.2, color='green', label='正常血压')
        # 正常高值
        ax.axvspan(120, 140, 0, 90/180, alpha=0.2, color='yellow', label='正常高值')
        # 高血压1级
        ax.axvspan(140, 160, 0, 100/180, alpha=0.2, color='orange', label='高血压1级')
        # 高血压2级
        ax.axvspan(160, 180, 0, 110/180, alpha=0.2, color='red', label='高血压2级')
        # 高血压3级
        ax.axvspan(180, 220, 0, 1, alpha=0.2, color='darkred', label='高血压3级')
        
        # 添加自定义图例
        legend_elements = [
            mpatches.Patch(color=color, label=risk)
            for risk, color in risk_colors.items()
            if risk in df['PWV风险'].unique()
        ]
        legend = ax.legend(handles=legend_elements, title='PWV风险分类')
        
        # 应用中文字体到图例
        apply_font_to_legend(legend)
        
        ax.set_xlabel('收缩压 (mmHg)')
        ax.set_ylabel('舒张压 (mmHg)')
        ax.set_title('血压与PWV风险的关系')
        ax.grid(True, alpha=0.3)
        
        # 应用中文字体到坐标轴
        apply_font_to_axis(ax)
        
        plt.tight_layout()
        
        # 保存图表
        bp_risk_file = 'bp_pwv_risk_heatmap.png'
        plt.savefig(f'{save_dir}/{bp_risk_file}')
        plt.close()
        output_files.append(bp_risk_file)
    
    # 5. 多变量关系桑基图
    if 'PWV风险' in df.columns and 'CVD风险等级' in df.columns:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # 准备桑基图数据
            source = []
            target = []
            value = []
            
            # PWV风险 -> CVD风险等级
            pwv_risk_levels = df['PWV风险'].unique()
            cvd_risk_levels = df['CVD风险等级'].unique()
            
            pwv_to_cvd = pd.crosstab(df['PWV风险'], df['CVD风险等级'])
            
            for i, pwv_risk in enumerate(pwv_risk_levels):
                for j, cvd_risk in enumerate(cvd_risk_levels):
                    if pwv_to_cvd.loc[pwv_risk, cvd_risk] > 0:
                        source.append(i)
                        target.append(len(pwv_risk_levels) + j)
                        value.append(pwv_to_cvd.loc[pwv_risk, cvd_risk])
            
            # 创建桑基图
            fig = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=list(pwv_risk_levels) + list(cvd_risk_levels)
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=value
                )
            )])
            
            fig.update_layout(
                title_text="PWV风险与CVD风险等级的关系",
                font_size=12
            )
            
            # 保存为HTML
            sankey_file = 'risk_relationship_sankey.html'
            fig.write_html(f'{save_dir}/{sankey_file}')
            
            # 保存为图片
            sankey_img_file = 'risk_relationship_sankey.png'
            fig.write_image(f'{save_dir}/{sankey_img_file}')
            
            output_files.append(sankey_img_file)
            
        except ImportError:
            print("警告: 未安装plotly，跳过桑基图创建")
    
    return output_files

def create_subgroup_visualizations(df, save_dir='output/image'):
    """
    按照子群体创建可视化图表
    
    参数:
        df: 数据DataFrame，包含子群体分类列
        save_dir: 保存图表的目录
    
    返回:
        保存的图表文件名列表
    """
    print("\n创建子群体可视化...")
    
    # 确保目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    output_files = []
    
    # 检查是否存在子群体分类列
    subgroup_columns = ['Gender', 'bmi分类', '血压状态', 'PWV风险', '综合风险等级']
    available_columns = [col for col in subgroup_columns if col in df.columns]
    
    if not available_columns:
        print("警告: 未找到子群体分类列，跳过子群体可视化")
        return []
    
    # 1. 创建子群体箱线图
    target_metrics = ['pwv', '收缩压', '舒张压', 'bmi', 'age']
    available_metrics = [col for col in target_metrics if col in df.columns]
    
    for subgroup_col in available_columns:
        fig = plt.figure(figsize=(15, 12)) # Create figure once for this subgroup_col
        
        for i, metric in enumerate(available_metrics, 1):
            ax = plt.subplot(3, 2, i)
            # 创建箱线图
            sns.boxplot(x=subgroup_col, y=metric, data=df, ax=ax)
            ax.set_title(f'按{subgroup_col}分组的{metric}分布')
            ax.tick_params(axis='x', rotation=45)
            # 应用中文字体到当前子图的坐标轴和标题
            if CHINESE_FONT_PROP:
                apply_font_to_axis(ax)
        
        plt.tight_layout() # Call tight_layout once per figure, after all subplots are drawn
        # 保存图表
        boxplot_file = f'subgroup_{subgroup_col}_boxplots.png'
        plt.savefig(f'{save_dir}/{boxplot_file}')
        plt.close()
        output_files.append(boxplot_file)
    
    # 2. 创建子群体小提琴图
    for subgroup_col in available_columns[:3]:  # 仅使用前三个分组变量，避免图表过多
        fig = plt.figure(figsize=(15, 12))
        
        for i, metric in enumerate(available_metrics[:4], 1):  # 仅使用前四个指标
            ax = plt.subplot(2, 2, i)
            # 创建小提琴图
            sns.violinplot(x=subgroup_col, y=metric, data=df, inner='quartile', ax=ax)
            ax.set_title(f'按{subgroup_col}分组的{metric}分布')
            ax.tick_params(axis='x', rotation=45)
            # 应用中文字体到当前子图的坐标轴和标题
            if CHINESE_FONT_PROP:
                apply_font_to_axis(ax)

        plt.tight_layout() # Call tight_layout once per figure
        # 保存图表
        violin_file = f'subgroup_{subgroup_col}_violinplots.png'
        plt.savefig(f'{save_dir}/{violin_file}')
        plt.close()
        output_files.append(violin_file)
    
    # 3. 创建计数图
    for subgroup_col in available_columns:
        fig = plt.figure(figsize=(12, 6))
        ax = fig.gca() # Get current axes for the single plot figure
        # 计算子群体计数
        subgroup_counts = df[subgroup_col].value_counts().sort_index()
        # 绘制条形图
        sns.barplot(x=subgroup_counts.index, y=subgroup_counts.values, ax=ax)
        ax.set_title(f'{subgroup_col}分布')
        ax.set_xlabel(subgroup_col)
        ax.set_ylabel('计数')
        ax.tick_params(axis='x', rotation=45)
        # 应用中文字体到当前图表的坐标轴和标题
        if CHINESE_FONT_PROP:
            apply_font_to_axis(ax)

        plt.tight_layout()
        # 保存图表
        count_file = f'subgroup_{subgroup_col}_counts.png'
        plt.savefig(f'{save_dir}/{count_file}')
        plt.close()
        output_files.append(count_file)
    
    # 4. 组合子群体的交互可视化（如果有足够的子群体）
    if len(available_columns) >= 2:
        # 选择前两个子群体列
        col1, col2 = available_columns[0], available_columns[1]
        
        # 创建交叉表
        crosstab = pd.crosstab(df[col1], df[col2])
        
        # 绘制热力图
        plt.figure(figsize=(12, 8))
        ax_heatmap1 = plt.gca()
        sns.heatmap(crosstab, annot=True, cmap='Blues', fmt='d', ax=ax_heatmap1)
        ax_heatmap1.set_title(f'{col1}与{col2}的关系')
        if CHINESE_FONT_PROP:
            apply_font_to_axis(ax_heatmap1)
            # Heatmap colorbar font might need specific handling if apply_font_to_axis doesn't cover it
            if ax_heatmap1.collections:
                cbar = ax_heatmap1.collections[0].colorbar
                if cbar:
                    for tick_label in cbar.ax.get_yticklabels():
                        tick_label.set_fontproperties(CHINESE_FONT_PROP)
                    if cbar.ax.get_ylabel(): # If there is a colorbar label
                         cbar.ax.set_ylabel(cbar.ax.get_ylabel(), fontproperties=CHINESE_FONT_PROP)

        plt.tight_layout()
        
        # 保存图表
        crosstab_file = f'subgroup_{col1}_{col2}_crosstab.png'
        plt.savefig(f'{save_dir}/{crosstab_file}')
        plt.close()
        output_files.append(crosstab_file)
        
        # 计算百分比
        crosstab_pct = pd.crosstab(df[col1], df[col2], normalize='index') * 100
        
        # 绘制百分比热力图
        plt.figure(figsize=(12, 8))
        ax_heatmap2 = plt.gca()
        sns.heatmap(crosstab_pct, annot=True, cmap='Blues', fmt='.1f', ax=ax_heatmap2)
        ax_heatmap2.set_title(f'{col1}与{col2}的关系 (%)')
        if CHINESE_FONT_PROP:
            apply_font_to_axis(ax_heatmap2)
            if ax_heatmap2.collections:
                cbar = ax_heatmap2.collections[0].colorbar
                if cbar:
                    for tick_label in cbar.ax.get_yticklabels():
                        tick_label.set_fontproperties(CHINESE_FONT_PROP)
                    if cbar.ax.get_ylabel():
                         cbar.ax.set_ylabel(cbar.ax.get_ylabel(), fontproperties=CHINESE_FONT_PROP)

        plt.tight_layout()
        
        # 保存图表
        crosstab_pct_file = f'subgroup_{col1}_{col2}_crosstab_pct.png'
        plt.savefig(f'{save_dir}/{crosstab_pct_file}')
        plt.close()
        output_files.append(crosstab_pct_file)
    
    return output_files

def create_all_enhanced_visualizations(df, save_dir='output/image'):
    """
    创建所有增强可视化图表
    
    参数:
        df: 数据DataFrame
        save_dir: 保存图表的目录
    
    返回:
        所有生成的图表文件列表
    """
    print("\n开始创建增强可视化图表...")
    
    # 确保目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    all_figures = []
    
    # 1. 高级热力图
    advanced_heatmap_params = [
        ('age', 'bmi', 'pwv'),
        ('收缩压', '舒张压', 'pwv'),
        ('age', 'pwv', '收缩压')
    ]
    
    for x_var, y_var, z_var in advanced_heatmap_params:
        if all(var in df.columns for var in [x_var, y_var, z_var]):
            all_figures.extend(create_advanced_heatmap(df, x_var, y_var, z_var, save_dir))
    
    # 2. 风险可视化
    all_figures.extend(create_risk_visualizations(df, save_dir))
    
    # 3. 子群体可视化
    all_figures.extend(create_subgroup_visualizations(df, save_dir))
    
    print(f"\n共创建了 {len(all_figures)} 个增强可视化图表，保存在 {save_dir} 目录中")
    
    return all_figures

if __name__ == '__main__':
    # 导入必要的模块
    from data_processing import load_and_prepare_data
    from advanced_analysis import run_advanced_analysis
    from clinical_analysis import run_clinical_analysis
    
    # 加载和预处理数据
    df = load_and_prepare_data()
    
    # 执行高级分析和临床分析
    advanced_results, df = run_advanced_analysis(df)
    df, clinical_results = run_clinical_analysis(df)
    
    # 创建增强可视化
    enhanced_figures = create_all_enhanced_visualizations(df) 