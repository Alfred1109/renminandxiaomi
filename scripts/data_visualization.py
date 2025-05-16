#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据可视化模块：提供PWV数据的基本可视化功能
包含分布图、相关性图表、箱线图和回归图等功能
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import platform
import urllib.request
import tempfile
import shutil
from scipy import stats
from datetime import datetime
import warnings
import logging
warnings.filterwarnings('ignore')

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('data_visualization')

# 导入字体配置模块
import sys
scripts_dir = os.path.abspath(os.path.dirname(__file__))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)
from font_config import enhance_plot_style, apply_color_scheme, create_custom_legend

# 初始化全局字体属性对象
CHINESE_FONT_PROP = None

# 强化版字体设置函数
def fix_chinese_font_display():
    """
    全面修复中文字体显示问题，包括坐标轴标签
    
    返回:
        字体属性对象或None
    """
    global CHINESE_FONT_PROP
    
    # 清理字体缓存
    try:
        import matplotlib
        cache_dir = matplotlib.get_cachedir()
        for cache_file in ['fontlist-v330.json', 'fontlist-v310.json', 'fontlist-v320.json', '.fontlist.json', 'fontlist-v300.json']:
            cache_path = os.path.join(cache_dir, cache_file)
            if os.path.exists(cache_path):
                logger.info(f"删除字体缓存: {cache_path}")
                os.remove(cache_path)
    except Exception as e:
        logger.warning(f"清理字体缓存失败: {e}")
    
    # 检查是否在WSL环境中
    is_wsl = False
    if platform.system() == 'Linux':
        try:
            with open('/proc/version', 'r') as f:
                if 'microsoft' in f.read().lower():
                    is_wsl = True
                    logger.info("检测到WSL环境，将尝试使用Windows字体")
        except:
            pass
    
    # 在WSL环境中尝试使用Windows字体
    if is_wsl:
        # Windows字体目录可能的路径
        windows_font_dirs = [
            '/mnt/c/Windows/Fonts',
            '/mnt/c/WINDOWS/Fonts'
        ]
        
        # 中文字体文件名
        chinese_font_files = [
            'simhei.ttf',      # 黑体
            'msyh.ttc',        # 微软雅黑
            'simsun.ttc',      # 宋体
            'simkai.ttf',      # 楷体
            'simfang.ttf',     # 仿宋
        ]
        
        # 搜索Windows字体目录
        for font_dir in windows_font_dirs:
            if os.path.exists(font_dir):
                logger.info(f"找到Windows字体目录: {font_dir}")
                for font_file in chinese_font_files:
                    font_path = os.path.join(font_dir, font_file)
                    if os.path.exists(font_path):
                        logger.info(f"找到Windows中文字体: {font_path}")
                        try:
                            # 创建字体属性对象
                            font_prop = fm.FontProperties(fname=font_path)
                            CHINESE_FONT_PROP = font_prop
                            
                            # 设置Matplotlib字体
                            plt.rcParams['font.family'] = ['sans-serif']
                            plt.rcParams['font.sans-serif'] = [font_prop.get_name(), 'SimHei', 'DejaVu Sans', 'sans-serif']
                            plt.rcParams['axes.unicode_minus'] = False
                            
                            return font_prop
                        except Exception as e:
                            logger.warning(f"加载Windows字体失败: {e}")
    
    # 检查用户已下载的字体
    user_font = os.path.expanduser("~/.fonts/SimHei.ttf")
    if os.path.exists(user_font):
        logger.info(f"使用用户下载的字体: {user_font}")
        font_prop = fm.FontProperties(fname=user_font)
        CHINESE_FONT_PROP = font_prop
        
        # 设置Matplotlib字体
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['font.sans-serif'] = [font_prop.get_name(), 'SimHei', 'DejaVu Sans', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        return font_prop
    
    # 尝试查找系统中的中文字体
    system_font_dirs = []
    if platform.system() == 'Linux':
        system_font_dirs = [
            '/usr/share/fonts/truetype',
            '/usr/share/fonts/opentype',
            '/usr/share/fonts/TTF',
            '/usr/local/share/fonts',
            '/usr/share/fonts/truetype/wqy',
            '/usr/share/fonts/wqy',
            '/usr/share/fonts/truetype/arphic',
            '/usr/share/fonts/noto-cjk',
            '/usr/share/fonts/truetype/noto',
            '/usr/share/fonts/chinese'
        ]
    elif platform.system() == 'Darwin':  # macOS
        system_font_dirs = [
            '/System/Library/Fonts',
            '/Library/Fonts',
            os.path.expanduser('~/Library/Fonts')
        ]
    
    # 搜索中文字体文件
    for font_dir in system_font_dirs:
        if not os.path.exists(font_dir):
            continue
            
        logger.info(f"搜索字体目录: {font_dir}")
        for root, dirs, files in os.walk(font_dir):
            for file in files:
                if file.lower().endswith(('.ttf', '.ttc', '.otf')):
                    if any(name in file.lower() for name in 
                          ['wqy', 'noto', 'cjk', 'simhei', 'simsun', 'yahei', 'kai']):
                        font_path = os.path.join(root, file)
                        logger.info(f"找到系统中文字体: {font_path}")
                        
                        try:
                            # 创建字体属性对象
                            font_prop = fm.FontProperties(fname=font_path)
                            CHINESE_FONT_PROP = font_prop
                            
                            # 设置Matplotlib字体
                            plt.rcParams['font.family'] = ['sans-serif']
                            plt.rcParams['font.sans-serif'] = [font_prop.get_name(), 'SimHei', 'DejaVu Sans', 'sans-serif']
                            plt.rcParams['axes.unicode_minus'] = False
                            
                            return font_prop
                        except Exception as e:
                            logger.warning(f"加载系统字体失败: {e}")
    
    # 如果找不到系统中文字体，尝试使用Matplotlib内置的serif字体
    logger.info("未找到系统中文字体，尝试使用Matplotlib内置字体")
    
    # 在没有中文字体的情况下设置默认字体
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Bitstream Vera Sans', 'Computer Modern Sans Serif',
                                      'Lucida Grande', 'Verdana', 'Geneva', 'Lucid', 'Arial', 'Helvetica',
                                      'Avant Garde', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 如果仍找不到中文字体，尝试下载
    return download_chinese_font()

def download_chinese_font():
    """
    下载中文字体并安装
    
    返回:
        字体属性对象或None
    """
    global CHINESE_FONT_PROP
    
    # SimHei字体下载链接，按优先级排序
    font_urls = [
        "https://github.com/StellarCN/scp_zh/raw/master/fonts/SimHei.ttf",  # 黑体
        "https://raw.githubusercontent.com/adobe-fonts/source-han-sans/release/OTF/SimplifiedChinese/SourceHanSansSC-Regular.otf",  # 思源黑体
        "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansSC-Regular.otf"  # Noto Sans SC
    ]
    
    # 创建字体目录
    font_dir = os.path.expanduser("~/.fonts")
    os.makedirs(font_dir, exist_ok=True)
    
    # 下载字体
    for url in font_urls:
        try:
            font_name = os.path.basename(url)
            download_path = os.path.join(font_dir, font_name)
            
            # 如果已存在，直接使用
            if os.path.exists(download_path):
                logger.info(f"使用已下载的字体: {download_path}")
                font_prop = fm.FontProperties(fname=download_path)
                CHINESE_FONT_PROP = font_prop
                
                # 设置Matplotlib字体
                plt.rcParams['font.family'] = ['sans-serif']
                plt.rcParams['font.sans-serif'] = [font_prop.get_name(), 'SimHei', 'DejaVu Sans', 'sans-serif']
                plt.rcParams['axes.unicode_minus'] = False
                
                return font_prop
            
            logger.info(f"开始下载字体: {url}")
            
            # 使用临时文件下载
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_path = temp_file.name
                
            urllib.request.urlretrieve(url, temp_path)
            
            # 移动到最终位置
            shutil.move(temp_path, download_path)
            
            logger.info(f"字体下载成功: {download_path}")
            
            # 创建字体属性对象
            font_prop = fm.FontProperties(fname=download_path)
            CHINESE_FONT_PROP = font_prop
            
            # 设置Matplotlib字体
            plt.rcParams['font.family'] = ['sans-serif']
            plt.rcParams['font.sans-serif'] = [font_prop.get_name(), 'SimHei', 'DejaVu Sans', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 刷新字体缓存
            try:
                fm._rebuild()
            except:
                pass
            
            return font_prop
            
        except Exception as e:
            logger.warning(f"下载字体失败: {url} - {e}")
    
    logger.error("所有字体下载尝试均失败，将使用系统默认字体")
    return None

def apply_font_to_axis(ax):
    """对坐标轴的标签和标题应用中文字体"""
    if CHINESE_FONT_PROP:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(CHINESE_FONT_PROP)
        
        # 设置坐标轴标签字体
        if ax.get_xlabel(): # 检查标签是否存在
            ax.xaxis.label.set_fontproperties(CHINESE_FONT_PROP)
        if ax.get_ylabel(): # 检查标签是否存在
            ax.yaxis.label.set_fontproperties(CHINESE_FONT_PROP)
        
        # 设置标题字体
        if ax.get_title(): # 检查标题是否存在
            ax.title.set_fontproperties(CHINESE_FONT_PROP)

def apply_font_to_figure(fig):
    """
    应用中文字体到整个图表
    
    参数:
        fig: matplotlib图表对象
    """
    # 应用到所有坐标轴
    for ax in fig.get_axes():
        apply_font_to_axis(ax)
    
    # 设置图例
    for legend in fig.legends:
        apply_font_to_legend(legend)
        
    # 修复每个坐标轴上的图例
    for ax in fig.get_axes():
        if ax.get_legend():
            apply_font_to_legend(ax.get_legend())

def apply_font_to_legend(legend):
    """
    专门用于修复图例中文显示问题
    
    参数:
        legend: matplotlib图例对象
    """
    global CHINESE_FONT_PROP
    if not legend or not CHINESE_FONT_PROP:
        return
    
    # 应用字体到图例文本
    for text in legend.get_texts():
        text.set_fontproperties(CHINESE_FONT_PROP)
    
    # 应用字体到图例标题
    if legend.get_title():
        legend.get_title().set_fontproperties(CHINESE_FONT_PROP)

# 在模块加载时修复中文字体
CHINESE_FONT_PROP = fix_chinese_font_display()

# 使用enhanced_plot_style函数设置图表样式
enhance_plot_style()

# 在绘图函数中添加中文字体修复处理
def plot_distribution_histograms(df, save_dir):
    """
    绘制数值列的分布直方图，展示数据分布特征
    
    参数:
        df: 数据框
        save_dir: 图像保存目录
    
    返回:
        生成的图像文件名列表
    """
    print("\n绘制分布直方图...")
    
    # 确保目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 要绘制的数值列 - using standardized names
    numeric_cols = [
        'pwv', 'age', 'sbp', 'dbp', 'bmi', 'height', 'weight',
        'cfpwv_speed', 'bapwv_right_speed', 'bapwv_left_speed',
        'cfpwv_carotid_si', 'cfpwv_carotid_ri', 'hrv_index',
        'creatinine_umol_l', 'urea_mmol_l', 'crp_mg_l', 
        'ef_percent', 'abi_right_pt_index',
        'cfpwv_carotid_daix', 'cfpwv_interval_ms', 'cfpwv_distance_cm',
        'bapwv_right_distance_cm', 'bapwv_left_interval_ms',
        'abi_right_brachial_index', 'bfv_carotid_mean_speed',
        'bnp_pg_ml', 'wbc_10_9', 'hb_g_l'
    ]
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    if not available_cols:
        print("警告: 未找到可用于绘制分布图的数值列")
        return []
    
    # 存储生成的图像文件名
    figure_files = []
    
    # 定义临床参考范围 (示例，应根据具体临床指南调整)
    # 对于PWV，这个范围可能强烈依赖于年龄，这里使用一个通用示例
    clinical_reference_ranges = {
        'pwv': {'lower': 5.0, 'upper': 10.0, 'label': 'PWV 参考范围 (5-10 m/s)'}, # 示例范围
        'sbp': {'lower': 90.0, 'upper': 139.0, 'label': 'SBP 参考范围 (90-139 mmHg)'},
        'bmi': {'lower': 18.5, 'upper': 24.9, 'label': 'BMI 参考范围 (18.5-24.9 kg/m²)'}
    }
    
    # 对每个数值列绘制分布图
    for col in available_cols:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制直方图和核密度估计
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color='skyblue', 
                    edgecolor='black', alpha=0.7, label='数据分布')
        
        # 添加垂直线表示均值和中位数
        mean_val = df[col].mean()
        median_val = df[col].median()
        std_val = df[col].std()
        
        ax.axvline(mean_val, color='red', linestyle='-', alpha=0.7, 
                  label=f'均值: {mean_val:.2f} ± {std_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='--', alpha=0.7, 
                  label=f'中位数: {median_val:.2f}')
        
        # 添加正态分布曲线进行比较
        x_norm = np.linspace(df[col].dropna().min(), df[col].dropna().max(), 100)
        y_norm = stats.norm.pdf(x_norm, mean_val, std_val) * len(df[col].dropna()) * (df[col].dropna().max() - df[col].dropna().min()) / 30 # Adjusted scaling
        ax.plot(x_norm, y_norm, 'purple', linestyle='-.', alpha=0.6, label='正态分布拟合')

        # 添加临床参考范围 (如果定义了)
        legend_items = [
            ('数据分布', 'skyblue', None), # color, marker
            (f'均值: {mean_val:.2f} ± {std_val:.2f}', 'red', None),
            (f'中位数: {median_val:.2f}', 'green', None),
            ('正态分布拟合', 'purple', None)
        ]

        if col in clinical_reference_ranges:
            ref_range = clinical_reference_ranges[col]
            ax.axvspan(ref_range['lower'], ref_range['upper'], color='lightgrey', alpha=0.4, zorder=0, label=ref_range['label'])
            # Add to custom legend items
            legend_items.append((ref_range['label'], 'lightgrey', None))

        # 设置图形标题和标签
        column_name_map = {
            'pwv': 'PWV值 (m/s)',
            'age': '年龄 (岁)',
            'sbp': '收缩压 (mmHg)',
            'dbp': '舒张压 (mmHg)',
            'bmi': 'BMI值 (kg/m²)',
            'height': '身高 (cm)',
            'weight': '体重 (kg)',
            'cfpwv_speed': 'cfPWV 速度 (m/s)',
            'bapwv_right_speed': '右侧 baPWV 速度 (m/s)',
            'bapwv_left_speed': '左侧 baPWV 速度 (m/s)',
            'cfpwv_carotid_si': '颈动脉 SI',
            'cfpwv_carotid_ri': '颈动脉 RI',
            'hrv_index': '心率变异性指数',
            'creatinine_umol_l': '肌酐 (umol/L)',
            'urea_mmol_l': '尿素 (mmol/L)',
            'crp_mg_l': 'CRP (mg/L)',
            'ef_percent': '射血分数 (%)',
            'abi_right_pt_index': '右侧胫后 ABI',
            'cfpwv_carotid_daix': '颈动脉 DAIX',
            'cfpwv_interval_ms': 'cfPWV 时间间隔 (ms)',
            'cfpwv_distance_cm': 'cfPWV 距离 (cm)',
            'bapwv_right_distance_cm': '右 baPWV 距离 (cm)',
            'bapwv_left_interval_ms': '左 baPWV 时间间隔 (ms)',
            'abi_right_brachial_index': '右肱踝 ABI',
            'bfv_carotid_mean_speed': '颈动脉平均血流速度 (m/s)',
            'bnp_pg_ml': 'BNP (pg/ml)',
            'wbc_10_9': '白细胞计数 (10^9/L)',
            'hb_g_l': '血红蛋白 (g/L)'
        }
        
        display_name = column_name_map.get(col, col)
        
        # 使用字体属性确保中文显示
        ax.set_title(f'{display_name}分布图', fontsize=16)
        ax.set_xlabel(display_name, fontsize=14)
        ax.set_ylabel('频数', fontsize=14)
        
        # 重要！应用字体到坐标轴，修复中文方块问题
        apply_font_to_axis(ax)
        
        # 添加统计信息文本框
        stats_text = f"样本数: {df[col].count()}\n"
        stats_text += f"均值: {mean_val:.2f}\n"
        stats_text += f"标准差: {std_val:.2f}\n"
        stats_text += f"中位数: {median_val:.2f}\n"
        stats_text += f"最小值: {df[col].min():.2f}\n"
        stats_text += f"最大值: {df[col].max():.2f}"
        
        # 放置在右上角并应用字体属性
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               fontproperties=CHINESE_FONT_PROP)
        
        # 创建详细的图例
        create_custom_legend(ax, legend_items, title='图例说明', 
                           loc='upper left', bbox_to_anchor=(0.01, 0.99))
        
        # 修复图例中文显示问题
        legend = ax.get_legend()
        if legend:
            apply_font_to_legend(legend)
        
        # 保存图形
        output_name = f'distribution_{col}.png'
        output_path = os.path.join(save_dir, output_name)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        figure_files.append(output_path)
        print(f"- 已保存 {col} 分布图: {output_path}")
    
    return figure_files

def create_output_dirs():
    """创建输出目录结构，包括按图表类型分类的子目录"""
    output_dirs = {
        'base': 'output',
        'image': 'output/image',
        'figures': 'output/figures',
        'tables': 'output/tables',
        'reports': 'output/reports',
    }
    
    # 按图表类型创建子目录
    visualization_subdirs = [
        'distribution',   # 分布图
        'boxplot',        # 箱线图
        'correlation',    # 相关性图
        'gender',         # 性别比较图
        'age_group',      # 年龄组分析图
        'regression',     # 回归分析图
        'other'           # 其他图表
    ]
    
    for subdir in visualization_subdirs:
        output_dirs[subdir] = f'output/figures/{subdir}'
    
    # 创建所有目录
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        
    print("✅ 输出目录已创建/确认")
    return output_dirs

def plot_boxplots(df, save_dir='output/image'):
    """
    绘制箱线图
    
    参数:
        df: 数据DataFrame
        save_dir: 图形保存目录
        
    返回:
        生成的图像文件名列表
    """
    print("\n绘制箱线图...")
    
    # 确保目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 要绘制的数值列
    numeric_cols = [
        'pwv', 'age', 'sbp', 'dbp', 'bmi', 'height', 'weight',
        'cfpwv_speed', 'bapwv_right_speed', 'bapwv_left_speed',
        'cfpwv_carotid_si', 'cfpwv_carotid_ri', 'hrv_index',
        'creatinine_umol_l', 'urea_mmol_l', 'crp_mg_l', 
        'ef_percent', 'abi_right_pt_index',
        'cfpwv_carotid_daix', 'cfpwv_interval_ms', 'cfpwv_distance_cm',
        'bapwv_right_distance_cm', 'bapwv_left_interval_ms',
        'abi_right_brachial_index', 'bfv_carotid_mean_speed',
        'bnp_pg_ml', 'wbc_10_9', 'hb_g_l'
    ]
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    if not available_cols:
        print("警告: 未找到可用于绘制箱线图的数值列")
        return []
    
    # 存储生成的图像文件名
    figure_files = []
    
    # 创建整体箱线图
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 使用更丰富的颜色方案
    colors = apply_color_scheme("default")
    
    # 绘制箱线图
    box_plot = sns.boxplot(data=df[available_cols], palette=colors[:len(available_cols)])
    
    # 添加标题和标签
    ax.set_title('主要指标箱线图', fontsize=16)
    ax.set_ylabel('值', fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    
    # 使用专门的函数修复坐标轴中文显示
    apply_font_to_axis(ax)
    
    # 为每个箱线图添加均值标记
    for i, col in enumerate(available_cols):
        mean_val = df[col].mean()
        ax.scatter(i, mean_val, marker='o', color='white', s=50, zorder=3)
        if CHINESE_FONT_PROP:
            ax.text(i, mean_val, f'均值: {mean_val:.2f}', ha='center', va='bottom',
                  fontsize=10, fontweight='bold', fontproperties=CHINESE_FONT_PROP)
        else:
            ax.text(i, mean_val, f'均值: {mean_val:.2f}', ha='center', va='bottom',
                  fontsize=10, fontweight='bold')
    
    # 添加图例说明
    legend_elements = [
        plt.Line2D([0], [0], color='k', marker='o', linestyle='None', 
                  markersize=8, label='均值'),
        plt.Line2D([0], [0], color='k', linestyle='-', 
                  label='中位数 (箱线图中线)'),
        plt.Rectangle((0, 0), 1, 1, fc="white", ec="k", lw=1, 
                     label='四分位区间 (Q1-Q3)'),
        plt.Line2D([0], [0], color='k', linestyle='-', 
                  label='数据范围 (非异常值)')
    ]
    
    # 创建图例
    legend = ax.legend(handles=legend_elements, title='图例说明', 
             loc='upper right', bbox_to_anchor=(1.0, 1.0))
    
    # 应用字体到图例
    apply_font_to_legend(legend)
    
    # 添加说明文本
    explanation = "箱线图说明:\n"
    explanation += "- 箱体表示从第一四分位数(Q1)到第三四分位数(Q3)的范围\n"
    explanation += "- 箱体中的线表示中位数\n"
    explanation += "- 白色圆点表示均值\n"
    explanation += "- 延伸线表示非异常值的数据范围\n"
    explanation += "- 超出延伸线的点表示异常值"
    
    # 放置说明文本
    text_box = plt.figtext(0.02, 0.02, explanation, wrap=True, fontsize=10,
               bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
    
    # 应用字体到文本框
    if CHINESE_FONT_PROP:
        text_box.set_fontproperties(CHINESE_FONT_PROP)
    
    # 保存图形
    file_name = 'overall_boxplot.png'
    file_path = os.path.join(save_dir, file_name)
    plt.tight_layout()
    plt.savefig(file_path, dpi=150)
    plt.close()
    
    figure_files.append(file_name)
    print(f"  已保存: {file_name}")
    
    # 对每个分组变量绘制分组箱线图
    groupby_cols = []
    
    # 添加可能的分组变量
    if 'gender' in df.columns:
        groupby_cols.append('gender')
    if '年龄组' in df.columns:
        groupby_cols.append('年龄组')
    if 'bmi分类' in df.columns:
        groupby_cols.append('bmi分类')
    if '血压状态' in df.columns:
        groupby_cols.append('血压状态')
    
    # 对于每个分组变量，绘制PWV的分组箱线图
    for group_col in groupby_cols:
        if 'pwv' in df.columns:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # 确定排序方式
            if group_col == '年龄组':
                order = sorted(df[group_col].unique(), key=lambda x: 
                              int(x.split('-')[0].replace('<', '').replace('+', '')))
            else:
                order = None
            
            # 创建箱线图
            sns.boxplot(x=group_col, y='pwv', data=df, order=order, ax=ax)
            
            # 添加每组的均值标签
            for i, (name, group) in enumerate(df.groupby(group_col)):
                mean_val = group['pwv'].mean()
                ax.text(i, mean_val, f'{mean_val:.2f}', ha='center', va='bottom', fontweight='bold')
            
            ax.set_title(f'按{group_col}分组的PWV箱线图', fontsize=15)
            ax.set_xlabel(group_col)
            ax.set_ylabel('PWV (m/s)')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            
            # 应用字体到坐标轴
            apply_font_to_axis(ax)
            
            # 如果有图例，修复图例字体
            if ax.get_legend():
                apply_font_to_legend(ax.get_legend())
            
            # 保存图形
            file_name = f'pwv_by_{group_col}_boxplot.png'
            file_path = os.path.join(save_dir, file_name)
            plt.tight_layout()
            plt.savefig(file_path)
            plt.close()
            
            figure_files.append(file_name)
            print(f"  已保存: {file_name}")
    
    return figure_files

def plot_correlation_heatmap(df, save_dir='output/image'):
    """
    绘制相关性热力图
    
    参数:
        df: 数据DataFrame
        save_dir: 图形保存目录
        
    返回:
        生成的图像文件名列表
    """
    print("\n绘制相关性热力图...")
    
    # 确保目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 选择数值列进行相关性分析
    numeric_cols = [
        'pwv', 'age', 'sbp', 'dbp', 'bmi', 'height', 'weight',
        'cfpwv_speed', 'bapwv_right_speed', 'bapwv_left_speed',
        'cfpwv_carotid_si', 'cfpwv_carotid_ri', 'hrv_index',
        'creatinine_umol_l', 'urea_mmol_l', 'crp_mg_l', 
        'ef_percent', 'abi_right_pt_index',
        'cfpwv_carotid_daix', 'cfpwv_interval_ms', 'cfpwv_distance_cm',
        'bapwv_right_distance_cm', 'bapwv_left_interval_ms',
        'abi_right_brachial_index', 'bfv_carotid_mean_speed',
        'bnp_pg_ml', 'wbc_10_9', 'hb_g_l'
    ]
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    if len(available_cols) < 2:
        print("警告: 数值列不足，无法绘制相关性热力图")
        return []
    
    # 存储生成的图像文件名
    figure_files = []
    
    # 计算相关系数矩阵
    corr_matrix = df[available_cols].corr()
    
    # 绘制热力图
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    heatmap_ax = sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        vmax=1,
        vmin=-1,
        center=0,
        square=True,
        linewidths=.5,
        annot=True,
        fmt='.2f',
        cbar_kws={"shrink": .8},
        ax=ax
    )
    
    ax.set_title('相关性热力图', fontsize=16)
    
    # 应用字体到坐标轴
    apply_font_to_axis(ax)
    
    # 获取colorbar对象并设置标签
    cbar = heatmap_ax.collections[0].colorbar
    cbar.set_label("相关系数", fontproperties=(CHINESE_FONT_PROP if CHINESE_FONT_PROP else None))
    if CHINESE_FONT_PROP:
        for t_label in cbar.ax.get_yticklabels():
            t_label.set_fontproperties(CHINESE_FONT_PROP)

    # 添加热力图说明文本
    correlation_explanation = (
        "相关系数说明:\n"
        "• 1.0: 完全正相关\n"
        "• 0.7-0.9: 强正相关\n"
        "• 0.4-0.6: 中等正相关\n"
        "• 0.1-0.3: 弱正相关\n"
        "• 0: 无相关性\n"
        "• -0.1至-0.3: 弱负相关\n"
        "• -0.4至-0.6: 中等负相关\n"
        "• -0.7至-0.9: 强负相关\n"
        "• -1.0: 完全负相关"
    )
    
    # 在图表下方添加说明
    text_box = plt.figtext(0.5, 0.01, correlation_explanation, ha="center", fontsize=10, 
               bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
    
    # 应用字体到文本框
    if CHINESE_FONT_PROP:
        text_box.set_fontproperties(CHINESE_FONT_PROP)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # 为底部说明文本留出空间
    
    # 保存图形
    file_name = 'correlation_heatmap.png'
    file_path = os.path.join(save_dir, file_name)
    plt.savefig(file_path, dpi=150)
    plt.close()
    
    figure_files.append(file_name)
    print(f"  已保存: {file_name}")
    
    # 绘制聚类热力图
    plt.figure(figsize=(12, 10))
    cluster_map = sns.clustermap(
        corr_matrix,
        cmap=cmap,
        vmax=1,
        vmin=-1,
        center=0,
        square=True,
        linewidths=.5,
        annot=True,
        fmt='.2f',
        cbar_kws={"shrink": .8}
    )
    
    # Attempt to apply font to clustermap's heatmap axes if possible
    apply_font_to_axis(cluster_map.ax_heatmap)
    if CHINESE_FONT_PROP:
        # Clustermap has its own figure, apply to its title if it exists
        # cluster_map.fig.suptitle("聚类热力图", fontproperties=CHINESE_FONT_PROP)
        # Setting title directly on ax_heatmap is usually better
        cluster_map.ax_heatmap.set_title("聚类热力图", fontproperties=CHINESE_FONT_PROP if CHINESE_FONT_PROP else None)

        # Attempt to set colorbar label for clustermap
        if hasattr(cluster_map, 'cax'):
            try:
                cluster_map.cax.set_ylabel("相关系数", fontproperties=CHINESE_FONT_PROP)
                for t_label in cluster_map.cax.get_yticklabels():
                    t_label.set_fontproperties(CHINESE_FONT_PROP)
            except Exception as e:
                logger.warning(f"Failed to set font for clustermap colorbar: {e}")

    # 保存图形
    file_name = 'correlation_clustermap.png'
    file_path = os.path.join(save_dir, file_name)
    plt.savefig(file_path)
    plt.close()
    
    figure_files.append(file_name)
    print(f"  已保存: {file_name}")
    
    return figure_files

def plot_gender_comparison(df, save_dir='output/image'):
    """
    绘制性别差异图
    
    参数:
        df: 数据DataFrame
        save_dir: 图形保存目录
        
    返回:
        生成的图像文件名列表
    """
    print("\n绘制性别差异图...")
    
    # 确保目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 检查是否存在性别列
    if 'gender' not in df.columns:
        print("警告: 数据中没有性别列")
        return []
    
    # 数值列
    numeric_cols = [
        'pwv', 'age', 'sbp', 'dbp', 'bmi', 'height', 'weight',
        'cfpwv_speed', 'bapwv_right_speed', 'bapwv_left_speed',
        'cfpwv_carotid_si', 'cfpwv_carotid_ri', 'hrv_index',
        'creatinine_umol_l', 'urea_mmol_l', 'crp_mg_l', 
        'ef_percent', 'abi_right_pt_index',
        'cfpwv_carotid_daix', 'cfpwv_interval_ms', 'cfpwv_distance_cm',
        'bapwv_right_distance_cm', 'bapwv_left_interval_ms',
        'abi_right_brachial_index', 'bfv_carotid_mean_speed',
        'bnp_pg_ml', 'wbc_10_9', 'hb_g_l'
    ]
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    if not available_cols:
        print("警告: 未找到可用于性别差异分析的数值列")
        return []
    
    # 存储生成的图像文件名
    figure_files = []
    
    # 创建性别标签映射
    gender_labels = {0: '女性', 1: '男性'}
    df_plot = df.copy()
    if 'gender' in df_plot.columns:
        df_plot['性别'] = df_plot['gender'].map(gender_labels)
    
    # 对每个数值列绘制性别差异图
    for col in available_cols:
        # 创建小提琴图+箱线图
        plt.figure(figsize=(10, 6))
        ax = sns.violinplot(x='性别', y=col, data=df_plot, inner='box')
        
        # 添加均值点和标签
        for i, (gender, group) in enumerate(df_plot.groupby('性别')):
            mean_val = group[col].mean()
            plt.scatter(i, mean_val, color='white', s=50, zorder=3)
            plt.text(i, mean_val, f'{mean_val:.2f}', ha='center', va='bottom',
                     color='black', fontweight='bold')
        
        ax.set_title(f'{col}的性别差异', fontsize=15)
        ax.set_ylabel(col)
        
        # 应用字体到坐标轴
        apply_font_to_axis(ax)
        
        # 计算并添加p值
        male_data = df_plot[df_plot['性别'] == '男性'][col].dropna()
        female_data = df_plot[df_plot['性别'] == '女性'][col].dropna()
        
        if len(male_data) >= 5 and len(female_data) >= 5:
            try:
                t_stat, p_value = stats.ttest_ind(male_data, female_data, equal_var=False)
                plt.text(0.5, max(male_data.max(), female_data.max()) * 1.05, 
                         f'p-value: {p_value:.4f}' + (' *' if p_value < 0.05 else ''),
                         ha='center', fontweight='bold')
            except:
                pass
        
        # 保存图形
        file_name = f'{col}_gender_comparison.png'
        file_path = os.path.join(save_dir, file_name)
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()
        
        figure_files.append(file_name)
        print(f"  已保存: {file_name}")
    
    return figure_files

def plot_age_group_analysis(df, save_dir='output/image'):
    """
    绘制年龄组分析图
    
    参数:
        df: 数据DataFrame
        save_dir: 图形保存目录
        
    返回:
        生成的图像文件名列表
    """
    print("\n绘制年龄组分析图...")
    
    # 确保目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 检查是否存在年龄组列
    if '年龄组' not in df.columns:
        if 'age' in df.columns:
            print("注意: 未找到'年龄组'列，尝试创建...")
            # 创建年龄组
            age_bins = [0, 30, 40, 50, 60, 70, 120]
            age_labels = ['<30', '30-39', '40-49', '50-59', '60-69', '70+']
            df = df.copy()
            df['年龄组'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)
        else:
            print("警告: 数据中没有年龄相关列")
            return []
    
    # 数值列
    numeric_cols = [
        'pwv', 'age', 'sbp', 'dbp', 'bmi', 'height', 'weight',
        'cfpwv_speed', 'bapwv_right_speed', 'bapwv_left_speed',
        'cfpwv_carotid_si', 'cfpwv_carotid_ri', 'hrv_index',
        'creatinine_umol_l', 'urea_mmol_l', 'crp_mg_l', 
        'ef_percent', 'abi_right_pt_index',
        'cfpwv_carotid_daix', 'cfpwv_interval_ms', 'cfpwv_distance_cm',
        'bapwv_right_distance_cm', 'bapwv_left_interval_ms',
        'abi_right_brachial_index', 'bfv_carotid_mean_speed',
        'bnp_pg_ml', 'wbc_10_9', 'hb_g_l'
    ]
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    if not available_cols:
        print("警告: 未找到可用于年龄组分析的数值列")
        return []
    
    # 存储生成的图像文件名
    figure_files = []
    
    # 只为PWV绘制详细的年龄组分析图
    if 'pwv' in available_cols:
        # 创建箱线图
        plt.figure(figsize=(12, 7))
        order = sorted(df['年龄组'].unique(), key=lambda x: 
                      int(x.split('-')[0].replace('<', '').replace('+', '')))
        
        ax = sns.boxplot(x='年龄组', y='pwv', data=df, order=order)
        
        # 应用字体到坐标轴
        apply_font_to_axis(ax)
        
        # 添加每组的均值标签
        for i, (age_group, group) in enumerate(df.groupby('年龄组')):
            if age_group in order:  # 确保按正确的顺序添加标签
                idx = order.index(age_group)
                mean_val = group['pwv'].mean()
                ax.text(idx, mean_val, f'{mean_val:.2f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('PWV随年龄组的变化', fontsize=15)
        ax.set_xlabel('年龄组')
        ax.set_ylabel('PWV (m/s)')
        
        # 保存图形
        file_name = 'pwv_by_age_group_boxplot.png'
        file_path = os.path.join(save_dir, file_name)
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()
        
        figure_files.append(file_name)
        print(f"  已保存: {file_name}")
        
        # 创建折线图
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # 计算每个年龄组的平均值和标准误差
        age_group_stats = df.groupby('年龄组')['pwv'].agg(['mean', 'std', 'count'])
        age_group_stats['se'] = age_group_stats['std'] / np.sqrt(age_group_stats['count'])
        
        # 重新排序
        age_group_stats = age_group_stats.reindex(order)
        
        # 绘制折线图
        ax.errorbar(
            range(len(order)),
            age_group_stats['mean'],
            yerr=age_group_stats['se'],
            marker='o',
            ms=8,
            capsize=5,
            elinewidth=2,
            linewidth=2
        )
        
        # 添加数据标签
        for i, (mean, se) in enumerate(zip(age_group_stats['mean'], age_group_stats['se'])):
            ax.text(i, mean + se + 0.1, f'{mean:.2f}±{se:.2f}', ha='center', fontweight='bold')
        
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order)
        ax.set_title('PWV随年龄组的变化趋势', fontsize=15)
        ax.set_xlabel('年龄组')
        ax.set_ylabel('PWV (m/s) ± SE')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 应用字体到坐标轴
        apply_font_to_axis(ax)
        
        # 保存图形
        file_name = 'pwv_by_age_group_trend.png'
        file_path = os.path.join(save_dir, file_name)
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()
        
        figure_files.append(file_name)
        print(f"  已保存: {file_name}")
    
    # 创建年龄组分布图
    fig, ax = plt.subplots(figsize=(10, 6))
    age_group_counts = df['年龄组'].value_counts().reindex(order)
    age_group_pct = df['年龄组'].value_counts(normalize=True).reindex(order) * 100
    
    sns.barplot(x=age_group_counts.index, y=age_group_counts.values, ax=ax)
    
    # 添加百分比标签
    for i, (count, pct) in enumerate(zip(age_group_counts, age_group_pct)):
        ax.text(i, count, f'{count}\n({pct:.1f}%)', ha='center', va='bottom')
    
    ax.set_title('各年龄组样本分布', fontsize=15)
    ax.set_xlabel('年龄组')
    ax.set_ylabel('样本数')
    
    # 应用字体到坐标轴
    apply_font_to_axis(ax)
    
    # 保存图形
    file_name = 'age_group_distribution.png'
    file_path = os.path.join(save_dir, file_name)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
    
    figure_files.append(file_name)
    print(f"  已保存: {file_name}")
    
    return figure_files

def plot_pwv_regression(df, save_dir='output/image'):
    """
    绘制PWV回归分析图
    
    参数:
        df: 数据DataFrame
        save_dir: 图形保存目录
        
    返回:
        生成的图像文件名列表
    """
    print("\n绘制PWV回归分析图...")
    
    # 确保目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 检查是否存在PWV列
    if 'pwv' not in df.columns:
        print("警告: 数据中没有pwv列")
        return []
    
    # 选择可能的预测变量 - standardized names
    predictor_cols = ['age', 'sbp', 'dbp', 'bmi', 'hrv_index', 'crp_mg_l', 
                      'bnp_pg_ml', 'wbc_10_9', 'hb_g_l', 'ef_percent', 
                      'creatinine_umol_l', 'abi_right_brachial_index',
                      # Adding a few more from the newly added comprehensive list that might be relevant
                      'cfpwv_carotid_si', 'cfpwv_carotid_ri', 'cfpwv_carotid_daix',
                      'bfv_carotid_mean_speed'
                      ] 
    available_predictors = [col for col in predictor_cols if col in df.columns and col != 'pwv']
    
    if not available_predictors:
        print("警告: 未找到可用于回归分析的预测变量")
        return []
    
    # 存储生成的图像文件名
    figure_files = []
    
    # Define a display name map for predictors as well for plot titles/labels
    predictor_display_name_map = {
        'age': '年龄',
        'sbp': '收缩压',
        'dbp': '舒张压',
        'bmi': 'BMI',
        'hrv_index': '心率变异性指数',
        'crp_mg_l': 'CRP',
        # New additions for this round in predictors
        'bnp_pg_ml': 'BNP',
        'wbc_10_9': '白细胞计数',
        'hb_g_l': '血红蛋白',
        'ef_percent': '射血分数',
        'creatinine_umol_l': '肌酐',
        'abi_right_brachial_index': '右肱踝ABI',
        # Adding display names for newly added predictors
        'cfpwv_carotid_si': '颈动脉SI',
        'cfpwv_carotid_ri': '颈动脉RI',
        'cfpwv_carotid_daix': '颈动脉DAIX',
        'bfv_carotid_mean_speed': '颈动脉平均血流速度'
        # Add others if needed
    }

    # 对每个预测变量绘制散点图和回归线
    for col in available_predictors:
        # 准备数据
        data = df[[col, 'pwv']].dropna()
        
        if len(data) < 10:
            print(f"警告: {col} 和 pwv 的有效数据点太少，跳过")
            continue
        
        # 计算回归线
        slope, intercept, r_value, p_value, std_err = stats.linregress(data[col], data['pwv'])
        
        # 创建散点图和回归线
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 散点图
        sns.regplot(
            x=col,
            y='pwv',
            data=data,
            ax=ax,
            scatter_kws={'alpha': 0.5},
            line_kws={'color': 'red'}
        )
        
        # 添加回归方程和R²
        equation = f'PWV = {intercept:.2f} + {slope:.4f} × {col}'
        r_squared = f'R² = {r_value**2:.4f}'
        p_value_text = f'p-value = {p_value:.4f}' + (' *' if p_value < 0.05 else '')
        
        text_box = ax.text(
            0.05, 0.95,
            f'{equation}\n{r_squared}\n{p_value_text}',
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.7),
            verticalalignment='top'
        )
        
        # 应用字体到文本框
        if CHINESE_FONT_PROP:
            text_box.set_fontproperties(CHINESE_FONT_PROP)
        
        # Use display name for title and xlabel
        col_display_name = predictor_display_name_map.get(col, col)
        ax.set_title(f'PWV与{col_display_name}的关系', fontsize=15)
        ax.set_xlabel(col_display_name)
        ax.set_ylabel('PWV (m/s)')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 应用字体到坐标轴
        apply_font_to_axis(ax)
        
        # 保存图形
        file_name = f'pwv_{col}_regression.png'
        file_path = os.path.join(save_dir, file_name)
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()
        
        figure_files.append(file_name)
        print(f"  已保存: {file_name}")
    
    # 绘制PWV与年龄的更详细回归分析
    if 'age' in available_predictors and 'gender' in df.columns:
        # 按性别分组的回归分析
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 男性和女性数据
        male_data = df[df['gender'] == 1][['age', 'pwv']].dropna()
        female_data = df[df['gender'] == 0][['age', 'pwv']].dropna()
        
        # 绘制散点图
        ax.scatter(male_data['age'], male_data['pwv'], color='blue', alpha=0.6, label='男性')
        ax.scatter(female_data['age'], female_data['pwv'], color='red', alpha=0.6, label='女性')
        
        # 计算并绘制回归线 - 男性
        if len(male_data) >= 10:
            slope_m, intercept_m, r_value_m, p_value_m, _ = stats.linregress(male_data['age'], male_data['pwv'])
            x_m = np.array([male_data['age'].min(), male_data['age'].max()])
            y_m = intercept_m + slope_m * x_m
            ax.plot(x_m, y_m, 'b-', label=f'男性: PWV = {intercept_m:.2f} + {slope_m:.4f}×年龄 (R²={r_value_m**2:.2f})')
        
        # 计算并绘制回归线 - 女性
        if len(female_data) >= 10:
            slope_f, intercept_f, r_value_f, p_value_f, _ = stats.linregress(female_data['age'], female_data['pwv'])
            x_f = np.array([female_data['age'].min(), female_data['age'].max()])
            y_f = intercept_f + slope_f * x_f
            ax.plot(x_f, y_f, 'r-', label=f'女性: PWV = {intercept_f:.2f} + {slope_f:.4f}×年龄 (R²={r_value_f**2:.2f})')
        
        ax.set_title('PWV与年龄的关系（按性别分组）', fontsize=15)
        ax.set_xlabel('年龄')
        ax.set_ylabel('PWV (m/s)')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 创建图例并应用字体
        legend = ax.legend()
        if CHINESE_FONT_PROP:
            for text in legend.get_texts():
                text.set_fontproperties(CHINESE_FONT_PROP)
        
        # 应用字体到坐标轴
        apply_font_to_axis(ax)
        
        # 保存图形
        file_name = 'pwv_age_gender_regression.png'
        file_path = os.path.join(save_dir, file_name)
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()
        
        figure_files.append(file_name)
        print(f"  已保存: {file_name}")
    
    return figure_files

def plot_all_visualizations(df, analysis_results=None, save_dir='output/image'):
    """
    执行所有可视化函数
    
    参数:
        df: 数据DataFrame
        analysis_results: 分析结果字典
        save_dir: 图形保存目录
        
    返回:
        生成的所有图像文件名列表
    """
    print("\n======== 开始生成数据可视化 ========")
    
    # 检查输入数据
    if df is None or df.empty:
        print("错误: 无有效数据可视化")
        return []
    
    # 创建目录结构
    output_dirs = create_output_dirs()
    
    # 存储所有生成的图像文件名
    all_figures = []
    
    # 1. 绘制分布直方图
    distribution_dir = output_dirs['distribution']
    histograms = plot_distribution_histograms(df, save_dir=distribution_dir)
    all_figures.extend(histograms)
    
    # 2. 绘制箱线图
    boxplot_dir = output_dirs['boxplot']
    boxplots = plot_boxplots(df, save_dir=boxplot_dir)
    all_figures.extend(boxplots)
    
    # 3. 绘制相关性热力图
    correlation_dir = output_dirs['correlation']
    heatmaps = plot_correlation_heatmap(df, save_dir=correlation_dir)
    all_figures.extend(heatmaps)
    
    # 4. 绘制性别差异图
    gender_dir = output_dirs['gender']
    gender_plots = plot_gender_comparison(df, save_dir=gender_dir)
    all_figures.extend(gender_plots)
    
    # 5. 绘制年龄组分析图
    age_group_dir = output_dirs['age_group']
    age_group_plots = plot_age_group_analysis(df, save_dir=age_group_dir)
    all_figures.extend(age_group_plots)
    
    # 6. 绘制PWV回归分析图
    regression_dir = output_dirs['regression']
    regression_plots = plot_pwv_regression(df, save_dir=regression_dir)
    all_figures.extend(regression_plots)
    
    print(f"\n======== 数据可视化完成，共生成 {len(all_figures)} 个图表 ========")
    print(f"图表已按类型保存在 output/figures 各子目录中")
    
    return all_figures

if __name__ == "__main__":
    # 如果作为独立脚本运行，从data_processing导入load_and_prepare_data
    from data_processing import load_and_prepare_data
    from data_analysis import analyze_pwv_data
    
    # 创建输出目录
    create_output_dirs()
    
    # 加载和预处理数据
    df = load_and_prepare_data()
    
    if df is not None:
        # 分析数据
        analysis_results = analyze_pwv_data(df)
        
        # 生成可视化
        output_dir = os.path.join("output", "image")
        figure_files = plot_all_visualizations(df, analysis_results, output_dir)
        
        print(f"\n已生成 {len(figure_files)} 个图表，保存在 {output_dir} 目录中") 