#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
中文字体处理模块
为matplotlib图表解决中文显示问题
"""

import os
import platform
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
import warnings

# 内嵌字体文件路径
EMBEDDED_FONT_PATH = os.path.join(os.path.dirname(__file__), 'fonts', 'SourceHanSansSC-Regular.otf')

def setup_chinese_font():
    """
    设置matplotlib的中文字体
    直接使用内嵌的思源黑体字体文件
    返回一个可用于中文显示的FontProperties对象
    """
    # 检查内嵌字体文件是否存在
    if os.path.exists(EMBEDDED_FONT_PATH):
        print(f"使用内嵌中文字体: {EMBEDDED_FONT_PATH}")
        
        # 手动添加字体文件
        font_dirs = [os.path.dirname(EMBEDDED_FONT_PATH)]
        font_files = fm.findSystemFonts(fontpaths=font_dirs)
        for font_file in font_files:
            fm.fontManager.addfont(font_file)
        
        # 创建字体属性对象
        chinese_font = FontProperties(fname=EMBEDDED_FONT_PATH)
        
        # 设置matplotlib的字体参数
        plt.rcParams['font.sans-serif'] = ['Source Han Sans SC', 'DejaVu Sans', 'Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
        return chinese_font
    else:
        print(f"警告: 内嵌字体文件未找到: {EMBEDDED_FONT_PATH}")
        return find_system_chinese_font()

def find_system_chinese_font():
    """
    在系统中查找可用的中文字体
    根据不同操作系统查找常见的中文字体
    """
    system = platform.system()
    chinese_fonts = []
    
    print(f"在{system}系统中查找中文字体...")
    
    # 针对不同操作系统的默认中文字体路径
    if system == 'Windows':
        font_dirs = [
            os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts')
        ]
        chinese_fonts = [
            'Microsoft YaHei',  # 微软雅黑
            'SimHei',           # 黑体
            'SimSun',           # 宋体
            'FangSong',         # 仿宋
            'KaiTi',            # 楷体
            'Arial Unicode MS'  
        ]
    elif system == 'Darwin':  # macOS
        chinese_fonts = [
            'PingFang SC',      # 苹方
            'Heiti SC',         # 黑体-简
            'STHeiti',          # 华文黑体
            'Hiragino Sans GB', # 冬青黑体
            'Apple LiGothic Medium',
            'Apple LiSung Light',
            'Osaka',
            'STFangsong',       # 华文仿宋
            'STSong'            # 华文宋体
        ]
    else:  # Linux 和其他系统
        # 尝试使用WSL中Windows字体
        wsl_font_dirs = [
            '/mnt/c/Windows/Fonts'
        ]
        
        # 检查WSL中的Windows字体目录
        for wsl_font_dir in wsl_font_dirs:
            if os.path.exists(wsl_font_dir):
                print(f"找到WSL Windows字体目录: {wsl_font_dir}")
                font_dirs = [wsl_font_dir]
                # 尝试添加WSL中的Windows字体
                wsl_font_files = [
                    os.path.join(wsl_font_dir, 'simhei.ttf'),  # 黑体
                    os.path.join(wsl_font_dir, 'simsun.ttc'),  # 宋体
                    os.path.join(wsl_font_dir, 'msyh.ttc')     # 微软雅黑
                ]
                for font_file in wsl_font_files:
                    if os.path.exists(font_file):
                        print(f"找到WSL Windows字体: {font_file}")
                        try:
                            fm.fontManager.addfont(font_file)
                        except Exception as e:
                            print(f"添加字体失败: {e}")
        
        # Linux字体
        chinese_fonts = [
            'Noto Sans CJK SC', # Google Noto字体
            'Noto Sans CJK TC',
            'Noto Sans CJK JP',
            'Noto Serif CJK SC',
            'WenQuanYi Micro Hei',  # 文泉驿微米黑
            'WenQuanYi Zen Hei',    # 文泉驿正黑
            'Droid Sans Fallback',  # Android默认字体
            'AR PL UMing CN',       # 文鼎PL细上海宋
            'AR PL UKai CN',        # 文鼎PL中楷
            'AR PL New Sung',       # 文鼎PL新宋
            'Source Han Sans CN',   # 思源黑体
            'Source Han Serif CN'   # 思源宋体
        ]
    
    # 获取所有系统字体
    system_fonts = [f.name for f in fm.fontManager.ttflist]
    print(f"系统字体列表: {system_fonts[:5]}... (共{len(system_fonts)}个)")
    
    # 查找第一个可用的中文字体
    for font in chinese_fonts:
        if font in system_fonts:
            print(f"找到系统中文字体: {font}")
            # 设置matplotlib的字体参数
            plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans', 'Arial', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
            
            # 创建字体属性对象
            chinese_font = FontProperties(family=font)
            return chinese_font
    
    # 如果找不到预定义的中文字体，尝试查找包含"黑体"、"宋体"等关键词的字体
    for font in system_fonts:
        for keyword in ['黑体', '宋体', '雅黑', 'SimHei', 'SimSun', 'Hei', 'Sans', 'Song']:
            if keyword.lower() in font.lower():
                print(f"找到系统中文字体: {font} (包含关键词 {keyword})")
                plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans', 'Arial', 'sans-serif']
                plt.rcParams['axes.unicode_minus'] = False
                return FontProperties(family=font)
    
    # 最后的备选方案：使用系统默认无衬线字体
    print("未找到中文字体，使用系统默认sans-serif字体")
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    return FontProperties()

def setup_chinese_font_with_fallback():
    """
    带有备用方案的字体设置方法
    优先使用内嵌字体，如果失败再尝试系统字体
    """
    try:
        # 首先尝试使用内嵌字体
        chinese_font = setup_chinese_font()
        
        # 简单测试是否可用 - 创建一个带中文的图并关闭
        plt.figure(figsize=(1, 1))
        plt.title("测试中文")
        plt.close()
        
        print("中文字体设置成功")
        return chinese_font
    except Exception as e:
        warnings.warn(f"设置中文字体失败: {e}，尝试使用备用方法")
        
        # 备用方案2: 使用matplotlib自带字体，仅标注ASCII字符，中文可能显示为方块
        print("无法设置中文字体，图表中的中文可能显示为方块")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        return FontProperties()

# 为了便于导入，提供一个默认的setup函数
setup = setup_chinese_font_with_fallback

def apply_to_figure(fig, chinese_font):
    """
    应用中文字体到整个图表
    
    参数:
        fig: matplotlib图表对象
        chinese_font: 中文字体属性对象
    """
    if fig is None or chinese_font is None:
        return
    
    # 设置标题和轴标签
    for ax in fig.get_axes():
        title = ax.get_title()
        if title:
            ax.set_title(title, fontproperties=chinese_font)
        
        xlabel = ax.get_xlabel()
        if xlabel:
            ax.set_xlabel(xlabel, fontproperties=chinese_font)
        
        ylabel = ax.get_ylabel()
        if ylabel:
            ax.set_ylabel(ylabel, fontproperties=chinese_font)
        
        # 设置刻度标签
        for label in ax.get_xticklabels():
            label.set_fontproperties(chinese_font)
        
        for label in ax.get_yticklabels():
            label.set_fontproperties(chinese_font)
        
        # 设置图例
        legend = ax.get_legend()
        if legend:
            for text in legend.get_texts():
                text.set_fontproperties(chinese_font)

if __name__ == "__main__":
    # 测试字体设置
    chinese_font = setup()
    
    fig = plt.figure(figsize=(10, 6))
    plt.title("中文字体测试", fontproperties=chinese_font)
    plt.xlabel("横轴", fontproperties=chinese_font)
    plt.ylabel("纵轴", fontproperties=chinese_font)
    plt.plot([1, 2, 3, 4], [1, 4, 9, 16], label="测试数据")
    plt.legend(prop=chinese_font)
    
    # 应用字体到整个图表
    apply_to_figure(fig, chinese_font)
    
    # 保存测试图表
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'figures', 'other')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "chinese_font_test.png")
    plt.savefig(output_file, dpi=300)
    
    print(f"字体测试图表已保存至: {output_file}")
    # plt.show()
    
    print("字体测试完成") 