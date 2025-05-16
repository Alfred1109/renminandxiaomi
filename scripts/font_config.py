#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
字体配置模块：配置matplotlib和seaborn的字体设置
解决中文显示问题并提供更多样式选项
"""

import os
import sys
import platform
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import subprocess
import tempfile
import shutil
import urllib.request
from pathlib import Path
import logging

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('font_config')

def find_system_chinese_fonts():
    """
    系统级搜索可用的中文字体文件
    
    返回:
        找到的中文字体文件路径列表
    """
    system = platform.system()
    chinese_font_files = []
    
    # 默认字体路径
    font_dirs = []
    
    if system == 'Windows':
        font_dirs = [
            os.path.join(os.environ['WINDIR'], 'Fonts'),
            os.path.expanduser('~/AppData/Local/Microsoft/Windows/Fonts')
        ]
        patterns = ['*sim*.ttf', '*micro*.ttf', '*msyh*.ttf', '*yahei*.ttf', '*hei*.ttf']
    elif system == 'Darwin':  # macOS
        font_dirs = [
            '/System/Library/Fonts',
            '/Library/Fonts',
            os.path.expanduser('~/Library/Fonts')
        ]
        patterns = ['*ping*.ttf', '*ping*.ttc', '*hei*.ttf', '*hei*.ttc', 
                  '*kaiti*.ttf', '*kaiti*.ttc', '*song*.ttf', '*song*.ttc']
    else:  # Linux and other systems
        font_dirs = [
            '/usr/share/fonts',
            '/usr/local/share/fonts',
            os.path.expanduser('~/.fonts'),
            '/usr/share/fonts/truetype',
            '/usr/share/fonts/opentype',
            # 附加路径，一些Linux发行版特有的字体路径
            '/usr/share/fonts/truetype/wqy',
            '/usr/share/fonts/wqy',
            '/usr/share/fonts/truetype/arphic',
            '/usr/share/fonts/noto-cjk',
            # Debian/Ubuntu特定路径
            '/usr/share/fonts/truetype/noto',
            # Arch/Manjaro特定路径 
            '/usr/share/fonts/noto-cjk',
            # CentOS/RHEL特定路径
            '/usr/share/fonts/wqy-zenhei',
            # 容器环境内常见路径
            '/usr/share/fonts/chinese',
            '/opt/fonts'
        ]
        patterns = ['*wqy*.ttf', '*wqy*.ttc', '*noto*sc*.ttf', '*noto*sc*.otf',
                  '*noto*cjk*.ttf', '*noto*cjk*.otf', '*source*han*.ttf',
                  '*source*han*.otf', '*droid*fallback*.ttf', '*arphic*.ttf',
                  '*arphic*.ttc', '*sim*.ttf', '*sim*.ttc']
    
    # 递归搜索所有字体目录
    import glob
    for font_dir in font_dirs:
        if os.path.exists(font_dir):
            for pattern in patterns:
                matches = glob.glob(os.path.join(font_dir, '**', pattern), recursive=True)
                for match in matches:
                    if match not in chinese_font_files:
                        chinese_font_files.append(match)
    
    if chinese_font_files:
        logger.info(f"找到 {len(chinese_font_files)} 个中文字体文件")
        for font in chinese_font_files[:3]:  # 只显示前三个找到的字体
            logger.info(f"- {os.path.basename(font)}")
        if len(chinese_font_files) > 3:
            logger.info(f"- ... 以及其他 {len(chinese_font_files) - 3} 个字体")
    else:
        logger.warning("未在系统中找到任何中文字体文件")
    
    return chinese_font_files

def configure_chinese_fonts():
    """
    配置matplotlib支持中文显示的字体
    
    根据不同操作系统，自动查找并配置合适的中文字体
    """
    import matplotlib
    import os
    system = platform.system()
    
    # 检查是否在WSL2环境下运行
    is_wsl = False
    if system == 'Linux':
        try:
            with open('/proc/version', 'r') as f:
                if 'microsoft' in f.read().lower():
                    is_wsl = True
                    logger.info("检测到WSL2环境")
        except:
            pass
    
    # 清理字体缓存，确保新设置生效
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
        
    # WSL2环境下的特殊处理 - 尝试访问Windows字体目录
    if is_wsl:
        logger.info("WSL2环境：尝试访问Windows字体目录")
        try:
            # 尝试多种可能的Windows字体目录映射
            windows_font_candidates = [
                '/mnt/c/Windows/Fonts',
                '/mnt/c/WINDOWS/Fonts'
            ]
            
            for windows_fonts_dir in windows_font_candidates:
                if os.path.exists(windows_fonts_dir):
                    logger.info(f"找到Windows字体目录: {windows_fonts_dir}")
                    
                    # 检查几个常见的中文字体
                    cn_font_candidates = ['simhei.ttf', 'msyh.ttc', 'simsun.ttc']
                    for font_file in cn_font_candidates:
                        font_path = os.path.join(windows_fonts_dir, font_file)
                        if os.path.exists(font_path):
                            logger.info(f"找到Windows中文字体: {font_path}")
                            
                            # 注册找到的字体
                            try:
                                font_prop = fm.FontProperties(fname=font_path)
                                plt.rcParams['font.family'] = 'sans-serif'
                                plt.rcParams['font.sans-serif'] = [font_prop.get_name(), 'sans-serif']
                                plt.rcParams['axes.unicode_minus'] = False
                                
                                # 测试字体
                                fig = plt.figure(figsize=(1, 1))
                                plt.text(0.5, 0.5, '测试中文', ha='center', va='center', fontproperties=font_prop)
                                plt.close(fig)
                                
                                logger.info(f"成功使用WSL下的Windows字体: {font_prop.get_name()}")
                                return font_prop.get_name()
                            except Exception as e:
                                logger.warning(f"注册Windows字体失败: {e}")
                    break
        except Exception as e:
            logger.warning(f"访问Windows字体目录失败: {e}")
    
    # 直接尝试设置SimHei字体 - 这是最常用的中文显示字体
    try:
        logger.info("尝试直接设置SimHei字体...")
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif'] 
        plt.rcParams['axes.unicode_minus'] = False
        
        # 测试中文显示
        fig = plt.figure(figsize=(1, 1))
        plt.text(0.5, 0.5, '测试中文', ha='center', va='center')
        plt.close(fig)
        logger.info("SimHei字体设置成功！")
        return "SimHei"
    except Exception as e:
        logger.warning(f"直接设置SimHei字体失败: {e}")
    
    # 获取已有的字体列表
    existing_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 调试信息
    logger.info(f"当前系统: {system}")
    
    # 更直接地处理Linux上的字体问题
    if system == 'Linux':
        # 优先使用更可靠的方法 - 直接使用matplotlib内置支持
        try:
            # 尝试使用matplotlib内置的中文字体支持
            logger.info("尝试使用matplotlib内置中文字体...")
            matplotlib.rcParams['font.family'] = ['sans-serif']
            matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'AR PL UMing CN', 'TW-Sung', 'WenQuanYi Zen Hei', 
                                                     'Hiragino Sans GB', 'Noto Sans CJK SC', 'Droid Sans Fallback']
            matplotlib.rcParams['axes.unicode_minus'] = False
            
            # 检查当前可用字体
            available_fonts = [f for f in matplotlib.rcParams['font.sans-serif'] if any(f == font.name for font in fm.fontManager.ttflist)]
            if available_fonts:
                logger.info(f"找到可用的内置中文字体: {', '.join(available_fonts[:3])}")
            
            # 测试中文显示
            logger.info("测试中文字体显示...")
            fig = plt.figure(figsize=(1, 1))
            plt.text(0.5, 0.5, '测试中文', ha='center', va='center')
            plt.close(fig)
            logger.info("matplotlib配置成功！")
            return "matplotlib内置中文字体"
        except Exception as e:
            logger.warning(f"使用matplotlib内置中文字体失败: {e}")
        
        # 查找系统中的中文字体文件
        logger.info("搜索系统中的中文字体文件...")
        chinese_font_files = find_system_chinese_fonts()
        
        if chinese_font_files:
            # 尝试注册找到的字体
            for font_file in chinese_font_files[:3]:  # 只尝试前三个
                try:
                    logger.info(f"尝试注册字体: {os.path.basename(font_file)}")
                    font_prop = fm.FontProperties(fname=font_file)
                    plt.rcParams['font.family'] = 'sans-serif'
                    plt.rcParams['font.sans-serif'] = [font_prop.get_name(), 'sans-serif']
                    plt.rcParams['axes.unicode_minus'] = False
                    
                    # 测试
                    fig = plt.figure(figsize=(1, 1))
                    plt.text(0.5, 0.5, '测试中文', ha='center', va='center')
                    plt.close(fig)
                    
                    logger.info(f"已成功注册并测试字体: {font_prop.get_name()}")
                    return font_prop.get_name()
                except Exception as e:
                    logger.warning(f"注册字体 {os.path.basename(font_file)} 失败: {e}")
        
        # 如果以上方法都失败，尝试下载字体
        logger.info("本地字体配置失败，尝试下载字体...")
        downloaded_font = download_and_install_font()
        if downloaded_font:
            return downloaded_font
    
    # 常规方法 - 查找系统中的中文字体
    chinese_fonts = []
    
    if system == 'Windows':
        # Windows系统常见中文字体
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'NSimSun', 'FangSong']
    elif system == 'Darwin':  # macOS
        # macOS系统常见中文字体
        chinese_fonts = ['PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'STSong', 'STFangsong']
    else:  # Linux and other systems
        # Linux系统常见中文字体
        chinese_fonts = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'Noto Sans CJK TC', 'Droid Sans Fallback', 
                        'AR PL UMing CN', 'AR PL KaitiM GB', 'SimHei']
    
    # 添加通用后备字体
    chinese_fonts.extend(['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif'])
    
    # 验证字体可用性
    available_fonts = []
    for font in chinese_fonts:
        try:
            if any(f.name == font for f in fm.fontManager.ttflist):
                available_fonts.append(font)
                logger.info(f"找到可用中文字体: {font}")
        except:
            continue
    
    if not available_fonts:
        logger.warning("未找到任何可用的中文字体，将使用系统默认字体")
        available_fonts = ['sans-serif']  # 使用默认无衬线字体作为后备
    
    # 配置matplotlib字体
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = available_fonts
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    logger.info(f"中文字体配置完成，将使用: {available_fonts[0]}")
    return available_fonts[0]

def download_and_install_font():
    """
    下载并安装中文字体
    
    返回:
        安装的字体名称或None
    """
    logger.info("开始下载并安装中文字体...")
    
    # 字体下载URL列表（按优先级排序）
    font_urls = [
        "https://raw.githubusercontent.com/StellarCN/scp_zh/master/fonts/SimHei.ttf",  # 更可靠的黑体链接
        "https://github.com/StellarCN/scp_zh/raw/master/fonts/SimHei.ttf",  # 黑体
        "https://github.com/adobe-fonts/source-han-sans/raw/release/OTF/SimplifiedChinese/SourceHanSansSC-Regular.otf",  # 思源黑体
        "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansSC-Regular.otf"  # Noto Sans SC
    ]
    
    # 创建字体目录
    font_dir = os.path.expanduser("~/.fonts")
    os.makedirs(font_dir, exist_ok=True)
    
    # 尝试下载字体
    installed_font = None
    for url in font_urls:
        try:
            font_name = os.path.basename(url)
            font_path = os.path.join(font_dir, font_name)
            
            # 如果字体已存在，跳过下载
            if os.path.exists(font_path):
                logger.info(f"字体文件已存在: {font_path}")
                installed_font = font_path
                break
            
            # 下载字体
            logger.info(f"正在下载字体: {url}")
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                with urllib.request.urlopen(url) as response:
                    shutil.copyfileobj(response, temp_file)
            
            # 移动到字体目录
            shutil.move(temp_file.name, font_path)
            logger.info(f"字体已安装到: {font_path}")
            
            # 刷新字体缓存
            try:
                subprocess.run(['fc-cache', '-f', '-v'], check=True)
                logger.info("字体缓存已刷新")
            except:
                logger.warning("无法刷新字体缓存，可能需要手动重启应用")
            
            installed_font = font_path
            break
        except Exception as e:
            logger.error(f"字体下载失败: {e}")
    
    if installed_font:
        # 注册新字体
        font_prop = register_font(installed_font)
        if font_prop:
            font_name = font_prop.get_name()
            
            # 设置为默认字体
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = [font_name, 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            
            logger.info(f"成功安装并注册字体: {font_name}")
            return font_name
    
    logger.error("所有字体下载尝试均失败")
    return None

def register_font(font_path):
    """
    注册字体文件到matplotlib
    
    参数:
        font_path: 字体文件路径
    
    返回:
        FontProperties对象或None
    """
    try:
        # 注册字体
        font_path = os.path.abspath(font_path)
        logger.info(f"注册字体: {font_path}")
        
        # 添加字体文件
        fm.fontManager.addfont(font_path)
        
        # 获取字体属性
        font_prop = fm.FontProperties(fname=font_path)
        font_name = font_prop.get_name()
        
        # 刷新字体管理器缓存
        fm._rebuild()
        
        # 验证字体是否已成功注册
        fonts_after = {f.name for f in fm.fontManager.ttflist}
        if font_name in fonts_after:
            logger.info(f"字体注册成功并已验证: {font_name}")
            
            # 立即将此字体设置为默认字体
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = [font_name, 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 测试字体是否能正常显示中文
            try:
                logger.info("测试字体中文显示能力...")
                fig = plt.figure(figsize=(1, 1))
                plt.text(0.5, 0.5, '测试中文', ha='center', va='center', fontproperties=font_prop)
                plt.close(fig)
                logger.info("字体中文显示测试通过！")
            except Exception as e:
                logger.warning(f"字体中文显示测试失败: {e}")
        else:
            logger.warning(f"字体似乎未成功注册，未在字体列表中找到: {font_name}")
            # 尝试使用字体文件的绝对路径
            logger.info("尝试使用字体文件的绝对路径...")
            plt.rcParams['font.sans-serif'] = ['sans-serif']
            return font_prop
            
        return font_prop
    except Exception as e:
        logger.error(f"字体注册失败: {e}")
        return None

def enhance_plot_style():
    """
    增强图表样式设置
    """
    plt.style.use('seaborn-v0_8')  # 使用seaborn样式
    
    # 设置图表风格
    plt.rcParams['figure.figsize'] = (10, 6)  # 默认图表大小
    plt.rcParams['figure.dpi'] = 100          # 默认DPI
    
    # 线条样式
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['lines.markersize'] = 8
    
    # 坐标轴样式
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    
    # 图例样式
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.framealpha'] = 0.8
    plt.rcParams['legend.fontsize'] = 'large'
    plt.rcParams['legend.edgecolor'] = '0.8'
    
    # 字体大小
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    
    logger.info("图表样式增强设置完成")

def configure_visualization_settings():
    """
    配置所有可视化设置
    """
    logger.info("开始全面配置可视化设置...")
    
    # 首先尝试使用本地中文字体
    primary_font = configure_chinese_fonts()
    
    # 如果无法找到合适的中文字体，尝试下载并安装
    if primary_font in ['sans-serif', None]:
        logger.info("没有找到合适的中文字体，尝试下载...")
        installed_font = download_and_install_font()
        if installed_font:
            logger.info(f"已下载并安装字体: {installed_font}")
            
            # 重新配置字体缓存
            try:
                import matplotlib
                cache_dir = matplotlib.get_cachedir()
                for cache_file in ['fontlist-v330.json', 'fontlist-v310.json', 'fontlist-v320.json', '.fontlist.json', 'fontlist-v300.json']:
                    cache_path = os.path.join(cache_dir, cache_file)
                    if os.path.exists(cache_path):
                        logger.info(f"删除字体缓存: {cache_path}")
                        os.remove(cache_path)
                fm._rebuild()
            except Exception as e:
                logger.warning(f"重置字体缓存失败: {e}")
            
            # 重新应用字体配置
            primary_font = configure_chinese_fonts()
    
    # 最后的尝试：如果仍然没有合适的中文字体，使用字体路径直接注册
    if primary_font in ['sans-serif', None]:
        logger.warning("常规字体配置方法失败，尝试使用直接路径方法...")
        try:
            system = platform.system()
            direct_font_path = None
            
            if system == 'Windows':
                # Windows常见中文字体路径
                candidates = [
                    r'C:\Windows\Fonts\simhei.ttf',  # 黑体
                    r'C:\Windows\Fonts\simsun.ttc',  # 宋体
                    r'C:\Windows\Fonts\msyh.ttc'     # 微软雅黑
                ]
            elif system == 'Darwin':  # macOS
                # macOS常见中文字体路径
                candidates = [
                    '/System/Library/Fonts/PingFang.ttc',
                    '/Library/Fonts/Arial Unicode.ttf',
                    '/System/Library/Fonts/STHeiti Light.ttc'
                ]
            else:  # Linux
                # Linux常见中文字体路径
                candidates = [
                    '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
                    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                    '/usr/share/fonts/truetype/arphic/uming.ttc'
                ]
            
            # 尝试找到并直接使用第一个存在的字体
            for font_path in candidates:
                if os.path.exists(font_path):
                    direct_font_path = font_path
                    break
            
            if direct_font_path:
                logger.info(f"直接使用字体文件: {direct_font_path}")
                font_prop = register_font(direct_font_path)
                if font_prop:
                    primary_font = font_prop.get_name()
        except Exception as e:
            logger.error(f"直接字体路径方法失败: {e}")
    
    # 应用增强的图表样式
    enhance_plot_style()
    
    # 返回配置信息
    from datetime import datetime
    config_info = {
        "primary_font": primary_font,
        "style": "enhanced",
        "timestamp": "已配置于" + datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    logger.info(f"可视化设置配置完成: {config_info}")
    return config_info

# 颜色方案
COLOR_SCHEMES = {
    "default": ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
    "blue_red": ['#08306b', '#4292c6', '#9ecae1', '#deebf7', '#fee0d2', '#fc9272', '#de2d26', '#a50f15'],
    "green_purple": ['#00441b', '#1b7837', '#5aae61', '#a6dba0', '#d9f0d3', '#e7d4e8', '#c2a5cf', '#9970ab', '#762a83', '#40004b'],
    "sequential_blue": ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b'],
    "sequential_red": ['#fff5f0', '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a', '#ef3b2c', '#cb181d', '#a50f15', '#67000d'],
    "diverging": ['#543005', '#8c510a', '#bf812d', '#dfc27d', '#f6e8c3', '#f5f5f5', '#c7eae5', '#80cdc1', '#35978f', '#01665e', '#003c30']
}

# 更多样式设置，可以在需要时使用
def apply_color_scheme(scheme_name="default"):
    """
    应用预定义的颜色方案
    
    参数:
        scheme_name: 颜色方案名称，必须是COLOR_SCHEMES中的一个
        
    返回:
        颜色列表
    """
    if scheme_name not in COLOR_SCHEMES:
        logger.warning(f"未找到颜色方案 '{scheme_name}'，将使用默认方案")
        scheme_name = "default"
    
    colors = COLOR_SCHEMES[scheme_name]
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)
    
    return colors

def create_custom_legend(ax, items, title=None, **kwargs):
    """
    创建自定义图例
    
    参数:
        ax: matplotlib轴对象
        items: 图例项列表，格式为 [(label, color, marker), ...]
        title: 图例标题
        **kwargs: 传递给ax.legend()的其他参数
        
    返回:
        legend对象
    """
    handles = []
    for label, color, marker in items:
        handles.append(plt.Line2D([], [], color=color, marker=marker, 
                                 linestyle='-' if marker is None else 'None',
                                 markersize=10, label=label))
    
    legend = ax.legend(handles=handles, title=title, **kwargs)
    if legend and title:
        legend.get_title().set_fontweight('bold')
    
    return legend

# 如果直接运行此脚本，则执行配置
if __name__ == "__main__":
    config = configure_visualization_settings()
    logger.info(f"可视化设置配置完成: {config}") 