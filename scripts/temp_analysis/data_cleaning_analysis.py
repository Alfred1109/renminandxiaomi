#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PWV数据缺失情况与清洗过程分析脚本
分析数据缺失模式、清洗过程和数据质量变化
输出详细结果到Excel表格和可视化图表
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

# 加载原始数据
def load_raw_data():
    """加载原始数据，不进行任何清洗处理"""
    data_paths = [
        os.path.join(project_root, "docs", "excel", "pwv数据采集表 -去公式.xlsx"),
        os.path.join(project_root, "docs", "excel", "原始数据", "pwv数据采集表 -去公式.xlsx"),
        "docs/excel/pwv数据采集表 -去公式.xlsx"
    ]
    
    for path in data_paths:
        if os.path.exists(path):
            print(f"正在加载原始数据文件: {path}")
            raw_df = pd.read_excel(path)
            print(f"成功加载原始数据: {raw_df.shape[0]}行 x {raw_df.shape[1]}列")
            return raw_df
    
    print("未找到原始数据文件")
    return None

# 加载清洗后的数据
def load_clean_data():
    """加载清洗处理后的数据"""
    try:
        df = load_and_prepare_data()
        print(f"成功加载清洗后数据: {df.shape[0]}行 x {df.shape[1]}列")
        return df
    except Exception as e:
        print(f"加载清洗数据时出错: {e}")
        return None

# 分析缺失值情况
def analyze_missing_values(raw_df, clean_df, output_dir="output/tables"):
    """
    分析原始数据和清洗后数据的缺失值情况
    
    参数:
        raw_df: 原始数据DataFrame
        clean_df: 清洗后数据DataFrame
        output_dir: 输出目录
    """
    if raw_df is None:
        print("无原始数据可分析")
        return None
    
    # 创建输出目录
    os.makedirs(os.path.join(project_root, output_dir), exist_ok=True)
    
    # 1. 分析原始数据缺失情况
    print("\n分析原始数据缺失情况...")
    raw_missing = raw_df.isnull().sum().sort_values(ascending=False)
    raw_missing_percent = (raw_df.isnull().sum() / raw_df.isnull().count() * 100).sort_values(ascending=False)
    raw_missing_analysis = pd.concat([raw_missing, raw_missing_percent], axis=1, keys=['缺失数量', '缺失率(%)'])
    raw_missing_analysis = raw_missing_analysis[raw_missing_analysis['缺失数量'] > 0]
    
    # 2. 分析清洗后数据缺失情况
    clean_missing = None
    if clean_df is not None:
        print("\n分析清洗后数据缺失情况...")
        clean_missing = clean_df.isnull().sum().sort_values(ascending=False)
        clean_missing_percent = (clean_df.isnull().sum() / clean_df.isnull().count() * 100).sort_values(ascending=False)
        clean_missing_analysis = pd.concat([clean_missing, clean_missing_percent], axis=1, keys=['缺失数量', '缺失率(%)'])
        clean_missing_analysis = clean_missing_analysis[clean_missing_analysis['缺失数量'] > 0]
    
    # 3. 关键变量缺失情况分析
    print("\n分析关键变量的缺失情况...")
    key_variables = [
        '基础信息-年龄', '受试者-性别', '基础信息-身高', '基础信息-体重', 
        '收缩压', '舒张压', '竞品信息-脉搏波传导速度', 
        'cfPWV-速度m/s', 'baPWV-右侧-速度m/s', 'baPWV-左侧-速度m/s'
    ]
    
    # 确保所有关键变量都在DataFrame中
    available_keys = [var for var in key_variables if var in raw_df.columns]
    
    # 创建关键变量缺失表
    key_missing_data = []
    for var in available_keys:
        total = len(raw_df)
        non_null = raw_df[var].count()
        missing = total - non_null
        missing_rate = (missing / total) * 100
        
        key_missing_data.append({
            '变量名': var,
            '总记录数': total,
            '非空记录数': non_null,
            '缺失记录数': missing,
            '缺失率(%)': missing_rate
        })
    
    key_missing_df = pd.DataFrame(key_missing_data)
    
    # 4. 按变量类型分组的缺失分析
    variable_groups = {
        '基础信息类': ['基础信息-年龄', '受试者-性别', '基础信息-身高', '基础信息-体重'],
        'PWV指标类': ['竞品信息-脉搏波传导速度', 'cfPWV-速度m/s', 'baPWV-右侧-速度m/s', 'baPWV-左侧-速度m/s'],
        '血压指标类': ['收缩压', '舒张压'],
        '血流速度类': [col for col in raw_df.columns if '血流速度' in col],
        'ABI指标类': [col for col in raw_df.columns if 'ABI' in col],
        '生化指标类': ['CRP', 'PCT', 'TnI', 'CK-MB', '肌红蛋白', 'BNP', '肌酐', 'WBC', 'RBC', 'Hb', 'PLT', 'PT', 'APTT', 'D-dimer']
    }
    
    # 以更灵活的方式匹配列名
    for group_name, patterns in variable_groups.items():
        if group_name in ['血流速度类', 'ABI指标类']:
            continue  # 这两类已经用前缀匹配了
        
        matched_cols = []
        for pattern in patterns:
            # 使用部分匹配
            matched_cols.extend([col for col in raw_df.columns if pattern in col])
        
        variable_groups[group_name] = list(set(matched_cols))  # 去重
    
    # 计算每组的缺失情况
    group_missing_data = []
    for group_name, cols in variable_groups.items():
        # 过滤掉不在DataFrame中的列
        valid_cols = [col for col in cols if col in raw_df.columns]
        
        if valid_cols:
            total_missing = raw_df[valid_cols].isnull().sum().sum()
            total_cells = len(raw_df) * len(valid_cols)
            missing_rate = (total_missing / total_cells) * 100
            
            group_missing_data.append({
                '变量类型': group_name,
                '变量数量': len(valid_cols),
                '变量示例': valid_cols[0] if valid_cols else '',
                '缺失记录总数': total_missing,
                '总数据点数': total_cells,
                '平均缺失率(%)': missing_rate
            })
    
    group_missing_df = pd.DataFrame(group_missing_data)
    
    # 5. 评估缺失模式
    # 计算每种缺失类型的比例
    total_missing = raw_df.isnull().sum().sum()
    total_cells = raw_df.size
    overall_missing_rate = (total_missing / total_cells) * 100
    
    # 收集清洗过程中的数据变化
    data_changes = {
        '原始行数': len(raw_df),
        '原始列数': len(raw_df.columns),
        '原始缺失值总数': total_missing,
        '原始缺失率(%)': overall_missing_rate
    }
    
    if clean_df is not None:
        clean_total_missing = clean_df.isnull().sum().sum()
        clean_total_cells = clean_df.size
        clean_overall_missing_rate = (clean_total_missing / clean_total_cells) * 100
        
        data_changes.update({
            '清洗后行数': len(clean_df),
            '清洗后列数': len(clean_df.columns),
            '清洗后缺失值总数': clean_total_missing,
            '清洗后缺失率(%)': clean_overall_missing_rate,
            '数据保留率(%)': (len(clean_df) / len(raw_df)) * 100,
            '变量数增长率(%)': ((len(clean_df.columns) - len(raw_df.columns)) / len(raw_df.columns)) * 100,
            '缺失值减少率(%)': ((total_missing - clean_total_missing) / total_missing) * 100 if total_missing > 0 else 0
        })
    
    data_changes_df = pd.DataFrame([data_changes])
    
    # 6. 导出到Excel
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    excel_path = os.path.join(project_root, output_dir, f"数据缺失与清洗分析_{timestamp}.xlsx")
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        raw_missing_analysis.to_excel(writer, sheet_name='原始数据缺失分析', index=True)
        if clean_missing is not None:
            clean_missing_analysis.to_excel(writer, sheet_name='清洗后数据缺失分析', index=True)
        key_missing_df.to_excel(writer, sheet_name='关键变量缺失情况', index=False)
        group_missing_df.to_excel(writer, sheet_name='变量类型缺失分析', index=False)
        data_changes_df.to_excel(writer, sheet_name='数据清洗效果统计', index=False)
        
        # 自动调整列宽
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            for i, col in enumerate(worksheet.columns):
                max_width = 0
                for cell in col:
                    try:
                        if len(str(cell.value)) > max_width:
                            max_width = len(str(cell.value))
                    except:
                        pass
                adjusted_width = (max_width + 2)
                worksheet.column_dimensions[chr(65 + i)].width = adjusted_width
    
    print(f"缺失值分析结果已导出到Excel: {excel_path}")
    return {
        'raw_missing': raw_missing_analysis,
        'clean_missing': clean_missing_analysis if clean_missing is not None else None,
        'key_missing': key_missing_df,
        'group_missing': group_missing_df,
        'data_changes': data_changes_df,
        'excel_path': excel_path
    }

# 绘制缺失值可视化图表
def plot_missing_values(missing_data, output_dir="output/figures/other"):
    """
    绘制缺失值分析的可视化图表
    
    参数:
        missing_data: 缺失值分析结果字典
        output_dir: 输出目录
    """
    if missing_data is None:
        print("无缺失值数据可绘图")
        return
    
    # 创建输出目录
    os.makedirs(os.path.join(project_root, output_dir), exist_ok=True)
    
    # 1. 关键变量缺失率柱状图
    key_missing = missing_data.get('key_missing')
    if key_missing is not None and not key_missing.empty:
        plt.figure(figsize=(12, 6))
        
        # 排序数据
        key_missing_sorted = key_missing.sort_values(by='缺失率(%)', ascending=False)
        
        # 创建柱状图
        bars = plt.bar(key_missing_sorted['变量名'], key_missing_sorted['缺失率(%)'], color='skyblue')
        
        # 添加数据标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%',
                    ha='center', va='bottom', rotation=0)
        
        # 设置图表属性
        plt.title('关键变量缺失率')
        plt.xlabel('变量名')
        plt.ylabel('缺失率(%)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # 应用中文字体设置
        try:
            apply_to_figure(plt.gcf())
        except:
            pass
        
        # 保存图表
        output_path = os.path.join(project_root, output_dir, "key_variables_missing_rate.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"关键变量缺失率图已保存至: {output_path}")
        plt.close()
    
    # 2. 变量类型缺失率柱状图
    group_missing = missing_data.get('group_missing')
    if group_missing is not None and not group_missing.empty:
        plt.figure(figsize=(12, 6))
        
        # 排序数据
        group_missing_sorted = group_missing.sort_values(by='平均缺失率(%)', ascending=False)
        
        # 创建柱状图
        bars = plt.bar(group_missing_sorted['变量类型'], group_missing_sorted['平均缺失率(%)'], color='lightgreen')
        
        # 添加数据标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%',
                    ha='center', va='bottom', rotation=0)
        
        # 设置图表属性
        plt.title('不同类型变量的平均缺失率')
        plt.xlabel('变量类型')
        plt.ylabel('平均缺失率(%)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # 应用中文字体设置
        try:
            apply_to_figure(plt.gcf())
        except:
            pass
        
        # 保存图表
        output_path = os.path.join(project_root, output_dir, "variable_groups_missing_rate.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"变量类型缺失率图已保存至: {output_path}")
        plt.close()
    
    # 3. 数据清洗效果对比图
    data_changes = missing_data.get('data_changes')
    if data_changes is not None and not data_changes.empty:
        if '清洗后缺失率(%)' in data_changes.columns:
            # 创建前后对比数据
            metrics = ['缺失率(%)', '行数', '列数', '缺失值总数']
            
            for metric in metrics:
                if f'原始{metric}' in data_changes.columns and f'清洗后{metric}' in data_changes.columns:
                    plt.figure(figsize=(8, 6))
                    
                    # 提取数据
                    before = data_changes[f'原始{metric}'].values[0]
                    after = data_changes[f'清洗后{metric}'].values[0]
                    
                    # 创建柱状图
                    bars = plt.bar(['清洗前', '清洗后'], [before, after], color=['salmon', 'mediumseagreen'])
                    
                    # 添加数据标签
                    for bar in bars:
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.2f}' if metric == '缺失率(%)' else f'{int(height):,}',
                                ha='center', va='bottom')
                    
                    # 设置图表属性
                    plt.title(f'数据清洗前后{metric}对比')
                    plt.ylabel(metric)
                    plt.grid(axis='y', alpha=0.3)
                    
                    # 添加变化箭头和百分比
                    change = ((after - before) / before) * 100
                    change_text = f'变化: {change:.2f}%'
                    
                    plt.annotate(change_text, 
                                xy=(1.5, min(before, after) + abs(after - before)/2),
                                xytext=(0.5, min(before, after) + abs(after - before)/2),
                                arrowprops=dict(arrowstyle='->'),
                                ha='center')
                    
                    plt.tight_layout()
                    
                    # 应用中文字体设置
                    try:
                        apply_to_figure(plt.gcf())
                    except:
                        pass
                    
                    # 保存图表
                    metric_name = metric.replace('(%)', 'percent').replace(' ', '_')
                    output_path = os.path.join(project_root, output_dir, f"cleaning_effect_{metric_name}.png")
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    print(f"数据清洗{metric}对比图已保存至: {output_path}")
                    plt.close()
    
    # 4. 热图展示前20个高缺失率变量
    raw_missing = missing_data.get('raw_missing')
    if raw_missing is not None:
        top_missing = raw_missing.iloc[:20] if len(raw_missing) > 20 else raw_missing
        
        plt.figure(figsize=(12, 8))
        
        # 将索引转换为列以便绘图
        plot_data = top_missing.reset_index()
        plot_data.columns = ['变量名', '缺失数量', '缺失率(%)']
        
        # 绘制热图
        sns.heatmap(plot_data[['缺失率(%)']].T, annot=True, fmt='.2f', 
                    cmap='YlOrRd', linewidths=.5, 
                    xticklabels=plot_data['变量名'], 
                    yticklabels=False)
        
        plt.title('前20个高缺失率变量')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # 应用中文字体设置
        try:
            apply_to_figure(plt.gcf())
        except:
            pass
        
        # 保存图表
        output_path = os.path.join(project_root, output_dir, "top_missing_variables_heatmap.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"高缺失率变量热图已保存至: {output_path}")
        plt.close()

# 生成数据清洗报告
def generate_cleaning_report(missing_data, output_dir="output/tables"):
    """
    生成数据清洗报告
    
    参数:
        missing_data: 缺失值分析结果字典
        output_dir: 输出目录
    """
    if missing_data is None:
        print("无数据可生成清洗报告")
        return
    
    # 创建输出目录
    os.makedirs(os.path.join(project_root, output_dir), exist_ok=True)
    
    # 提取数据
    key_missing = missing_data.get('key_missing')
    group_missing = missing_data.get('group_missing')
    data_changes = missing_data.get('data_changes')
    
    # 创建报告DataFrame
    report_data = []
    
    # 1. 原始数据概况
    if data_changes is not None and not data_changes.empty:
        report_data.append({
            '章节': '1. 原始数据概况',
            '小节': '1.1 数据规模',
            '内容': f"原始数据包含{data_changes['原始行数'].values[0]}行×{data_changes['原始列数'].values[0]}列，" + 
                  f"清洗后保留{data_changes['清洗后行数'].values[0] if '清洗后行数' in data_changes else '未知'}行×" +
                  f"{data_changes['清洗后列数'].values[0] if '清洗后列数' in data_changes else '未知'}列"
        })
    
    # 2. 缺失值分析
    if key_missing is not None and not key_missing.empty:
        report_data.append({
            '章节': '2. 缺失值分析',
            '小节': '2.1 关键变量缺失情况',
            '内容': f"关键变量共{len(key_missing)}个，其中完全无缺失的有{len(key_missing[key_missing['缺失率(%)'] == 0])}个，" +
                  f"缺失率最高的变量是{key_missing.sort_values('缺失率(%)', ascending=False).iloc[0]['变量名']}，" +
                  f"缺失率为{key_missing.sort_values('缺失率(%)', ascending=False).iloc[0]['缺失率(%)']:.2f}%"
        })
    
    if group_missing is not None and not group_missing.empty:
        report_data.append({
            '章节': '2. 缺失值分析',
            '小节': '2.2 变量类型缺失情况',
            '内容': f"变量类型共{len(group_missing)}类，其中缺失率最高的是{group_missing.sort_values('平均缺失率(%)', ascending=False).iloc[0]['变量类型']}，" +
                  f"平均缺失率为{group_missing.sort_values('平均缺失率(%)', ascending=False).iloc[0]['平均缺失率(%)']:.2f}%，" +
                  f"包含{group_missing.sort_values('平均缺失率(%)', ascending=False).iloc[0]['变量数量']}个变量"
        })
    
    # 3. 数据清洗效果
    if data_changes is not None and not data_changes.empty and '清洗后缺失率(%)' in data_changes.columns:
        report_data.append({
            '章节': '3. 数据清洗效果',
            '小节': '3.1 缺失值变化',
            '内容': f"缺失值总数从{data_changes['原始缺失值总数'].values[0]:,}个减少到{data_changes['清洗后缺失值总数'].values[0]:,}个，" +
                  f"缺失率从{data_changes['原始缺失率(%)'].values[0]:.2f}%降低到{data_changes['清洗后缺失率(%)'].values[0]:.2f}%，" +
                  f"减少了{data_changes['缺失值减少率(%)'].values[0]:.2f}%"
        })
        
        report_data.append({
            '章节': '3. 数据清洗效果',
            '小节': '3.2 数据保留情况',
            '内容': f"数据行保留率为{data_changes['数据保留率(%)'].values[0]:.2f}%，" +
                  f"变量数增长率为{data_changes['变量数增长率(%)'].values[0]:.2f}%"
        })
    
    # 4. 数据质量评估
    if key_missing is not None and not key_missing.empty:
        high_quality_vars = len(key_missing[key_missing['缺失率(%)'] < 5])
        medium_quality_vars = len(key_missing[(key_missing['缺失率(%)'] >= 5) & (key_missing['缺失率(%)'] < 20)])
        low_quality_vars = len(key_missing[key_missing['缺失率(%)'] >= 20])
        
        report_data.append({
            '章节': '4. 数据质量评估',
            '小节': '4.1 变量质量分布',
            '内容': f"在关键变量中，{high_quality_vars}个变量质量高(缺失率<5%)，" +
                  f"{medium_quality_vars}个变量质量中等(缺失率5%-20%)，" +
                  f"{low_quality_vars}个变量质量较低(缺失率>20%)"
        })
    
    # 5. 建议
    if group_missing is not None and not group_missing.empty:
        high_missing_groups = group_missing[group_missing['平均缺失率(%)'] > 20].sort_values('平均缺失率(%)', ascending=False)
        
        if not high_missing_groups.empty:
            high_missing_group_names = ", ".join(high_missing_groups['变量类型'].values[:3])
            
            report_data.append({
                '章节': '5. 数据收集建议',
                '小节': '5.1 重点改进方向',
                '内容': f"建议重点改进{high_missing_group_names}等类型变量的数据收集完整性，" +
                      "可通过标准化收集流程、实时数据校验和双重录入等方式提高数据质量"
            })
    
    # 创建报告DataFrame
    report_df = pd.DataFrame(report_data)
    
    # 导出报告
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    excel_path = os.path.join(project_root, output_dir, f"数据清洗报告_{timestamp}.xlsx")
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        report_df.to_excel(writer, sheet_name='数据清洗报告', index=False)
        
        # 自动调整列宽
        worksheet = writer.sheets['数据清洗报告']
        for i, col in enumerate(report_df.columns):
            max_width = max(
                len(col),
                report_df[col].astype(str).map(len).max()
            )
            worksheet.column_dimensions[chr(65 + i)].width = max_width + 2
    
    print(f"数据清洗报告已导出到Excel: {excel_path}")
    return excel_path

def main():
    """主函数"""
    # 1. 打印程序标题
    print("="*80)
    print("PWV数据缺失情况与清洗过程分析程序")
    print("="*80)
    
    # 2. 加载原始数据
    print("\n加载原始数据...")
    raw_df = load_raw_data()
    
    if raw_df is None:
        print("无法加载原始数据，程序退出")
        return
    
    # 3. 加载清洗后数据
    print("\n加载清洗后数据...")
    clean_df = load_clean_data()
    
    # 4. 分析缺失值情况
    print("\n分析缺失值情况...")
    missing_data = analyze_missing_values(raw_df, clean_df)
    
    # 5. 绘制缺失值可视化图表
    print("\n绘制缺失值可视化图表...")
    plot_missing_values(missing_data)
    
    # 6. 生成数据清洗报告
    print("\n生成数据清洗报告...")
    report_path = generate_cleaning_report(missing_data)
    
    print("\n"+"="*80)
    print("分析完成！")
    print(f"缺失值分析结果: {missing_data['excel_path'] if missing_data else '未生成'}")
    print(f"数据清洗报告: {report_path}")
    print("="*80)

if __name__ == "__main__":
    main() 