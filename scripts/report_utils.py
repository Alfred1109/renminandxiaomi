#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility functions for the report generation process.
"""

import os
import re
import pandas as pd
import logging
import shutil
from typing import Dict, List, Optional
import warnings

logger = logging.getLogger(__name__)

def custom_slugify(text: str, separator: str = '-') -> str:
    """
    Creates a slug from a string. 
    Replaces spaces, parentheses, brackets, and other common punctuation with a separator.
    Converts to lowercase.
    Removes leading/trailing separators.
    """
    if not text: 
        return ""
    # Replace common CJK punctuation with their ASCII equivalents or spaces if no direct equivalent
    text = text.replace('（', '(').replace('）', ')')
    text = text.replace('【', '[').replace('】', ']')
    text = text.replace('：', ':').replace('，', ',').replace('。', '.')
    text = text.replace('《', '<').replace('》', '>')
    text = text.replace('、', separator) # CJK comma to separator
    
    # Handle cases like "PWV(指标)" -> "pwv-指标"
    # Replace specific patterns around parentheses with a separator
    text = re.sub(r'\s*\(\s*', separator, text) # space then (
    text = re.sub(r'\s*\)\s*', separator, text) # space then )
    text = re.sub(r'\(\s*', separator, text)    # (
    text = re.sub(r'\s*\)', separator, text)     # )
    
    # General replacements for remaining punctuation and spaces
    text = re.sub(r'[\s()（）[\]【】:,./"\'<>?!@#$%^&*+=|~`]+', separator, text)
    
    # Remove leading/trailing separators
    text = re.sub(rf'^{separator}+', '', text)
    text = re.sub(rf'{separator}+$', '', text)
    
    # Replace multiple occurrences of the separator with a single one
    text = re.sub(rf'{separator}{{2,}}', separator, text)
    
    return text.lower()

def dataframe_to_markdown_table(df: pd.DataFrame, title: str = None, floatfmt: str = ".2f", index: bool = False) -> str:
    """Converts a Pandas DataFrame to a Markdown table string with optional title."""
    if df is None or df.empty:
        return ""
    
    md_table = df.to_markdown(index=index, floatfmt=floatfmt)
    if title:
        return f"### {title}\n{md_table}\n"
    return md_table + "\n"

def create_report_directories(base_output_dir: str = "output") -> Dict[str, str]:
    """
    Creates standard report output directories.
    Returns a dictionary of created paths.
    """
    paths = {
        "base": base_output_dir,
        "reports": os.path.join(base_output_dir, "reports"),
        "tables": os.path.join(base_output_dir, "tables"),
        "figures_base": os.path.join(base_output_dir, "figures"), # Main category for figures
        "image_general": os.path.join(base_output_dir, "image") # More general image folder if used
    }
    # Specific figure subdirectories (can be extended)
    figure_subdirs = ["distribution", "boxplot", "correlation", "regression", "gender", "age_group", "other"]
    for subdir_name in figure_subdirs:
        paths[f"figures_{subdir_name}"] = os.path.join(paths["figures_base"], subdir_name)
    
    # Ensure risk prediction image directories exist if that module is active
    # This could be dynamically determined by looking at actual figure paths from analysis_results later
    risk_pred_base = os.path.join(paths["image_general"], "风险预测")
    paths["risk_prediction_base"] = risk_pred_base
    # Example sub-risk dir. This should align with how risk prediction modules save images.
    paths["risk_prediction_pwv_超标风险"] = os.path.join(risk_pred_base, "PWV超标风险") 

    for key, path in paths.items():
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as e:
            logger.error(f"创建目录失败 {path}: {e}")
            # Decide if this is a critical error or can be ignored
    logger.info(f"✅ 主要输出目录已在 '{base_output_dir}' 下创建/确认。")
    return paths

def backup_old_report(report_path: str) -> None:
    """Backs up an existing report file by appending a timestamp."""
    if os.path.exists(report_path):
        try:
            base, ext = os.path.splitext(report_path)
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S%f")[:-3] # Milliseconds
            backup_path = f"{base}_backup_{timestamp}{ext}"
            shutil.copy2(report_path, backup_path)
            logger.info(f"旧报告已备份至: {backup_path}")
        except Exception as e:
            logger.warning(f"备份旧报告 {report_path} 失败: {e}")

def extract_tables_from_markdown_to_excel(markdown_file: str, excel_file: str, sheet_prefix: str = "表格") -> None:
    """
    从Markdown文件中提取表格，并将它们保存到Excel文件中。
    每个表格将被保存在单独的工作表中。

    Args:
        markdown_file: Markdown文件的路径
        excel_file: 输出Excel文件的路径
        sheet_prefix: 工作表名称的前缀，默认为"表格"
    """
    logger.info(f"从Markdown文件 '{markdown_file}' 提取表格...")
    
    if not os.path.exists(markdown_file):
        raise FileNotFoundError(f"Markdown文件不存在: {markdown_file}")
    
    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(os.path.abspath(excel_file)), exist_ok=True)
    
    # 读取markdown文件内容
    with open(markdown_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # 用于从Markdown中提取表格的正则表达式
    # 匹配标题（可选）+ 表格
    table_pattern = r'(?:###\s*([^\n]+))?\n\|(.*?\n)+?(?=\n[^|]|\Z)'
    
    # 查找所有表格
    tables = re.findall(table_pattern, md_content, re.DOTALL)
    
    if not tables:
        logger.warning(f"在Markdown文件 '{markdown_file}' 中未找到表格。")
        # 创建一个空的Excel文件以满足期望
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            pd.DataFrame().to_excel(writer, sheet_name="无表格")
        return
    
    logger.info(f"在Markdown文件中找到 {len(tables)} 个表格。")
    
    # 创建ExcelWriter实例
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        for i, (title, table_content) in enumerate(tables):
            # 提取表格内容（包括标题行）
            table_text = f"|{table_content}"
            
            # 清理标题
            title = title.strip() if title else f"{sheet_prefix}{i+1}"
            
            try:
                # 使用pandas读取markdown表格
                dfs = pd.read_html(pd.io.common.StringIO(table_text.replace("|", "\\|")), flavor='bs4')
                if dfs:
                    df = dfs[0]
                    
                    # 确保工作表名称有效（Excel限制工作表名称长度为31个字符）
                    sheet_name = custom_slugify(title)[:25] + f"_{i+1}"
                    
                    # 写入到Excel的指定工作表
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    logger.info(f"表格 '{title}' 已保存到工作表 '{sheet_name}'。")
            except Exception as e:
                logger.error(f"处理表格 '{title}' 时出错: {e}")
                # 尝试使用备用方法解析表格
                try:
                    # 手动解析表格行
                    rows = [line.strip() for line in table_text.strip().split('\n')]
                    if len(rows) > 2:  # 确保有标题行和分隔行
                        # 从第一行获取列名
                        headers = [h.strip() for h in rows[0].split('|')[1:-1]]
                        # 跳过分隔行
                        data = []
                        for row in rows[2:]:
                            cells = [c.strip() for c in row.split('|')[1:-1]]
                            if len(cells) == len(headers):
                                data.append(cells)
                        
                        df = pd.DataFrame(data, columns=headers)
                        sheet_name = custom_slugify(title)[:25] + f"_{i+1}"
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                        logger.info(f"表格 '{title}' 已使用备用方法保存到工作表 '{sheet_name}'。")
                except Exception as backup_error:
                    logger.error(f"备用方法处理表格 '{title}' 也失败: {backup_error}")
    
    logger.info(f"所有表格已保存到Excel文件: {excel_file}")

if __name__ == '__main__':
    print("Testing report_utils.py standalone...")

    # Test custom_slugify
    test_strings = [
        "Test String with Spaces",
        "PWV指标(重要)解读",
        "2.1.1 PWV指标的临床意义解读",
        "  leading and trailing spaces  ",
        "multiple---separators---here",
        "文件名带有（括号）和【方括号】以及：冒号",
        "Section: Age & BMI (Correlation)",
        "ภาพรวมข้อมูล (Data Overview)" # Example with Thai characters to see behavior
    ]
    for s in test_strings:
        print(f"Original: '{s}' -> Slug: '{custom_slugify(s)}'")

    # Test dataframe_to_markdown_table
    sample_df = pd.DataFrame({'col A': [1, 2.345], 'col B': ['text1', 'text2']})
    print("\nMarkdown table from DataFrame:")
    print(dataframe_to_markdown_table(sample_df, title="Sample Table"))
    print(dataframe_to_markdown_table(sample_df, title="Sample Table with Index", index=True, floatfmt=".1f"))

    # Test create_report_directories
    print("\nCreating report directories under 'output_test_utils'...")
    created_paths = create_report_directories(base_output_dir="output_test_utils")
    # You would typically check if these directories exist in a real test case
    print(f"Created paths: {created_paths}")
    # Clean up test directory if desired
    # import shutil
    # shutil.rmtree("output_test_utils", ignore_errors=True)
    # print("Cleaned up 'output_test_utils' directory.")
    
    # Test extract_tables_from_markdown_to_excel
    print("\nTesting table extraction from Markdown to Excel...")
    # 创建一个临时的markdown文件用于测试
    test_md_content = """# 测试报告

### 示例表格1
| 指标 | 值 |
| ---- | --- |
| 均值 | 10.5 |
| 中位数 | 9.8 |
| 标准差 | 2.3 |

正常文本段落。

### 示例表格2
| 年龄段 | 样本数 | PWV均值 | PWV范围 |
| ------ | ----- | ------- | ------- |
| <40岁 | 120 | 1050 | 900-1200 |
| 40-60岁 | 250 | 1250 | 1000-1500 |
| >60岁 | 180 | 1450 | 1200-1700 |
"""
    
    test_md_file = "output_test_utils/test_tables.md"
    test_excel_file = "output_test_utils/test_tables_extracted.xlsx"
    
    # 确保测试目录存在
    os.makedirs("output_test_utils", exist_ok=True)
    
    # 写入测试markdown文件
    with open(test_md_file, 'w', encoding='utf-8') as f:
        f.write(test_md_content)
    
    # 测试提取函数
    try:
        extract_tables_from_markdown_to_excel(test_md_file, test_excel_file)
        print(f"表格已提取到Excel文件: {test_excel_file}")
    except Exception as e:
        print(f"表格提取测试失败: {e}")
    
    print("\nStandalone test for report_utils.py finished.") 