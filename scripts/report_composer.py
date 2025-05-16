#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Composes a list of structured ReportElement objects from analysis results and figures.
This list can then be rendered into various output formats (Markdown, HTML, Word).
"""

import os
import pandas as pd
import numpy as np
import logging
import json # Added for loading static content
import glob # Added for finding JSON files
from typing import List, Dict, Any, Union

# Assuming report_elements.py is in the same directory or accessible in PYTHONPATH
try:
    from scripts.report_elements import (
        ReportElement, TitleElement, HeadingElement, ParagraphElement,
        ImageElement, TableElement, UnorderedListElement, OrderedListElement,
        RawTextElement, LineBreakElement, CodeBlockElement, TOCElement
    )
    # Now custom_slugify should be correctly imported from report_utils
    from scripts.report_utils import custom_slugify, dataframe_to_markdown_table
except ImportError:
    # Fallback for direct execution or different project structures
    from report_elements import (
        ReportElement, TitleElement, HeadingElement, ParagraphElement,
        ImageElement, TableElement, UnorderedListElement, OrderedListElement,
        RawTextElement, LineBreakElement, CodeBlockElement, TOCElement
    )
    def custom_slugify(text, separator='-'): # Basic slugify for fallback
        text = str(text).lower()
        text = re.sub(r'[\s()（）[\]【】:,./\\]+', separator, text)
        text = re.sub(rf'{separator}+$', '', text)
        text = re.sub(rf'^{separator}+', '', text)
        return text
    def dataframe_to_markdown_table(df, title=None, floatfmt=".2f"): # Basic for fallback
        if df is None: return ""
        return df.to_markdown(index=False, floatfmt=floatfmt)
    import re # Needed for fallback slugify

logger = logging.getLogger(__name__)

STATIC_CONTENT_FILE = os.path.join(os.path.dirname(__file__), "report_static_content.json")

def load_static_content(file_path: str = STATIC_CONTENT_FILE) -> Dict[str, Any]:
    """Loads static text content from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
            return content.get("report_sections", {}) # Return the "report_sections" node
    except FileNotFoundError:
        logger.error(f"Static content file not found: {file_path}")
        return {}
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from static content file: {file_path}")
        return {}

def get_static_text(static_content: Dict, key: str, part: str = 'content', default: str = "[内容未找到]") -> str:
    """Helper to get specific text part (e.g., title or content) from loaded static content."""
    return static_content.get(key, {}).get(part, default)

def compose_report_elements(
    data: pd.DataFrame, # Added: The main DataFrame
    analysis_results: Dict[str, Any],
    figures_list: List[Union[Dict[str, str], str]],
    config: Dict[str, Any],
    data_cleaning_summary_md: str = "" # Added: Markdown string for data cleaning report
) -> List[ReportElement]:
    """
    Generates a structured list of ReportElement objects representing the report content.
    
    Args:
        data: The main pandas DataFrame after processing.
        analysis_results: Dictionary containing analysis data
        figures_list: List of figures, can be either strings (paths) or dictionaries with 'path' and 'caption' keys
        config: Configuration dictionary
        data_cleaning_summary_md: Markdown string for data cleaning report
    """
    elements: List[ReportElement] = []
    static_content = load_static_content() # Load static content

    # 处理figures_list中的项，确保它们都是字典格式
    normalized_figures = []
    for fig in figures_list:
        if isinstance(fig, str):
            # 如果是字符串，转换为字典格式
            normalized_figures.append({
                'path': fig,
                'caption': os.path.splitext(os.path.basename(fig))[0].replace('_', ' ').capitalize()
            })
        else:
            # 已经是字典，保持不变
            normalized_figures.append(fig)
    
    # 使用处理后的figures_list
    figures_list = normalized_figures

    report_title_text = config.get("report_title", "数据分析综合报告")
    elements.append(TitleElement(text=report_title_text))
    elements.append(TOCElement())
    elements.append(LineBreakElement(count=2))

    # --- 0. 引言与数据总体描述 (New Section Part) ---
    elements.append(HeadingElement(level=1, text="0. 引言与数据总体描述", element_id=custom_slugify("0. 引言与数据总体描述")))
    intro_text_p1 = get_static_text(static_content, "report_introduction", "paragraph1", 
                                  "本报告旨在对提供的PWV（脉搏波传导速度）相关数据进行全面的分析。")
    elements.append(ParagraphElement(text=intro_text_p1))
    
    if data is not None and not data.empty:
        elements.append(ParagraphElement(
            text=f"经过初步的数据清洗与预处理后，本次分析基于 {data.shape[0]} 名参与者的 {data.shape[1]} 个特征（变量）进行。"
        ))
    else:
        elements.append(ParagraphElement(
            text="数据加载或预处理步骤未能成功生成有效数据集，报告内容可能不完整。"
        ))
    
    intro_text_p2 = get_static_text(static_content, "report_introduction", "paragraph2",
                                  "分析内容包括数据概览、基础统计分析、多变量关系探索、亚组分析以及可能的风险因素评估。")
    elements.append(ParagraphElement(text=intro_text_p2))
    elements.append(LineBreakElement())

    # --- 1. 数据概览 ---
    h1_1_text = "1. 数据概览"
    elements.append(HeadingElement(level=1, text=h1_1_text, element_id=custom_slugify(h1_1_text)))
    
    pwv_bg_title = get_static_text(static_content, "pwv_professional_background", "title", "脉搏波传导速度(PWV)的专业背景")
    pwv_bg_content = get_static_text(static_content, "pwv_professional_background", "content")
    elements.append(HeadingElement(level=2, text=pwv_bg_title, element_id=custom_slugify(pwv_bg_title)))
    elements.append(ParagraphElement(text=pwv_bg_content))

    h2_age_dist_text = "年龄分布"
    elements.append(HeadingElement(level=2, text=h2_age_dist_text, element_id=custom_slugify(h2_age_dist_text)))
    age_dist_fig = next((f for f in figures_list if "distribution_age" in f['path']), None)
    if age_dist_fig:
        elements.append(ImageElement(path=age_dist_fig['path'], caption=age_dist_fig.get('caption', "年龄分布图")))
        elements.append(ParagraphElement(text=get_static_text(static_content, "age_distribution_interpretation")))
    else:
        elements.append(ParagraphElement(text="[年龄分布图未找到]"))
    
    if 'basic_stats' in analysis_results and 'age' in analysis_results['basic_stats'].index:
        age_stats = analysis_results['basic_stats'].loc['age']
        age_desc = f"参与者的平均年龄为 {age_stats['mean']:.2f} ± {age_stats['std']:.2f} 岁，年龄范围从 {age_stats['min']:.0f} 岁到 {age_stats['max']:.0f} 岁。"
        elements.append(ParagraphElement(text=age_desc))

    h2_pwv_dist_text = "PWV分布"
    elements.append(HeadingElement(level=2, text=h2_pwv_dist_text, element_id=custom_slugify(h2_pwv_dist_text)))
    pwv_dist_fig = next((f for f in figures_list if "distribution_pwv" in f['path']), None)
    if pwv_dist_fig:
        elements.append(ImageElement(path=pwv_dist_fig['path'], caption=pwv_dist_fig.get('caption', "PWV分布图")))
        elements.append(ParagraphElement(text=get_static_text(static_content, "pwv_distribution_interpretation")))
    else:
        elements.append(ParagraphElement(text="[PWV分布图未找到]"))

    if 'basic_stats' in analysis_results and 'pwv' in analysis_results['basic_stats'].index:
        pwv_stats = analysis_results['basic_stats'].loc['pwv']
        pwv_desc = f"整体PWV平均值为 {pwv_stats['mean']:.2f} ± {pwv_stats['std']:.2f} m/s，中位数为 {pwv_stats['50%']:.2f} m/s。"
        elements.append(ParagraphElement(text=pwv_desc))
    elements.append(LineBreakElement())
    
    # --- 2. 基本统计分析 ---
    h1_2_text = "2. 基本统计分析"
    elements.append(HeadingElement(level=1, text=h1_2_text, element_id=custom_slugify(h1_2_text)))

    h2_2_1_text = "2.1 主要血管指标统计"
    elements.append(HeadingElement(level=2, text=h2_2_1_text, element_id=custom_slugify(h2_2_1_text)))
    if 'basic_stats' in analysis_results:
        key_indicators = ['pwv', 'cfpwv_速度', 'bapwv_右侧_速度', 'bapwv_左侧_速度', '收缩压', '舒张压', '脉压差']
        basic_stats_df = analysis_results['basic_stats']
        display_stats_df = basic_stats_df[basic_stats_df.index.isin(key_indicators)]
        if not display_stats_df.empty:
            display_stats_df = display_stats_df.reindex(key_indicators).dropna(how='all')
            display_stats_df = display_stats_df[['mean', 'std', 'min', '25%', '50%', '75%', 'max', '变异系数']]
            display_stats_df.columns = ['均值', '标准差', '最小值', '25分位数', '中位数', '75分位数', '最大值', '变异系数(%)']
            display_stats_df.index.name = "指标"
            elements.append(TableElement(title="主要血管指标统计描述", dataframe=display_stats_df.reset_index()))
        else:
            elements.append(ParagraphElement(text="[主要血管指标统计数据未找到或选择的指标不存在]"))
    else:
        elements.append(ParagraphElement(text="[基础统计数据 (basic_stats) 未在分析结果中找到]"))

    pwv_interp_title = get_static_text(static_content, "pwv_clinical_interpretation", "title", "2.1.1 PWV指标的临床意义解读")
    pwv_interp_content = get_static_text(static_content, "pwv_clinical_interpretation", "content")
    elements.append(HeadingElement(level=3, text=pwv_interp_title, element_id=custom_slugify(pwv_interp_title)))
    elements.append(ParagraphElement(text=pwv_interp_content))
    elements.append(LineBreakElement())

    h2_2_2_text = "2.2 指标整体分布"
    elements.append(HeadingElement(level=2, text=h2_2_2_text, element_id=custom_slugify(h2_2_2_text)))
    overall_boxplot_fig = next((f for f in figures_list if "overall_boxplot" in f['path']), None)
    if overall_boxplot_fig:
        elements.append(ImageElement(path=overall_boxplot_fig['path'], caption=overall_boxplot_fig.get('caption', "主要指标整体分布箱线图")))
        elements.append(ParagraphElement(text=get_static_text(static_content, "overall_boxplot_interpretation")))
    else:
        elements.append(ParagraphElement(text="[主要指标整体分布箱线图未找到]"))
    elements.append(LineBreakElement())

    # --- 2.3 数据清洗、预处理与质量评估 --- (Expanded Section)
    h2_2_3_title = get_static_text(static_content, "data_cleaning_overview", "title", "2.3 数据清洗、预处理与质量评估")
    elements.append(HeadingElement(level=2, text=h2_2_3_title, element_id=custom_slugify(h2_2_3_title)))
    elements.append(ParagraphElement(text=get_static_text(static_content, "data_cleaning_overview", "content")))
    elements.append(LineBreakElement())

    # --- 2.3.1 主要清洗与预处理步骤概述 (Existing static content) ---
    h3_2_3_1_title = get_static_text(static_content, "data_cleaning_process_details_intro", "title", "2.3.1 主要清洗与预处理步骤概述")
    elements.append(HeadingElement(level=3, text=h3_2_3_1_title, element_id=custom_slugify(h3_2_3_1_title)))
    elements.append(ParagraphElement(text=get_static_text(static_content, "data_cleaning_process_details_intro", "content")))

    # This part can still use data_preparation_summary if available from advanced_analysis or similar
    data_prep_summary = analysis_results.get('data_preparation_summary', {}) 
    initial_shape_str = data_prep_summary.get('initial_shape_str')
    final_shape_str = data_prep_summary.get('final_shape_str')
    
    if initial_shape_str and final_shape_str:
        elements.append(ParagraphElement(text=f"原始数据维度：{initial_shape_str}。经过一系列处理后，用于最终分析的数据集特征为：{final_shape_str}。"))
    elif final_shape_str: # If only final is available (e.g. from df.shape directly)
         elements.append(ParagraphElement(text=f"清洗后用于分析的数据集特征为：{final_shape_str}。"))


    cleaning_steps = data_prep_summary.get('cleaning_steps', []) # Default to empty list
    if cleaning_steps: # Only add if cleaning_steps has content
        elements.append(ParagraphElement(text="关键的清洗步骤包括："))
        elements.append(UnorderedListElement(items=cleaning_steps))
    # No "else" needed, if no steps, this section is just skipped.
    elements.append(LineBreakElement())

    # --- 2.3.2 详细数据清洗流程报告 (New section for the detailed MD report) ---
    h3_2_3_2_title = "2.3.2 详细数据清洗流程报告" # New heading
    elements.append(HeadingElement(level=3, text=h3_2_3_2_title, element_id=custom_slugify(h3_2_3_2_title)))
    if data_cleaning_summary_md and data_cleaning_summary_md.strip():
        elements.append(ParagraphElement(text="以下是由数据处理模块（`data_processing.py`）生成的详细数据清洗和预处理过程报告："))
        elements.append(RawTextElement(text=data_cleaning_summary_md))
    else:
        elements.append(ParagraphElement(text="[详细的数据清洗流程报告未提供或为空。这部分内容应由 `data_processing.py` 生成并传递至此。]"))
    elements.append(LineBreakElement())

    # --- 2.3.3 缺失数据分析与处理 (Adjusted heading level and content) ---
    h3_2_3_3_title = get_static_text(static_content, "missing_data_analysis_intro", "title", "2.3.3 补充：历史缺失数据分析与处理") # Adjusted title
    elements.append(HeadingElement(level=3, text=h3_2_3_3_title, element_id=custom_slugify(h3_2_3_3_title)))
    
    elements.append(ParagraphElement(text=get_static_text(static_content, "missing_data_analysis_intro", "content"))) # General intro

    # Reference to detailed data cleaning Excel files (legacy, might be redundant if new report is good)
    # detailed_missing_excel_path = analysis_results.get('detailed_missing_analysis_excel_path')
    # tabular_cleaning_report_excel_path = analysis_results.get('tabular_cleaning_report_excel_path')
    # Consider if this section is still needed or if the new MD report covers it.
    # For now, let's keep the logic for finding the JSON cleaning summary as it's a different artifact.
    
    # Try to find the JSON cleaning summary file (generated by data_processing.py)
    json_cleaning_summary_path_pattern = os.path.join(config.get("base_output_dir", "output"), "tables", "data_cleaning_summary_*.json")
    latest_json_summary_file = None
    try:
        json_files = glob.glob(json_cleaning_summary_path_pattern)
        if json_files:
            latest_json_summary_file = max(json_files, key=os.path.getctime)
    except Exception as e:
        logger.warning(f"Error searching for JSON cleaning summary files: {e}")

    if latest_json_summary_file:
        elements.append(ParagraphElement(
            text=f"数据清洗的详细量化指标和各列处理摘要已保存为JSON文件，可供深入审查：`{os.path.relpath(latest_json_summary_file, config.get('base_output_dir', 'output'))}`. "
                 "该文件位于 `output/tables/` 目录下。"
        ))
    else:
         elements.append(ParagraphElement(
            text="[详细的JSON格式数据清洗摘要文件未找到。该文件通常由 `data_processing.py` 生成并保存在 `output/tables/` 目录下。]"
        ))
    elements.append(LineBreakElement())
    
    # Embed top missing variables heatmap if available (legacy, check if still relevant)
    missing_heatmap_fig = next((f for f in figures_list if isinstance(f, dict) and "top_missing_variables_heatmap" in f.get('path', '').lower()), None)
    if missing_heatmap_fig:
        elements.append(ParagraphElement(text="历史分析中生成的缺失变量热图："))
        elements.append(ImageElement(path=missing_heatmap_fig['path'], caption=missing_heatmap_fig.get('caption', "主要变量缺失情况热图 (历史)"))),
        elements.append(ParagraphElement(text="上图直观展示了数据集中缺失比例较高的主要变量及其分布情况。"))
        elements.append(LineBreakElement())
    
    # Fallback for older 'data_missing_and_cleaning_details_md' if the new one isn't there (legacy)
    # This can be removed if the new data_cleaning_summary_md is always expected to be primary.
    # For now, keeping it as a conditional fallback or supplementary information.
    if 'data_missing_and_cleaning_details_md' in analysis_results and not (data_cleaning_summary_md and data_cleaning_summary_md.strip()):
        elements.append(ParagraphElement(text="补充信息：以下是由历史脚本 (`data_missing_and_cleaning_analysis.py`) 生成的关于数据清洗和预处理步骤的叙述性报告："))
        elements.append(RawTextElement(text=analysis_results['data_missing_and_cleaning_details_md']))
    elif 'missing_summary' in analysis_results and isinstance(analysis_results['missing_summary'], pd.DataFrame) and not (data_cleaning_summary_md and data_cleaning_summary_md.strip()):
        elements.append(ParagraphElement(text="补充信息：历史缺失值统计概要："))
        elements.append(TableElement(title="缺失值统计概要 (历史)", dataframe=analysis_results['missing_summary']))

    elements.append(LineBreakElement(count=2))

    # --- 3. 关联性分析 ---
    h1_3_text = "3. 关联性分析"
    elements.append(HeadingElement(level=1, text=h1_3_text, element_id=custom_slugify(h1_3_text)))

    h2_3_1_text = "3.1 PWV与年龄的关系"
    elements.append(HeadingElement(level=2, text=h2_3_1_text, element_id=custom_slugify(h2_3_1_text)))
    pwv_age_reg_fig = next((f for f in figures_list if "pwv_age_regression" in f['path']), None)
    if pwv_age_reg_fig:
        elements.append(ImageElement(path=pwv_age_reg_fig['path'], caption=pwv_age_reg_fig.get('caption', "PWV与年龄的散点回归图")))
    else:
        elements.append(ParagraphElement(text="[PWV与年龄的散点回归图未找到]"))

    if 'correlation_analysis' in analysis_results:
        corr_df = analysis_results['correlation_analysis']
        if isinstance(corr_df, pd.DataFrame) and not corr_df.empty:
            pwv_age_corr_series = corr_df[
                ((corr_df['变量1'] == 'pwv') & (corr_df['变量2'] == 'age')) |
                ((corr_df['变量1'] == 'age') & (corr_df['变量2'] == 'pwv'))
            ]
            if not pwv_age_corr_series.empty:
                r_val = pwv_age_corr_series['相关系数'].iloc[0]
                p_val = pwv_age_corr_series['P值'].iloc[0]
                corr_text = f"PWV与年龄呈显著正相关 (Pearson r = {r_val:.3f}, p = {p_val:.3g})。随着年龄的增长，PWV值有明显升高的趋势。"
                elements.append(ParagraphElement(text=corr_text))
            else:
                elements.append(ParagraphElement(text="[未找到PWV与年龄的相关性数据]"))
        else:
            elements.append(ParagraphElement(text="[相关性分析数据格式不正确或为空]"))
    
    # 3.1.1 年龄与PWV关系的深入解析 - NEWLY ADDED SECTION
    age_pwv_deep_title = get_static_text(static_content, "age_pwv_deep_analysis", "title", "3.1.1 年龄与PWV关系的深入解析")
    age_pwv_deep_content = get_static_text(static_content, "age_pwv_deep_analysis", "content")
    elements.append(HeadingElement(level=3, text=age_pwv_deep_title, element_id=custom_slugify(age_pwv_deep_title)))
    elements.append(ParagraphElement(text=age_pwv_deep_content))
    elements.append(LineBreakElement())

    # --- 3.2 PWV与血压的关系 ---
    h2_3_2_text = get_static_text(static_content, "pwv_bp_relationship", "title", "3.2 PWV与血压的关系")
    elements.append(HeadingElement(level=2, text=h2_3_2_text, element_id=custom_slugify(h2_3_2_text)))
    
    # Assuming a scatter plot for PWV vs SBP (Systolic Blood Pressure)
    pwv_sbp_reg_fig_path = os.path.join(config.get('output_figures_dir', 'output/figures'), "correlation", "pwv_sbp_regression.png")
    pwv_sbp_reg_fig = next((f for f in figures_list if "pwv_sbp_regression" in f['path']), {'path': pwv_sbp_reg_fig_path, 'caption': "PWV与收缩压的散点回归图"})
    elements.append(ImageElement(path=pwv_sbp_reg_fig['path'], caption=pwv_sbp_reg_fig.get('caption')))
    
    # Add descriptive text for PWV and SBP relationship
    if 'correlation_analysis' in analysis_results:
        corr_df = analysis_results['correlation_analysis']
        if isinstance(corr_df, pd.DataFrame) and not corr_df.empty:
            pwv_sbp_corr = corr_df[
                ((corr_df['变量1'] == 'pwv') & (corr_df['变量2'] == '收缩压')) |
                ((corr_df['变量1'] == '收缩压') & (corr_df['变量2'] == 'pwv'))
            ]
            if not pwv_sbp_corr.empty:
                r_val = pwv_sbp_corr['相关系数'].iloc[0]
                p_val = pwv_sbp_corr['P值'].iloc[0]
                sbp_corr_text = f"PWV与收缩压呈显著正相关 (Pearson r = {r_val:.3f}, p = {p_val:.3g})。收缩压的增高与PWV值的上升趋势一致。"
                elements.append(ParagraphElement(text=sbp_corr_text))
            else:
                elements.append(ParagraphElement(text="[未找到PWV与收缩压的相关性数据]"))
        else:
            elements.append(ParagraphElement(text="[相关性分析数据格式不正确或为空]"))

    # 3.2.1 PWV与血压关系机理探讨 - from static content
    pwv_bp_mechanism_title = get_static_text(static_content, "pwv_bp_mechanism", "title", "3.2.1 PWV与血压关系机理探讨")
    pwv_bp_mechanism_content = get_static_text(static_content, "pwv_bp_mechanism", "content")
    elements.append(HeadingElement(level=3, text=pwv_bp_mechanism_title, element_id=custom_slugify(pwv_bp_mechanism_title)))
    elements.append(ParagraphElement(text=pwv_bp_mechanism_content))
    elements.append(LineBreakElement())

    # --- 3.3 相关性热图 ---
    h2_3_3_text = "3.3 主要指标间相关性热图"
    elements.append(HeadingElement(level=2, text=h2_3_3_text, element_id=custom_slugify(h2_3_3_text)))
    
    corr_heatmap_fig_path = os.path.join(config.get('output_figures_dir', 'output/figures'), "correlation", "correlation_heatmap.png")
    corr_heatmap_fig = next((f for f in figures_list if "correlation_heatmap" in f['path']), {'path': corr_heatmap_fig_path, 'caption': "主要血管参数相关性热图"})
    elements.append(ImageElement(path=corr_heatmap_fig['path'], caption=corr_heatmap_fig.get('caption')))
    
    # Interpretation from static content
    heatmap_interp_content = get_static_text(static_content, "correlation_heatmap_interpretation", "content")
    elements.append(ParagraphElement(text=heatmap_interp_content))
    elements.append(LineBreakElement())

    # --- 3.4 血管健康核心指标关联深度分析 ---
    vascular_metrics_title = get_static_text(static_content, "vascular_metrics_deep_dive", "title", "3.4 血管健康核心指标关联深度分析")
    vascular_metrics_content = get_static_text(static_content, "vascular_metrics_deep_dive", "content")
    elements.append(HeadingElement(level=2, text=vascular_metrics_title, element_id=custom_slugify(vascular_metrics_title)))
    elements.append(ParagraphElement(text=vascular_metrics_content))
    # Placeholder: Could add specific correlation tables or pair plots if available
    # Example: if 'some_specific_metric_correlation_table' in analysis_results:
    # elements.append(TableElement(dataframe=analysis_results['some_specific_metric_correlation_table']))
    elements.append(LineBreakElement())

    # --- 3.5 离群值分析 ---
    outlier_analysis_title = get_static_text(static_content, "outlier_analysis", "title", "3.5 离群值分析")
    outlier_analysis_content = get_static_text(static_content, "outlier_analysis", "content")
    elements.append(HeadingElement(level=2, text=outlier_analysis_title, element_id=custom_slugify(outlier_analysis_title)))
    elements.append(ParagraphElement(text=outlier_analysis_content))
    
    pwv_outliers_boxplot_fig_path = os.path.join(config.get('output_figures_dir', 'output/figures'), "boxplot", "pwv_outliers_boxplot.png")
    pwv_outliers_boxplot_fig = next((f for f in figures_list if "pwv_outliers_boxplot" in f['path']), {'path': pwv_outliers_boxplot_fig_path, 'caption': "PWV离群值箱线图分析"})
    elements.append(ImageElement(path=pwv_outliers_boxplot_fig['path'], caption=pwv_outliers_boxplot_fig.get('caption')))
    # Placeholder: Could add summary statistics of outliers if available in analysis_results
    # Example: if 'outlier_summary_stats' in analysis_results:
    # elements.append(ParagraphElement(text=analysis_results['outlier_summary_stats_text']))
    elements.append(LineBreakElement())

    # --- 3.6 PWV内部指标比较与相关性 --- (NEW SECTION)
    h2_3_6_title_text = "3.6 PWV内部指标比较与相关性"
    elements.append(HeadingElement(level=2, text=h2_3_6_title_text, element_id=custom_slugify(h2_3_6_title_text)))
    elements.append(ParagraphElement(text=get_static_text(static_content, "pwv_internal_comparison_intro", "content", "本节分析PWV测量中的内部一致性与相关性，例如比较左右两侧baPWV的差异，以及cfPWV与baPWV之间的关联强度。")))
    elements.append(LineBreakElement())

    # Add Figures from temp_pwv_analysis.py
    # Figures are moved to output/figures/pwv_internal_comparison/ by report_generator.py
    pwv_dist_boxplot_fig = next((f for f in figures_list if isinstance(f, dict) and "pwv_distributions_boxplot" in f.get('path', '').lower()), None)
    if pwv_dist_boxplot_fig:
        elements.append(ImageElement(path=pwv_dist_boxplot_fig['path'], caption=pwv_dist_boxplot_fig.get('caption', "PWV各项指标分布箱线图")))
    else:
        elements.append(ParagraphElement(text="[PWV各项指标分布箱线图未找到]"))

    cf_ba_scatter_fig = next((f for f in figures_list if isinstance(f, dict) and "cfpwv_vs_bapwv_scatter" in f.get('path', '').lower()), None)
    if cf_ba_scatter_fig:
        elements.append(ImageElement(path=cf_ba_scatter_fig['path'], caption=cf_ba_scatter_fig.get('caption', "cfPWV与baPWV相关性散点图")))
    else:
        elements.append(ParagraphElement(text="[cfPWV与baPWV相关性散点图未找到]"))
    
    elements.append(ParagraphElement(text=get_static_text(static_content, "pwv_internal_figures_interpretation", "content", "上述图表展示了不同PWV测量值的分布情况以及它们之间的相关性模式。")))
    elements.append(LineBreakElement())

    # Add Statistical Summaries from temp_pwv_analysis.py (parsed from stdout)
    temp_pwv_stats = analysis_results.get('temp_pwv_analysis_stats', {})
    stats_summary_parts = []

    if 'bapwv_rl_diff_pval' in temp_pwv_stats:
        pval = temp_pwv_stats['bapwv_rl_diff_pval']
        significance_text = "存在显著差异" if pval < 0.05 else "未发现显著差异"
        stats_summary_parts.append(f"左右两侧baPWV比较 (Wilcoxon符号秩检验 P值: {pval:.4g})，提示两侧测量值{significance_text}。")
    else:
        stats_summary_parts.append("左右两侧baPWV差异的统计检验结果未捕获。")

    # Note: Parsing Spearman r and its p-value from temp_pwv_analysis.py stdout was marked as fragile.
    # We will just acknowledge if the key exists or not for now.
    if 'cfpwv_vs_bapwv_avg_spearman_r' in temp_pwv_stats: # This key was not actually populated in generator due to parsing difficulty
        # r_val = temp_pwv_stats['cfpwv_vs_bapwv_avg_spearman_r']
        # p_val_corr = temp_pwv_stats.get('cfpwv_vs_bapwv_avg_spearman_p', 'N/A') # Assuming p value might also be parsed
        # stats_summary_parts.append(f"cfPWV与平均baPWV之间的相关性 (Spearman r = {r_val}, p = {p_val_corr})。")
        stats_summary_parts.append("cfPWV与平均baPWV之间的Spearman相关性系数已计算，具体数值需查阅脚本原始输出或增强解析逻辑。")
    elif 'temp_pwv_analysis_stats' in analysis_results: # Check if the general stats dict exists, even if specific key is missing
        stats_summary_parts.append("cfPWV与平均baPWV相关性的具体统计数值的解析较为复杂，建议参考原始脚本输出或增强解析部分。")

    if stats_summary_parts:
        elements.append(HeadingElement(level=3, text="统计摘要", element_id=custom_slugify("pwv内部比较统计摘要")))
        for part in stats_summary_parts:
            elements.append(ParagraphElement(text=part))
    else:
        elements.append(ParagraphElement(text="[未能从temp_pwv_analysis.py的输出中捕获关键统计数据。]"))
    
    elements.append(ParagraphElement(text=get_static_text(static_content, "pwv_internal_analysis_conclusion", "content", "这些内部比较有助于评估数据质量和不同PWV指标间的一致性，为后续分析提供基础。")))
    elements.append(LineBreakElement())

    # --- 4. 深入数据探索与模式识别 ---
    h1_4_text = "4. 深入数据探索与模式识别"
    elements.append(HeadingElement(level=1, text=h1_4_text, element_id=custom_slugify(h1_4_text)))
    elements.append(ParagraphElement(text=get_static_text(static_content, "advanced_data_exploration_intro", "content", "本章节将深入探讨数据中的模式，通过亚组分析、聚类等方法识别关键特征。")))
    elements.append(LineBreakElement())

    # --- 4.1 不同年龄组的PWV对比 ---
    h2_4_1_title = get_static_text(static_content, "pwv_by_age_group", "title", "4.1 不同年龄组的PWV对比")
    elements.append(HeadingElement(level=2, text=h2_4_1_title, element_id=custom_slugify(h2_4_1_title)))
    
    # Intro paragraph from static content or default
    elements.append(ParagraphElement(text=get_static_text(static_content, "pwv_by_age_group", "intro_content", "为了更清晰地展示PWV随年龄变化的规律，我们将受试者划分为不同的年龄组，并比较各组间PWV的差异。")))

    # Find the plot - orchestrator should add plots from subgroup_analysis to figures_list
    # Example: caption might be "cfPWV by 年龄组 Boxplot" or path contains unique identifiers
    age_group_plot = next((f for f in figures_list if isinstance(f, dict) and "cfPWV" in f.get('caption', '').lower() and "年龄组" in f.get('caption', '').lower() and "boxplot" in f.get('caption', '').lower()), None)
    if not age_group_plot: # Fallback to path based search if caption matching fails
         age_group_plot = next((f for f in figures_list if isinstance(f, dict) and "cfPWV_速度m_s" in f.get('path','') and "年龄组" in f.get('path','') and "boxplot" in f.get('path','')), None)

    if age_group_plot:
        elements.append(ImageElement(path=age_group_plot['path'], caption=age_group_plot.get('caption', "不同年龄组的cfPWV分布")))
    else:
        elements.append(ParagraphElement(text="[不同年龄组cfPWV对比图表未找到]"))

    # Add statistical summary from subgroup_analysis_summary data
    age_group_stats_text_parts = []
    if 'subgroup_analysis_summary' in analysis_results:
        subgroup_summary_df = analysis_results['subgroup_analysis_summary']
        if isinstance(subgroup_summary_df, pd.DataFrame):
            # cfPWV by Age Group
            cfpwv_age_stats = subgroup_summary_df[
                (subgroup_summary_df['Target'] == 'cfPWV-速度m/s') & 
                (subgroup_summary_df['Group'] == '年龄组')
            ]
            if not cfpwv_age_stats.empty:
                p_val = cfpwv_age_stats['P_Value'].iloc[0]
                test_used = cfpwv_age_stats['TestUsed'].iloc[0]
                age_group_stats_text_parts.append(f"cfPWV在不同年龄组间的差异具有统计学意义 ({test_used}, P值 {p_val:.3g})。")

            # baPWV by Age Group
            bapwv_age_stats = subgroup_summary_df[
                (subgroup_summary_df['Target'] == 'baPWV-平均速度m/s') & # Using actual name from CSV
                (subgroup_summary_df['Group'] == '年龄组')
            ]
            if not bapwv_age_stats.empty:
                p_val = bapwv_age_stats['P_Value'].iloc[0]
                test_used = bapwv_age_stats['TestUsed'].iloc[0]
                age_group_stats_text_parts.append(f"baPWV在不同年龄组间的差异也具有统计学意义 ({test_used}, P值 {p_val:.3g})。")
    
    if age_group_stats_text_parts:
        elements.append(ParagraphElement(text=" ".join(age_group_stats_text_parts) + " 通常表现为随年龄增长PWV升高。"))
    else:
        elements.append(ParagraphElement(text=get_static_text(static_content, "pwv_by_age_group", "stats_fallback", "[年龄组PWV对比的统计显著性分析结果待补充]")))
    
    elements.append(ParagraphElement(text=get_static_text(static_content, "pwv_by_age_group", "interpretation", "关注不同年龄段的PWV参考值范围，对于早期识别高危个体具有重要意义。")))
    
    # Placeholder for descriptive stats table (this would require more data from subgroup_analysis.py)
    # For now, we point to the general availability of such data.
    # The orchestrator script would need to run subgroup_analysis.py, capture its descriptive stats,
    # and place them into analysis_results with a key like 'descriptive_stats_age_group_pwv' for this to be populated.
    if 'descriptive_stats_age_group_pwv' in analysis_results and isinstance(analysis_results['descriptive_stats_age_group_pwv'], pd.DataFrame):
        elements.append(TableElement(title="各年龄组PWV描述性统计", dataframe=analysis_results['descriptive_stats_age_group_pwv']))
    else:
        elements.append(ParagraphElement(text="[各年龄组PWV的详细描述性统计数据需查阅对应亚组分析的输出。]"))
    elements.append(LineBreakElement())

    # --- 4.2 PWV指标的性别差异研究 ---
    h2_4_2_title = get_static_text(static_content, "pwv_by_gender", "title", "4.2 PWV指标的性别差异研究")
    elements.append(HeadingElement(level=2, text=h2_4_2_title, element_id=custom_slugify(h2_4_2_title)))
    
    # Find the plot for gender
    gender_plot = next((f for f in figures_list if isinstance(f, dict) and "cfPWV" in f.get('caption', '').lower() and "性别" in f.get('caption', '').lower() and "boxplot" in f.get('caption', '').lower()), None)
    if not gender_plot: # Fallback to path based search
         gender_plot = next((f for f in figures_list if isinstance(f, dict) and "cfPWV_速度m_s" in f.get('path','') and "性别" in f.get('path','') and "boxplot" in f.get('path','')), None)

    if gender_plot:
        elements.append(ImageElement(path=gender_plot['path'], caption=gender_plot.get('caption', "不同性别的cfPWV分布")))
    else:
        # Fallback to a generic path if not found in figures_list, assuming orchestrator handles it
        generic_gender_plot_path = os.path.join(config.get('output_figures_dir', 'output/figures'), "subgroup_analysis", "gender", "boxplot_cfPWV_速度m_s_by_性别.png")
        elements.append(ImageElement(path=generic_gender_plot_path, caption="不同性别的cfPWV分布情况（默认路径）"))
        elements.append(ParagraphElement(text="[提示: 性别亚组cfPWV对比图表未在figures_list中明确指定，使用了默认路径]"))

    # Add statistical summary from subgroup_analysis_summary data for Gender
    gender_stats_text_parts = []
    if 'subgroup_analysis_summary' in analysis_results:
        subgroup_summary_df = analysis_results['subgroup_analysis_summary']
        if isinstance(subgroup_summary_df, pd.DataFrame):
            # cfPWV by Gender
            cfpwv_gender_stats = subgroup_summary_df[
                (subgroup_summary_df['Target'] == 'cfPWV-速度m/s') & 
                (subgroup_summary_df['Group'] == '性别')
            ]
            if not cfpwv_gender_stats.empty:
                p_val_cf = cfpwv_gender_stats['P_Value'].iloc[0]
                test_used_cf = cfpwv_gender_stats['TestUsed'].iloc[0]
                gender_stats_text_parts.append(f"cfPWV ({test_used_cf}, P = {p_val_cf:.3g})")

            # baPWV by Gender
            bapwv_gender_stats = subgroup_summary_df[
                (subgroup_summary_df['Target'] == 'baPWV-平均速度m/s') & 
                (subgroup_summary_df['Group'] == '性别')
            ]
            if not bapwv_gender_stats.empty:
                p_val_ba = bapwv_gender_stats['P_Value'].iloc[0]
                test_used_ba = bapwv_gender_stats['TestUsed'].iloc[0]
                gender_stats_text_parts.append(f"baPWV ({test_used_ba}, P = {p_val_ba:.3g})")
    
    if gender_stats_text_parts:
        summary_text = "统计分析显示，无论是 " + "还是 ".join(gender_stats_text_parts) + "，在本研究样本中其均值/中位数在不同性别间的差异均未达到统计学上的显著水平。"
        elements.append(ParagraphElement(text=summary_text))
    else:
        elements.append(ParagraphElement(text=get_static_text(static_content, "pwv_by_gender", "stats_fallback", "[性别PWV对比的统计显著性分析结果待补充]")))
    
    elements.append(ParagraphElement(text=get_static_text(static_content, "pwv_by_gender", "interpretation", "虽然本研究未发现显著差异，但性别对心血管健康的影响仍是值得关注的领域。")))

    # Placeholder for descriptive stats table by gender
    if 'descriptive_stats_gender_pwv' in analysis_results and isinstance(analysis_results['descriptive_stats_gender_pwv'], pd.DataFrame):
        elements.append(TableElement(title="不同性别PWV描述性统计", dataframe=analysis_results['descriptive_stats_gender_pwv']))
    else:
        elements.append(ParagraphElement(text="[不同性别PWV的详细描述性统计数据需查阅对应亚组分析的输出。]"))
    elements.append(LineBreakElement())

    # --- 4.3 BMI 分类亚组分析 ---
    h2_4_3_title = get_static_text(static_content, "bmi_subgroup_analysis_intro", "title", "4.3 BMI 分类亚组分析")
    elements.append(HeadingElement(level=2, text=h2_4_3_title, element_id=custom_slugify(h2_4_3_title)))
    elements.append(ParagraphElement(text=get_static_text(static_content, "bmi_subgroup_analysis_intro", "content")))
    
    # Find the plot for BMI group
    bmi_plot = next((f for f in figures_list if isinstance(f, dict) and "cfPWV" in f.get('caption', '').lower() and "bmi" in f.get('caption', '').lower() and "boxplot" in f.get('caption', '').lower()), None)
    if not bmi_plot: # Fallback to path based search
         bmi_plot = next((f for f in figures_list if isinstance(f, dict) and "cfPWV_速度m_s" in f.get('path','') and "BMI_组" in f.get('path','') and "boxplot" in f.get('path','')), None)

    if bmi_plot:
        elements.append(ImageElement(path=bmi_plot['path'], caption=bmi_plot.get('caption', "不同BMI分组的cfPWV分布")))
    else:
        # Fallback to a generic path
        generic_bmi_plot_path = os.path.join(config.get('output_figures_dir', 'output/figures'), "subgroup_analysis", "bmi_group", "boxplot_cfPWV_速度m_s_by_BMI_组.png")
        elements.append(ImageElement(path=generic_bmi_plot_path, caption="不同BMI分组的cfPWV分布情况（默认路径）"))
        elements.append(ParagraphElement(text="[提示: BMI亚组cfPWV对比图表未在figures_list中明确指定，使用了默认路径]"))

    # Add statistical summary from subgroup_analysis_summary data for BMI Group
    bmi_stats_text_parts = []
    if 'subgroup_analysis_summary' in analysis_results:
        subgroup_summary_df = analysis_results['subgroup_analysis_summary']
        if isinstance(subgroup_summary_df, pd.DataFrame):
            # cfPWV by BMI Group
            cfpwv_bmi_stats = subgroup_summary_df[
                (subgroup_summary_df['Target'] == 'cfPWV-速度m/s') & 
                (subgroup_summary_df['Group'] == 'BMI_组')
            ]
            if not cfpwv_bmi_stats.empty:
                p_val_cf = cfpwv_bmi_stats['P_Value'].iloc[0]
                test_used_cf = cfpwv_bmi_stats['TestUsed'].iloc[0]
                bmi_stats_text_parts.append(f"cfPWV ({test_used_cf}, P = {p_val_cf:.3g})")

            # baPWV by BMI Group
            bapwv_bmi_stats = subgroup_summary_df[
                (subgroup_summary_df['Target'] == 'baPWV-平均速度m/s') & 
                (subgroup_summary_df['Group'] == 'BMI_组')
            ]
            if not bapwv_bmi_stats.empty:
                p_val_ba = bapwv_bmi_stats['P_Value'].iloc[0]
                test_used_ba = bapwv_bmi_stats['TestUsed'].iloc[0]
                bmi_stats_text_parts.append(f"baPWV ({test_used_ba}, P = {p_val_ba:.3g})")
    
    if bmi_stats_text_parts:
        summary_text = "根据本研究的统计分析结果，" + " 和 ".join(bmi_stats_text_parts) + " 在不同BMI分组（偏瘦、正常、超重、肥胖）间的差异均未达到统计学显著水平。"
        elements.append(ParagraphElement(text=summary_text))
    else:
        elements.append(ParagraphElement(text=get_static_text(static_content, "pwv_by_bmi", "stats_fallback", "[BMI分组PWV对比的统计显著性分析结果待补充]")))

    elements.append(ParagraphElement(text=get_static_text(static_content, "pwv_by_bmi", "interpretation", "不同体重状态对血管健康的影响复杂，可能受多种混杂因素影响。")))

    # Placeholder for descriptive stats table by BMI group
    if 'descriptive_stats_bmi_group_pwv' in analysis_results and isinstance(analysis_results['descriptive_stats_bmi_group_pwv'], pd.DataFrame):
        elements.append(TableElement(title="不同BMI分组PWV描述性统计", dataframe=analysis_results['descriptive_stats_bmi_group_pwv']))
    else:
        elements.append(ParagraphElement(text="[各BMI分组PWV的详细描述性统计数据需查阅对应亚组分析的输出。]"))
    elements.append(LineBreakElement())

    # --- 4.4 血压状态亚组分析 ---
    h2_4_4_title = get_static_text(static_content, "pwv_by_bp_status", "title", "4.4 血压状态亚组分析")
    elements.append(HeadingElement(level=2, text=h2_4_4_title, element_id=custom_slugify(h2_4_4_title)))
    elements.append(ParagraphElement(text=get_static_text(static_content, "pwv_by_bp_status", "intro_content", "血压是影响动脉硬度的关键因素，本节分析不同血压状态（如正常、高血压前期、高血压）下PWV的差异。")))

    # Find the plot for BP status
    bp_status_plot = next((f for f in figures_list if isinstance(f, dict) and "cfPWV" in f.get('caption', '').lower() and ("血压" in f.get('caption', '').lower() or "bp_status" in f.get('caption', '').lower()) and "boxplot" in f.get('caption', '').lower()), None)
    if not bp_status_plot: # Fallback to path based search
         bp_status_plot = next((f for f in figures_list if isinstance(f, dict) and "cfPWV_速度m_s" in f.get('path','') and "高血压状态" in f.get('path','') and "boxplot" in f.get('path','')), None)
    
    if bp_status_plot:
        elements.append(ImageElement(path=bp_status_plot['path'], caption=bp_status_plot.get('caption', "不同血压状态分组的cfPWV分布")))
    else:
        # Fallback to a generic path
        generic_bp_status_plot_path = os.path.join(config.get('output_figures_dir', 'output/figures'), "subgroup_analysis", "bp_status", "boxplot_cfPWV_速度m_s_by_高血压状态.png")
        elements.append(ImageElement(path=generic_bp_status_plot_path, caption="不同血压状态分组的cfPWV分布情况（默认路径）"))
        elements.append(ParagraphElement(text="[提示: 血压状态亚组cfPWV对比图表未在figures_list中明确指定，使用了默认路径]"))

    # Add statistical summary from subgroup_analysis_summary data for BP Status
    bp_status_stats_text_parts = []
    if 'subgroup_analysis_summary' in analysis_results:
        subgroup_summary_df = analysis_results['subgroup_analysis_summary']
        if isinstance(subgroup_summary_df, pd.DataFrame):
            # cfPWV by BP Status (高血压状态)
            cfpwv_bp_stats = subgroup_summary_df[
                (subgroup_summary_df['Target'] == 'cfPWV-速度m/s') & 
                (subgroup_summary_df['Group'] == '高血压状态')
            ]
            if not cfpwv_bp_stats.empty:
                p_val_cf = cfpwv_bp_stats['P_Value'].iloc[0]
                test_used_cf = cfpwv_bp_stats['TestUsed'].iloc[0]
                significance_cf = "P < 0.001" if p_val_cf < 0.001 else f"P = {p_val_cf:.3g}"
                bp_status_stats_text_parts.append(f"cfPWV ({test_used_cf}, {significance_cf})")

            # baPWV by BP Status (高血压状态)
            bapwv_bp_stats = subgroup_summary_df[
                (subgroup_summary_df['Target'] == 'baPWV-平均速度m/s') & 
                (subgroup_summary_df['Group'] == '高血压状态')
            ]
            if not bapwv_bp_stats.empty:
                p_val_ba = bapwv_bp_stats['P_Value'].iloc[0]
                test_used_ba = bapwv_bp_stats['TestUsed'].iloc[0]
                significance_ba = "P < 0.001" if p_val_ba < 0.001 else f"P = {p_val_ba:.3g}"
                bp_status_stats_text_parts.append(f"baPWV ({test_used_ba}, {significance_ba})")
    
    if bp_status_stats_text_parts:
        summary_text = "统计分析显示，" + " 和 ".join(bp_status_stats_text_parts) + " 在不同血压状态组之间均存在显著差异，通常表现为血压水平较高的人群其PWV也较高。"
        elements.append(ParagraphElement(text=summary_text))
    else:
        elements.append(ParagraphElement(text=get_static_text(static_content, "pwv_by_bp_status", "stats_fallback", "[血压状态PWV对比的统计显著性分析结果待补充]")))

    elements.append(ParagraphElement(text=get_static_text(static_content, "pwv_by_bp_status", "interpretation", "高血压是动脉粥样硬化和血管僵硬度增加的重要危险因素，有效控制血压对维持血管健康至关重要。")))

    # Placeholder for descriptive stats table by BP status
    if 'descriptive_stats_bp_status_pwv' in analysis_results and isinstance(analysis_results['descriptive_stats_bp_status_pwv'], pd.DataFrame):
        elements.append(TableElement(title="不同血压状态分组PWV描述性统计", dataframe=analysis_results['descriptive_stats_bp_status_pwv']))
    else:
        elements.append(ParagraphElement(text="[各血压状态分组PWV的详细描述性统计数据需查阅对应亚组分析的输出。]"))
    elements.append(LineBreakElement())

    # --- 4.5 糖尿病状态亚组分析 --- (NEW SECTION)
    h2_4_5_title_text = "4.5 糖尿病状态亚组分析"
    elements.append(HeadingElement(level=2, text=h2_4_5_title_text, element_id=custom_slugify(h2_4_5_title_text)))
    elements.append(ParagraphElement(text=get_static_text(static_content, "pwv_by_diabetes_status", "intro_content", "糖尿病是心血管疾病的重要危险因素，已知会加速动脉硬化进程。本节旨在比较糖尿病患者与非糖尿病患者的PWV水平。")))

    # Find the plot for Diabetes status
    diabetes_plot = next((f for f in figures_list if isinstance(f, dict) and "cfPWV" in f.get('caption', '').lower() and ("糖尿病" in f.get('caption', '').lower() or "diabetes" in f.get('caption', '').lower()) and "boxplot" in f.get('caption', '').lower()), None)
    if not diabetes_plot: # Fallback to path based search
         diabetes_plot = next((f for f in figures_list if isinstance(f, dict) and "cfPWV_速度m_s" in f.get('path','') and "糖尿病状态" in f.get('path','') and "boxplot" in f.get('path','')), None)

    if diabetes_plot:
        elements.append(ImageElement(path=diabetes_plot['path'], caption=diabetes_plot.get('caption', "不同糖尿病状态分组的cfPWV分布")))
    else:
        # Fallback to a generic path
        generic_diabetes_plot_path = os.path.join(config.get('output_figures_dir', 'output/figures'), "subgroup_analysis", "diabetes_status", "boxplot_cfPWV_速度m_s_by_糖尿病状态.png")
        elements.append(ImageElement(path=generic_diabetes_plot_path, caption="不同糖尿病状态分组的cfPWV分布情况（默认路径）"))
        elements.append(ParagraphElement(text="[提示: 糖尿病状态亚组cfPWV对比图表未在figures_list中明确指定，使用了默认路径]"))

    # Add statistical summary from subgroup_analysis_summary data for Diabetes Status
    diabetes_stats_text_parts = []
    if 'subgroup_analysis_summary' in analysis_results:
        subgroup_summary_df = analysis_results['subgroup_analysis_summary']
        if isinstance(subgroup_summary_df, pd.DataFrame):
            # cfPWV by Diabetes Status (糖尿病状态)
            cfpwv_diabetes_stats = subgroup_summary_df[
                (subgroup_summary_df['Target'] == 'cfPWV-速度m/s') & 
                (subgroup_summary_df['Group'] == '糖尿病状态')
            ]
            if not cfpwv_diabetes_stats.empty:
                p_val_cf = cfpwv_diabetes_stats['P_Value'].iloc[0]
                test_used_cf = cfpwv_diabetes_stats['TestUsed'].iloc[0]
                significance_cf = "P < 0.001" if p_val_cf < 0.001 else f"P = {p_val_cf:.3g}"
                diabetes_stats_text_parts.append(f"cfPWV ({test_used_cf}, {significance_cf})")

            # baPWV by Diabetes Status (糖尿病状态)
            bapwv_diabetes_stats = subgroup_summary_df[
                (subgroup_summary_df['Target'] == 'baPWV-平均速度m/s') & 
                (subgroup_summary_df['Group'] == '糖尿病状态')
            ]
            if not bapwv_diabetes_stats.empty:
                p_val_ba = bapwv_diabetes_stats['P_Value'].iloc[0]
                test_used_ba = bapwv_diabetes_stats['TestUsed'].iloc[0]
                significance_ba = "P < 0.001" if p_val_ba < 0.001 else f"P = {p_val_ba:.3g}"
                diabetes_stats_text_parts.append(f"baPWV ({test_used_ba}, {significance_ba})")
    
    if diabetes_stats_text_parts:
        summary_text = "统计分析显示，" + " 和 ".join(diabetes_stats_text_parts) + " 在不同糖尿病状态组（如非糖尿病、糖尿病前期、糖尿病）之间均存在显著差异。通常，糖尿病患者的PWV水平显著高于非糖尿病人群。"
        elements.append(ParagraphElement(text=summary_text))
    else:
        elements.append(ParagraphElement(text=get_static_text(static_content, "pwv_by_diabetes_status", "stats_fallback", "[糖尿病状态PWV对比的统计显著性分析结果待补充]")))

    elements.append(ParagraphElement(text=get_static_text(static_content, "pwv_by_diabetes_status", "interpretation", "早期筛查和管理糖尿病患者的血管健康状况对于预防心血管并发症至关重要。")))

    # Placeholder for descriptive stats table by diabetes status
    if 'descriptive_stats_diabetes_status_pwv' in analysis_results and isinstance(analysis_results['descriptive_stats_diabetes_status_pwv'], pd.DataFrame):
        elements.append(TableElement(title="不同糖尿病状态分组PWV描述性统计", dataframe=analysis_results['descriptive_stats_diabetes_status_pwv']))
    else:
        elements.append(ParagraphElement(text="[各糖尿病状态分组PWV的详细描述性统计数据需查阅对应亚组分析的输出。]"))
    elements.append(LineBreakElement())

    # --- 4.6 统计功效分析示例 --- (Was 4.5)
    h2_4_6_title = get_static_text(static_content, "power_analysis_intro", "title", "4.6 统计功效分析示例")
    elements.append(HeadingElement(level=2, text=h2_4_6_title, element_id=custom_slugify(h2_4_6_title)))
    elements.append(ParagraphElement(text=get_static_text(static_content, "power_analysis_intro", "content")))

    # Attempt to display a summary from the loaded power analysis results
    power_sample_req_df = analysis_results.get('power_analysis_sample_requirements')
    if power_sample_req_df is not None and isinstance(power_sample_req_df, pd.DataFrame) and not power_sample_req_df.empty:
        # Example: Summarize for a key variable, e.g., 'pwv' or the first one in the table if 'pwv' not directly named
        # The '变量名' column might contain 'pwv', 'age', '性别PWV差异', etc.
        target_var_summary = "pwv" # Prioritize pwv if available
        summary_row = power_sample_req_df[power_sample_req_df['变量名'].str.contains(target_var_summary, case=False)]
        if summary_row.empty and not power_sample_req_df.empty:
            summary_row = power_sample_req_df.iloc[[0]] # Fallback to the first variable
            target_var_summary = summary_row['变量名'].iloc[0]
        
        if not summary_row.empty:
            actual_n = summary_row['实际样本量'].iloc[0]
            
            # Safely get values, providing 'N/A' if column missing or access fails
            real_effect_val = 'N/A'
            if '真实效应量' in summary_row.columns and not summary_row['真实效应量'].empty:
                real_effect_val = summary_row['真实效应量'].iloc[0]
            elif '效应量' in summary_row.columns and not summary_row['效应量'].empty:
                real_effect_val = summary_row['效应量'].iloc[0]

            req_80_real_val = 'N/A'
            if '80%功效所需样本量(真实)' in summary_row.columns and not summary_row['80%功效所需样本量(真实)'].empty:
                req_80_real_val = summary_row['80%功效所需样本量(真实)'].iloc[0]
            elif '80%功效所需样本量' in summary_row.columns and not summary_row['80%功效所需样本量'].empty:
                req_80_real_val = summary_row['80%功效所需样本量'].iloc[0]

            suff_80_real_val = 'N/A'
            if '是否满足80%功效(真实)' in summary_row.columns and not summary_row['是否满足80%功效(真实)'].empty:
                suff_80_real_val = summary_row['是否满足80%功效(真实)'].iloc[0]
            elif '是否满足80%功效' in summary_row.columns and not summary_row['是否满足80%功效'].empty:
                suff_80_real_val = summary_row['是否满足80%功效'].iloc[0]

            req_90_real_val = 'N/A'
            if '90%功效所需样本量(真实)' in summary_row.columns and not summary_row['90%功效所需样本量(真实)'].empty:
                req_90_real_val = summary_row['90%功效所需样本量(真实)'].iloc[0]
            elif '90%功效所需样本量' in summary_row.columns and not summary_row['90%功效所需样本量'].empty:
                req_90_real_val = summary_row['90%功效所需样本量'].iloc[0]

            suff_90_real_val = 'N/A'
            if '是否满足90%功效(真实)' in summary_row.columns and not summary_row['是否满足90%功效(真实)'].empty:
                suff_90_real_val = summary_row['是否满足90%功效(真实)'].iloc[0]
            elif '是否满足90%功效' in summary_row.columns and not summary_row['是否满足90%功效'].empty:
                suff_90_real_val = summary_row['是否满足90%功效'].iloc[0]

            real_effect_str = f"{real_effect_val:.3f}" if isinstance(real_effect_val, (float, int)) else str(real_effect_val)

            power_summary_text = (
                f"以变量 '{target_var_summary}' 为例 (真实效应量: {real_effect_str})： "
                f"本研究实际样本量为 {actual_n}。 "
                f"为达到80%统计功效，所需样本量为 {req_80_real_val} (当前样本是否满足: {suff_80_real_val})。 "
                f"为达到90%统计功效，所需样本量为 {req_90_real_val} (当前样本是否满足: {suff_90_real_val})。"
            )
            elements.append(ParagraphElement(text=power_summary_text))
        else:
            elements.append(ParagraphElement(text="[未能提取关键变量的功效分析摘要。详细信息请查阅Excel报告。]"))
        
        power_excel_path = analysis_results.get('power_analysis_excel_path')
        if power_excel_path:
            elements.append(ParagraphElement(text=f"详细的效应量计算和各变量样本量需求分析已存至：`{os.path.basename(power_excel_path)}` (位于表格输出目录)。"))
        # Optionally, display a small part of the table directly
        # elements.append(TableElement(title="样本量需求分析概要 (部分)", dataframe=power_sample_req_df.head(3)))
    else:
        elements.append(ParagraphElement(text="[统计功效及样本量分析的详细数据未在分析结果中提供或加载失败。]"))

    # Embed power analysis figures
    power_figures_found = False
    for fig_info in figures_list:
        is_dict = isinstance(fig_info, dict)
        has_path = False
        has_caption_keywords = False
        if is_dict:
            has_path = fig_info.get('path') is not None
            caption = fig_info.get('caption', '')
            has_caption_keywords = ("样本量需求对比" in caption) or ("功效分析曲线" in caption)

        if is_dict and has_path and has_caption_keywords:
            elements.append(ImageElement(path=fig_info['path'], caption=fig_info.get('caption', "功效分析相关图表")))
            power_figures_found = True
    
    if not power_figures_found:
        elements.append(ParagraphElement(text="[功效分析相关图表未找到或未添加到figures_list中。]"))
    
    elements.append(LineBreakElement())

    # --- 4.7 基于关键指标的聚类分析 --- (Was 4.6)
    h2_4_7_title = get_static_text(static_content, "cluster_analysis_intro", "title", "4.8 基于关键指标的聚类分析")
    elements.append(HeadingElement(level=2, text=h2_4_7_title, element_id=custom_slugify(h2_4_7_title)))
    elements.append(ParagraphElement(text=get_static_text(static_content, "cluster_analysis_intro", "content")))
    
    cluster_fig_path = os.path.join(config.get('output_figures_dir', 'output/figures'), "advanced_analysis", "cluster_analysis_results.png") # Example
    cluster_fig = next((f for f in figures_list if "cluster_analysis" in f['path'].lower() or "clustering_result" in f['path'].lower()), None)
    if cluster_fig:
        elements.append(ImageElement(path=cluster_fig['path'], caption=cluster_fig.get('caption', "聚类分析结果图")))
    else:
        # Try path from logs
        cluster_fig_alt_path = os.path.join(config.get('output_figures_dir', 'output/figures'), "other", "kmeans_silhouette_plot.png") # As seen in logs
        cluster_fig_alt = next((f for f in figures_list if "kmeans_silhouette_plot.png" in f['path']), None)
        if cluster_fig_alt:
            elements.append(ImageElement(path=cluster_fig_alt['path'], caption=cluster_fig_alt.get('caption', "聚类分析 (剪影图)")))
        else:
             elements.append(ParagraphElement(text="[聚类分析图表未找到]"))


    if 'advanced_analysis' in analysis_results and 'cluster_summary_stats' in analysis_results['advanced_analysis']:
        cluster_summary_df = analysis_results['advanced_analysis']['cluster_summary_stats']
        if isinstance(cluster_summary_df, pd.DataFrame) and not cluster_summary_df.empty:
            elements.append(TableElement(title="各聚类特征统计", dataframe=cluster_summary_df))
        else:
            elements.append(ParagraphElement(text="[聚类特征统计数据未找到或格式不正确]"))

    if 'advanced_analysis' in analysis_results and 'cluster_profile_descriptions' in analysis_results['advanced_analysis']:
        cluster_profiles = analysis_results['advanced_analysis']['cluster_profile_descriptions']
        if isinstance(cluster_profiles, list) and cluster_profiles:
            elements.append(HeadingElement(level=3, text="各聚类画像描述", element_id=custom_slugify("各聚类画像描述")))
            elements.append(UnorderedListElement(items=cluster_profiles))
        elif isinstance(cluster_profiles, str): # If it's a single string block
            elements.append(ParagraphElement(text=cluster_profiles))
        # Ensure a line break after cluster profiles, before next section header
        elements.append(LineBreakElement())
    else: # If cluster_profile_descriptions is not found, still add a line break to separate from next section
        elements.append(LineBreakElement())

    # --- 4.8 PWV与年龄参考值的比较分析 --- (Was 4.7)
    h2_4_8_title = get_static_text(static_content, "pwv_reference_comparison_intro", "title", "4.9 PWV与年龄参考值的比较分析")
    elements.append(HeadingElement(level=2, text=h2_4_8_title, element_id=custom_slugify(h2_4_8_title)))
    elements.append(ParagraphElement(text=get_static_text(static_content, "pwv_reference_comparison_intro", "content")))
    
    pwv_ref_comparison_fig_path = os.path.join(config.get('output_figures_dir', 'output/figures'), "advanced_analysis", "pwv_reference_comparison_chart.png") # Example
    pwv_ref_comparison_fig = next((f for f in figures_list if "pwv_reference_comparison" in f['path'].lower()), None)
    if pwv_ref_comparison_fig:
        elements.append(ImageElement(path=pwv_ref_comparison_fig['path'], caption=pwv_ref_comparison_fig.get('caption', "PWV与参考值比较图")))
    else:
        elements.append(ParagraphElement(text="[PWV与参考值比较图表未找到]"))

    if 'advanced_analysis' in analysis_results and 'pwv_reference_comparison_stats' in analysis_results['advanced_analysis']:
        pwv_ref_stats_df = analysis_results['advanced_analysis']['pwv_reference_comparison_stats']
        if isinstance(pwv_ref_stats_df, pd.DataFrame) and not pwv_ref_stats_df.empty:
            elements.append(TableElement(title="PWV与年龄参考值比较统计", dataframe=pwv_ref_stats_df))
        else:
            elements.append(ParagraphElement(text="[PWV与参考值比较统计数据未找到或格式不正确]"))
    elements.append(LineBreakElement())

    # --- 5. 风险预测模型 ---
    h1_5_text = get_static_text(static_content, "risk_prediction_intro", "title", "5. 风险预测模型")
    elements.append(HeadingElement(level=1, text=h1_5_text, element_id=custom_slugify(h1_5_text)))
    elements.append(ParagraphElement(text=get_static_text(static_content, "risk_prediction_intro", "content")))
    elements.append(LineBreakElement())

    # --- 5.1 PWV超标风险预测模型 ---
    h2_5_1_text = get_static_text(static_content, "pwv_exceedance_model_intro", "title", "5.1 PWV超标风险预测模型")
    elements.append(HeadingElement(level=2, text=h2_5_1_text, element_id=custom_slugify(h2_5_1_text)))
    elements.append(ParagraphElement(text=get_static_text(static_content, "pwv_exceedance_model_intro", "content")))
    elements.append(LineBreakElement())

    # --- 5.1.1 模型性能评估 ---
    h3_5_1_1_text = get_static_text(static_content, "model_performance_heading", "title", "5.1.1 模型性能评估")
    elements.append(HeadingElement(level=3, text=h3_5_1_1_text, element_id=custom_slugify(h3_5_1_1_text)))
    elements.append(ParagraphElement(text=get_static_text(static_content, "model_performance_heading", "content")))

    # Placeholder for model performance metrics table or text
    model_name = "PWV超标风险" # As seen in logs
    if (
        'risk_prediction' in analysis_results and
        model_name in analysis_results['risk_prediction'] and
        'metrics' in analysis_results['risk_prediction'][model_name]
    ):
        metrics = analysis_results['risk_prediction'][model_name]['metrics']
        metrics_text_parts = []
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                if isinstance(value, float):
                    metrics_text_parts.append(f"{key.capitalize()}: {value:.4f}")
                else:
                    metrics_text_parts.append(f"{key.capitalize()}: {value}")
            metrics_summary_text = "模型在测试集上的主要性能指标如下： " + "; ".join(metrics_text_parts) + "。"
            elements.append(ParagraphElement(text=metrics_summary_text))
            
            # Displaying metrics as a list as well
            metrics_list_items = [f"{key.replace('_', ' ').capitalize()}: {value:.4f}" if isinstance(value, float) else f"{key.replace('_', ' ').capitalize()}: {value}" for key, value in metrics.items()]
            elements.append(UnorderedListElement(items=metrics_list_items))

        elif isinstance(metrics, pd.DataFrame):
            elements.append(TableElement(title=f"{model_name} 模型性能指标", dataframe=metrics))
        else:
            elements.append(ParagraphElement(text=f"[{model_name} 模型的性能指标数据格式无法直接展示。]"))
    else:
        elements.append(ParagraphElement(text=f"[{model_name} 模型的性能指标未在分析结果中找到。]"))
    elements.append(LineBreakElement())

    # Add key performance figures
    figures_base_path = os.path.join(config.get('output_figures_dir', 'output/figures'), "..", "image", "风险预测", model_name) # Adjusting base path based on logs
    
    roc_curve_path = os.path.join(figures_base_path, f"{model_name}_roc_curve.png")
    roc_fig = next((f for f in figures_list if f"{model_name}_roc_curve" in f['path']), {'path': roc_curve_path, 'caption': f'{model_name} ROC曲线'})
    elements.append(ImageElement(path=roc_fig['path'], caption=roc_fig.get('caption')))

    pr_curve_path = os.path.join(figures_base_path, f"{model_name}_pr_curve.png")
    pr_fig = next((f for f in figures_list if f"{model_name}_pr_curve" in f['path']), {'path': pr_curve_path, 'caption': f'{model_name} PR曲线'})
    elements.append(ImageElement(path=pr_fig['path'], caption=pr_fig.get('caption')))

    confusion_matrix_path = os.path.join(figures_base_path, f"{model_name}_confusion_matrix.png")
    cm_fig = next((f for f in figures_list if f"{model_name}_confusion_matrix" in f['path']), {'path': confusion_matrix_path, 'caption': f'{model_name} 混淆矩阵'})
    elements.append(ImageElement(path=cm_fig['path'], caption=cm_fig.get('caption')))
    elements.append(LineBreakElement())

    # --- 5.1.2 模型可解释性分析 (SHAP) ---
    h3_5_1_2_text = get_static_text(static_content, "shap_analysis_heading", "title", "5.1.2 模型可解释性分析 (SHAP)")
    elements.append(HeadingElement(level=3, text=h3_5_1_2_text, element_id=custom_slugify(h3_5_1_2_text)))
    elements.append(ParagraphElement(text=get_static_text(static_content, "shap_analysis_heading", "content")))
    elements.append(ParagraphElement(text=get_static_text(static_content, "shap_summary_interpretation", "content")))

    shap_summary_bar_path = os.path.join(figures_base_path, f"{model_name}_shap_summary_bar.png")
    shap_bar_fig = next((f for f in figures_list if f"{model_name}_shap_summary_bar" in f['path']), {'path': shap_summary_bar_path, 'caption': f'{model_name} SHAP特征重要性条形图'})
    elements.append(ImageElement(path=shap_bar_fig['path'], caption=shap_bar_fig.get('caption')))
    
    shap_summary_beeswarm_path = os.path.join(figures_base_path, f"{model_name}_shap_summary_beeswarm.png")
    shap_beeswarm_fig = next((f for f in figures_list if f"{model_name}_shap_summary_beeswarm" in f['path']), {'path': shap_summary_beeswarm_path, 'caption': f'{model_name} SHAP特征重要性蜂群图'})
    elements.append(ImageElement(path=shap_beeswarm_fig['path'], caption=shap_beeswarm_fig.get('caption')))

    # Example of adding a SHAP dependence plot (can add more if needed based on logs or typical analysis)
    # Assuming 'age' is a top feature, as is common.
    # The log showed '年龄收缩压_交互', 'age', 'bapwv_右侧_速度', 'ABI-右侧-胫后血压', 'pwv' as dependence plots
    top_feature_for_dependence = "age" # Placeholder, can be dynamically determined or set
    shap_dependence_age_path = os.path.join(figures_base_path, f"{model_name}_shap_dependence_{top_feature_for_dependence}.png")
    shap_dep_fig_age = next((f for f in figures_list if f"{model_name}_shap_dependence_{top_feature_for_dependence}" in f['path']), {'path': shap_dependence_age_path, 'caption': f'{model_name} SHAP {top_feature_for_dependence} 依赖图'})
    elements.append(ImageElement(path=shap_dep_fig_age['path'], caption=shap_dep_fig_age.get('caption')))
    
    top_feature_for_dependence_2 = "年龄收缩压_交互" 
    shap_dependence_interaction_path = os.path.join(figures_base_path, f"{model_name}_shap_dependence_{top_feature_for_dependence_2}.png")
    shap_dep_fig_interaction = next((f for f in figures_list if f"{model_name}_shap_dependence_{top_feature_for_dependence_2}" in f['path']), {'path': shap_dependence_interaction_path, 'caption': f'{model_name} SHAP {top_feature_for_dependence_2} 依赖图'})
    elements.append(ImageElement(path=shap_dep_fig_interaction['path'], caption=shap_dep_fig_interaction.get('caption')))
    elements.append(LineBreakElement())

    # --- End of Section 5 ---

    # --- 6. 临床风险评估与分层 ---
    h1_6_text = get_static_text(static_content, "clinical_risk_analysis_intro", "title", "6. 临床风险评估与分层")
    elements.append(HeadingElement(level=1, text=h1_6_text, element_id=custom_slugify(h1_6_text)))
    elements.append(ParagraphElement(text=get_static_text(static_content, "clinical_risk_analysis_intro", "content")))
    elements.append(LineBreakElement())

    clinical_data = analysis_results.get('clinical', {})

    # --- 6.1 基于PWV的风险分类 ---
    h2_6_1_text = get_static_text(static_content, "pwv_risk_classification_intro", "title", "6.1 基于PWV的风险分类")
    elements.append(HeadingElement(level=2, text=h2_6_1_text, element_id=custom_slugify(h2_6_1_text)))
    elements.append(ParagraphElement(text=get_static_text(static_content, "pwv_risk_classification_intro", "content")))
    
    pwv_risk_stats = clinical_data.get('pwv_risk_classification_stats')
    if pwv_risk_stats is not None and isinstance(pwv_risk_stats, (pd.DataFrame, pd.Series)):
        if isinstance(pwv_risk_stats, pd.Series):
            pwv_risk_stats = pwv_risk_stats.reset_index()
            pwv_risk_stats.columns = ['PWV风险等级', '数量'] # Or get from config/logs
        # Attempt to get percentage if available
        pwv_risk_percentage = clinical_data.get('pwv_risk_classification_percentage')
        if pwv_risk_percentage is not None and isinstance(pwv_risk_percentage, (pd.DataFrame, pd.Series)):
            if isinstance(pwv_risk_percentage, pd.Series):
                 pwv_risk_percentage = pwv_risk_percentage.reset_index()
                 pwv_risk_percentage.columns = ['PWV风险等级', '百分比(%)']
            pwv_risk_stats = pd.merge(pwv_risk_stats, pwv_risk_percentage, on='PWV风险等级', how='left')
        elements.append(TableElement(title="PWV风险分类统计", dataframe=pwv_risk_stats))
    else:
        elements.append(ParagraphElement(text="[PWV风险分类统计数据未找到或格式不正确]"))
    elements.append(LineBreakElement())

    # --- 6.2 综合风险评分 ---
    h2_6_2_text = get_static_text(static_content, "comprehensive_risk_score_intro", "title", "6.2 综合风险评分")
    elements.append(HeadingElement(level=2, text=h2_6_2_text, element_id=custom_slugify(h2_6_2_text)))
    elements.append(ParagraphElement(text=get_static_text(static_content, "comprehensive_risk_score_intro", "content")))

    comp_risk_score_stats = clinical_data.get('comprehensive_risk_score_stats') # e.g., describe() output
    if comp_risk_score_stats is not None and isinstance(comp_risk_score_stats, (pd.Series, pd.DataFrame)):
        elements.append(ParagraphElement(text="综合风险评分基本统计:"))
        if isinstance(comp_risk_score_stats, pd.Series):
            # Convert Series (like describe output) to a more readable list or simple table
            desc_items = [f"{idx.capitalize()}: {val:.2f}" if isinstance(val, (float, np.number)) else f"{idx.capitalize()}: {val}" for idx, val in comp_risk_score_stats.items()]
            elements.append(UnorderedListElement(items=desc_items))
        else: # DataFrame
            elements.append(TableElement(title="综合风险评分描述性统计", dataframe=comp_risk_score_stats.reset_index()))
    else:
        elements.append(ParagraphElement(text="[综合风险评分统计数据未找到]"))
    elements.append(LineBreakElement(count=1))

    comp_risk_level_stats = clinical_data.get('comprehensive_risk_level_stats') # Counts of levels
    if comp_risk_level_stats is not None and isinstance(comp_risk_level_stats, (pd.DataFrame, pd.Series)):
        if isinstance(comp_risk_level_stats, pd.Series):
            comp_risk_level_stats = comp_risk_level_stats.reset_index()
            comp_risk_level_stats.columns = ['综合风险等级', '数量']
        comp_risk_level_percentage = clinical_data.get('comprehensive_risk_level_percentage')
        if comp_risk_level_percentage is not None and isinstance(comp_risk_level_percentage, (pd.DataFrame, pd.Series)):
            if isinstance(comp_risk_level_percentage, pd.Series):
                comp_risk_level_percentage = comp_risk_level_percentage.reset_index()
                comp_risk_level_percentage.columns = ['综合风险等级', '百分比(%)']
            comp_risk_level_stats = pd.merge(comp_risk_level_stats, comp_risk_level_percentage, on='综合风险等级', how='left')
        elements.append(TableElement(title="综合风险等级分布", dataframe=comp_risk_level_stats))
    else:
        elements.append(ParagraphElement(text="[综合风险等级分布数据未找到]"))
    elements.append(LineBreakElement())

    # --- 6.3 10年心血管疾病(CVD)风险评估 ---
    h2_6_3_text = get_static_text(static_content, "cvd_risk_assessment_intro", "title", "6.3 10年心血管疾病(CVD)风险评估")
    elements.append(HeadingElement(level=2, text=h2_6_3_text, element_id=custom_slugify(h2_6_3_text)))
    elements.append(ParagraphElement(text=get_static_text(static_content, "cvd_risk_assessment_intro", "content")))

    cvd_risk_stats = clinical_data.get('cvd_risk_stats') # e.g., describe() output for 10-year CVD risk %
    if cvd_risk_stats is not None and isinstance(cvd_risk_stats, (pd.Series, pd.DataFrame)):
        elements.append(ParagraphElement(text="10年CVD风险(%)基本统计:"))
        if isinstance(cvd_risk_stats, pd.Series):
            desc_items = [f"{idx.capitalize()}: {val:.2f}%" if idx == 'mean' or idx == 'std' or '%' in idx else f"{idx.capitalize()}: {val:.2f}" for idx, val in cvd_risk_stats.items()]
            elements.append(UnorderedListElement(items=desc_items))
        else: # DataFrame
            elements.append(TableElement(title="10年CVD风险(%)描述性统计", dataframe=cvd_risk_stats.reset_index()))
    else:
        elements.append(ParagraphElement(text="[10年CVD风险统计数据未找到]"))
    elements.append(LineBreakElement(count=1))

    cvd_risk_level_stats = clinical_data.get('cvd_risk_level_stats')
    if cvd_risk_level_stats is not None and isinstance(cvd_risk_level_stats, (pd.DataFrame, pd.Series)):
        if isinstance(cvd_risk_level_stats, pd.Series):
            cvd_risk_level_stats = cvd_risk_level_stats.reset_index()
            cvd_risk_level_stats.columns = ['CVD风险等级', '数量']
        cvd_risk_level_percentage = clinical_data.get('cvd_risk_level_percentage')
        if cvd_risk_level_percentage is not None and isinstance(cvd_risk_level_percentage, (pd.DataFrame, pd.Series)):
            if isinstance(cvd_risk_level_percentage, pd.Series):
                cvd_risk_level_percentage = cvd_risk_level_percentage.reset_index()
                cvd_risk_level_percentage.columns = ['CVD风险等级', '百分比(%)']
            cvd_risk_level_stats = pd.merge(cvd_risk_level_stats, cvd_risk_level_percentage, on='CVD风险等级', how='left')
        elements.append(TableElement(title="CVD风险等级分布", dataframe=cvd_risk_level_stats))
    else:
        elements.append(ParagraphElement(text="[CVD风险等级分布数据未找到]"))
    elements.append(LineBreakElement())

    age_cvd_risk_crosstab = clinical_data.get('age_cvd_risk_crosstab')
    if age_cvd_risk_crosstab is not None and isinstance(age_cvd_risk_crosstab, pd.DataFrame):
        elements.append(ParagraphElement(text="各年龄组的CVD风险等级分布如下表所示（百分比%）："))
        elements.append(TableElement(title="年龄组与CVD风险等级交叉分析 (%) ", dataframe=age_cvd_risk_crosstab.reset_index()))
    else:
        elements.append(ParagraphElement(text="[年龄组与CVD风险等级交叉分析数据未找到]"))
    elements.append(LineBreakElement())

    # --- 6.4 高风险人群特征分析 ---
    h2_6_4_text = get_static_text(static_content, "high_risk_group_characteristics_intro", "title", "6.4 高风险人群特征分析")
    elements.append(HeadingElement(level=2, text=h2_6_4_text, element_id=custom_slugify(h2_6_4_text)))
    elements.append(ParagraphElement(text=get_static_text(static_content, "high_risk_group_characteristics_intro", "content")))
    
    high_risk_profile = clinical_data.get('high_risk_group_profile') # e.g., describe() output for high CVD risk group
    if high_risk_profile is not None and isinstance(high_risk_profile, pd.DataFrame):
        elements.append(ParagraphElement(text="CVD高风险人群的主要临床特征统计如下："))
        elements.append(TableElement(title="CVD高风险人群特征描述", dataframe=high_risk_profile.reset_index()))
    else:
        elements.append(ParagraphElement(text="[高风险人群特征数据未找到或格式不正确]"))
    elements.append(LineBreakElement())

    # --- 7. 结论与建议 ---
    # Using overall_conclusion_title for the main H1 of section 7
    h1_7_text = get_static_text(static_content, "overall_conclusion_title", "title", "7. 总结与主要发现")
    elements.append(HeadingElement(level=1, text=h1_7_text, element_id=custom_slugify(h1_7_text)))
    elements.append(ParagraphElement(text=get_static_text(static_content, "overall_conclusion_title", "content")))
    elements.append(LineBreakElement())

    # --- 7.1 临床与健康管理建议 ---
    h2_7_1_text = get_static_text(static_content, "recommendations_title", "title", "7.1 临床与健康管理建议")
    elements.append(HeadingElement(level=2, text=h2_7_1_text, element_id=custom_slugify(h2_7_1_text)))
    elements.append(ParagraphElement(text=get_static_text(static_content, "recommendations_title", "content")))
    
    recommendations = static_content.get("recommendations_list", {}).get("items", [])
    if recommendations:
        elements.append(UnorderedListElement(items=recommendations))
    else:
        elements.append(ParagraphElement(text="[具体建议列表未在静态内容中配置]"))
    elements.append(LineBreakElement())

    # --- 7.2 未来研究展望 ---
    h2_7_2_text = get_static_text(static_content, "future_research_title", "title", "7.2 未来研究展望")
    elements.append(HeadingElement(level=2, text=h2_7_2_text, element_id=custom_slugify(h2_7_2_text)))
    elements.append(ParagraphElement(text=get_static_text(static_content, "future_research_title", "content")))
    elements.append(LineBreakElement())
    
    # --- End of Report ---

    logger.info(f"报告元素列表组合完毕，共 {len(elements)} 个元素。")
    return elements

if __name__ == '__main__':
    # Dummy data for testing compose_report_elements
    print("Testing report_composer.py standalone...")
    
    # Ensure static content file exists for standalone test
    if not os.path.exists(STATIC_CONTENT_FILE):
        print(f"Warning: Static content file {STATIC_CONTENT_FILE} not found for standalone test.")

    dummy_analysis_results = {
        'basic_stats': pd.DataFrame({
            'mean': {'age': 50.5, 'pwv': 9.5, '收缩压': 130.0, '舒张压': 80.0, '脉压差': 50.0, 'cfpwv_速度': 9.2, 'bapwv_右侧_速度': 11.0, 'bapwv_左侧_速度': 11.1},
            'std': {'age': 10.2, 'pwv': 1.5, '收缩压': 15.0, '舒张压': 10.0, '脉压差': 8.0, 'cfpwv_速度': 1.2, 'bapwv_右侧_速度': 1.8, 'bapwv_左侧_速度': 1.9},
            'min': {'age': 25, 'pwv': 6.0, '收缩压': 100, '舒张压': 60, '脉压差': 30, 'cfpwv_速度': 6.1, 'bapwv_右侧_速度': 7.0, 'bapwv_左侧_速度': 7.1},
            '25%': {'age': 45, 'pwv': 8.5, '收缩压': 120, '舒张压': 75, '脉压差': 45, 'cfpwv_速度': 8.2, 'bapwv_右侧_速度': 9.9, 'bapwv_左侧_速度': 10.0},
            '50%': {'age': 50, 'pwv': 9.2, '收缩压': 128, '舒张压': 80, '脉压差': 48, 'cfpwv_速度': 9.0, 'bapwv_右侧_速度': 10.8, 'bapwv_左侧_速度': 10.9},
            '75%': {'age': 60, 'pwv': 10.5, '收缩压': 140, '舒张压': 88, '脉压差': 55, 'cfpwv_速度': 10.0, 'bapwv_右侧_速度': 12.0, 'bapwv_左侧_速度': 12.1},
            'max': {'age': 75, 'pwv': 12.5, '收缩压': 160, '舒张压': 95, '脉压差': 65, 'cfpwv_速度': 12.1, 'bapwv_右侧_速度': 15.0, 'bapwv_左侧_速度': 15.2},
            '变异系数': {'age': 20.2, 'pwv': 15.8, '收缩压': 11.5, '舒张压': 12.5, '脉压差': 16.0, 'cfpwv_速度': 13.0, 'bapwv_右侧_速度': 16.4, 'bapwv_左侧_速度': 17.1}
        }).rename_axis("指标"),
        'correlation_analysis': pd.DataFrame({
            '变量1': ['pwv', 'pwv', 'age', 'pwv'],
            '变量2': ['age', '收缩压', '收缩压', '舒张压'],
            '相关系数': [0.6, 0.4, 0.3, 0.15],
            'P值': [0.001, 0.005, 0.01, 0.04]
        }),
        'missing_summary': pd.DataFrame({
            '列名': ['cfpwv_速度', 'bapwv_右侧_速度', '血常规-白细胞'],
            '缺失数量': [10, 5, 50],
            '缺失比例(%)': [4.5, 2.25, 22.5],
            '处理方法': ['中位数填充', '中位数填充', '未处理/删除列']
        }),
        'pwv_stats_by_age_group': pd.DataFrame({
            '年龄组': ['<40', '40-60', '>60'],
            'PWV均值': [1000, 1250, 1500],
            'PWV标准差': [150, 180, 220]
        }),
        'pwv_stats_by_gender': pd.DataFrame({
            '性别': ['男', '女'],
            'PWV均值': [1280, 1220],
            'PWV标准差': [190, 170]
        })
    }
    dummy_figures_list = [
        {'path': 'output/figures/distribution/distribution_age.png', 'caption': '年龄分布直方图'},
        {'path': 'output/figures/distribution/distribution_pwv.png', 'caption': 'PWV分布直方图'},
        {'path': 'output/figures/boxplot/overall_boxplot.png', 'caption': '主要指标箱线图'},
        {'path': 'output/figures/regression/pwv_age_regression.png', 'caption': 'PWV与年龄回归'},
        {'path': 'output/figures/regression/pwv_收缩压_regression.png', 'caption': 'PWV与收缩压回归'},
        {'path': 'output/figures/regression/pwv_舒张压_regression.png', 'caption': 'PWV与舒张压回归'},
        {'path': 'output/figures/correlation/correlation_heatmap.png', 'caption': '相关性热力图'},
        {'path': 'output/figures/boxplot/pwv_outliers_boxplot.png', 'caption': 'PWV离群值箱线图'},
        {'path': 'output/figures/age_group/pwv_by_age_group_boxplot.png', 'caption': '不同年龄组PWV分布箱线图'},
        {'path': 'output/figures/gender/pwv_by_gender_boxplot.png', 'caption': '不同性别PWV分布箱线图'},
        # Add paths for risk prediction figures based on logs
        {'path': 'output/image/风险预测/PWV超标风险/PWV超标风险_roc_curve.png', 'caption': 'PWV超标风险 ROC曲线'},
        {'path': 'output/image/风险预测/PWV超标风险/PWV超标风险_pr_curve.png', 'caption': 'PWV超标风险 PR曲线'},
        {'path': 'output/image/风险预测/PWV超标风险/PWV超标风险_confusion_matrix.png', 'caption': 'PWV超标风险 混淆矩阵'},
        {'path': 'output/image/风险预测/PWV超标风险/PWV超标风险_shap_summary_bar.png', 'caption': 'PWV超标风险 SHAP条形图'},
        {'path': 'output/image/风险预测/PWV超标风险/PWV超标风险_shap_summary_beeswarm.png', 'caption': 'PWV超标风险 SHAP蜂群图'},
        {'path': 'output/image/风险预测/PWV超标风险/PWV超标风险_shap_dependence_age.png', 'caption': 'PWV超标风险 SHAP age依赖图'},
        {'path': 'output/image/风险预测/PWV超标风险/PWV超标风险_shap_dependence_年龄收缩压_交互.png', 'caption': 'PWV超标风险 SHAP 年龄收缩压_交互依赖图'}
    ]
    dummy_config = {
        "report_title": "测试PWV数据分析报告(结构化)",
        "image_base_dir_markdown": "output/figures", 
        "output_figures_dir": "output/figures" # Used for constructing figure paths
    }

    # Add dummy risk prediction results to mock_analysis_results for testing
    dummy_analysis_results['risk_prediction'] = {
        "PWV超标风险": {
            "metrics": {
                'accuracy': 0.625, 
                'f1': 0.3636, 
                'precision': 0.375, 
                'recall': 0.3529, 
                'roc_auc': 0.6712
            }
        }
    }

    # Add dummy clinical analysis results for testing Section 6
    dummy_analysis_results['clinical'] = {
        'pwv_risk_classification_stats': pd.Series({'正常': 213, '边缘': 6, '显著风险': 2}, name="数量").rename_axis("PWV风险等级"),
        'pwv_risk_classification_percentage': pd.Series({'正常': 96.38, '边缘': 2.71, '显著风险': 0.90}, name="百分比(%)").rename_axis("PWV风险等级"),
        'comprehensive_risk_score_stats': pd.Series({
            'count': 221.00, 'mean': 3.53, 'std': 1.85, 'min': 0.00, 
            '25%': 2.50, '50%': 3.50, '75%': 5.00, 'max': 8.00
        }, name="综合风险评分"),
        'comprehensive_risk_level_stats': pd.Series({'中等风险': 96, '高风险': 52, '低风险': 44, '极高风险': 20}, name="数量").rename_axis("综合风险等级"),
        'comprehensive_risk_level_percentage': pd.Series({'中等风险': 45.28, '高风险': 24.53, '低风险': 20.75, '极高风险': 9.43}, name="百分比(%)").rename_axis("综合风险等级"),
        'cvd_risk_stats': pd.Series({
            'count': 221.00, 'mean': 4.49, 'std': 2.54, 'min': 1.00,
            '25%': 2.30, '50%': 4.70, '75%': 6.70, 'max': 13.30
        }, name="10年CVD风险(%)"),
        'cvd_risk_level_stats': pd.Series({'低风险': 140, '中等风险': 77, '高风险': 4, '极高风险': 0}, name="数量").rename_axis("CVD风险等级"),
        'cvd_risk_level_percentage': pd.Series({'低风险': 63.35, '中等风险': 34.84, '高风险': 1.81, '极高风险': 0.00}, name="百分比(%)").rename_axis("CVD风险等级"),
        'age_cvd_risk_crosstab': pd.DataFrame({
            '年龄组': ['<30', '30-39', '40-49', '50-59', '60-69', '70+'],
            '低风险': [100.0, 100.0, 100.0, 96.43, 58.46, 22.06],
            '中等风险': [0.0, 0.0, 0.0, 3.57, 41.54, 72.06],
            '高风险': [0.0, 0.0, 0.0, 0.0, 0.0, 5.88]
        }),
        'high_risk_group_profile': pd.DataFrame({
            '指标': ['age', 'gender', 'bmi', '收缩压', '舒张压', 'pwv'],
            'count': [4.0]*6,
            'mean': [78.75, 1.0, 25.97, 146.75, 73.50, 12.15],
            'std': [6.85, 0.0, 0.52, 7.04, 7.72, 0.87]
            # Min, 25%, 50%, 75%, Max can be added if needed from logs
        }).set_index('指标')
    }

    composed_elements = compose_report_elements(dummy_analysis_results, dummy_figures_list, dummy_config)

    print(f"\n--- Composed Elements ({len(composed_elements)}) ---")
    for i, element in enumerate(composed_elements):
        print(f"{i+1}. Type: {element.element_type}")
        if hasattr(element, 'text'):
            # 修复f-string中的转义字符问题
            if element.text:
                text_preview = str(element.text)[:70]
                text_preview = text_preview.replace('\n', ' ')
                print(f"   Text: {text_preview}...")
            else:
                print("   Text: ...")
        if hasattr(element, 'level'):
            print(f"   Level: {element.level}")
        if hasattr(element, 'path'):
            print(f"   Path: {element.path}")
        if isinstance(element, TableElement) and element.dataframe is not None:
            print(f"   Table Title: {element.title}")
            # Test with a more robust markdown conversion for tables during test print
            try: from tabulate import tabulate
            except: tabulate = lambda df, headers, tablefmt: str(df.head(2))
            print(tabulate(element.dataframe.head(2), headers='keys', tablefmt='pipe'))
            
        if hasattr(element, 'items'):
             print(f"   Items count: {len(element.items)}")

    print("\nStandalone test for report_composer.py finished.") 

    # 用于调试
    for element in composed_elements:
        if isinstance(element, HeadingElement):
            print(f"Heading L{element.level}: {element.text}...")
        elif isinstance(element, ParagraphElement):
            # 修复之前的反斜杠转义问题
            text_preview = str(element.text)[:70]
            text_preview = text_preview.replace('\n', ' ')
            print(f"   Text: {text_preview}...")
        elif isinstance(element, ImageElement):
            print(f"   Image: {element.path}...")