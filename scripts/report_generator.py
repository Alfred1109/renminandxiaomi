#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main script to generate all report formats (Markdown, HTML, Word, etc.)
by orchestrating report_composer and report_renderers.
"""

import os
import logging
import pandas as pd
from typing import List, Dict, Any
import subprocess
import json
import glob

try:
    from scripts.report_composer import compose_report_elements
    from scripts.report_renderers import render_elements_to_markdown, render_elements_to_html, render_elements_to_word
    from scripts.report_utils import create_report_directories, extract_tables_from_markdown_to_excel
except ImportError:
    # Fallback for potential path issues or direct execution
    from report_composer import compose_report_elements
    from report_renderers import render_elements_to_markdown, render_elements_to_html, render_elements_to_word
    from report_utils import create_report_directories, extract_tables_from_markdown_to_excel

logger = logging.getLogger(__name__)

def generate_all_reports(
    data: pd.DataFrame, # Added: The main DataFrame, useful for general stats in intro
    analysis_results: Dict[str, Any],
    figures_list: List[Dict[str, str]], # List of dicts, each with 'path' and 'caption'
    config: Dict[str, Any],
    data_cleaning_summary_md: str = "" # Added: Markdown string for data cleaning
) -> List[str]:
    """
    Generates all configured report outputs.

    Args:
        data: The main pandas DataFrame after processing.
        analysis_results: A dictionary containing all data and statistical results.
        figures_list: A list of dictionaries, where each dictionary contains 'path' 
                      (relative to project root, e.g., 'output/figures/some_plot.png') 
                      and 'caption' for a figure.
        config: A dictionary containing configuration options, including:
                - report_title (str): The main title for the report.
                - base_output_dir (str): The base directory for all outputs (e.g., "output").
                - reports_subdir (str): Subdirectory for reports (e.g., "reports").
                - tables_subdir (str): Subdirectory for tables (e.g., "tables").
                - markdown_filename (str): Filename for the Markdown report.
                - html_filename (str): Filename for the HTML report.
                - word_filename (str): Filename for the Word report.
                - excel_filename (str): Filename for the Excel table summary.
        data_cleaning_summary_md: Markdown string detailing data cleaning steps and results.
    """
    generated_paths = [] # To store paths of generated reports
    logger.info("============================== 开始生成所有报告 ==============================")

    base_output_dir = config.get("base_output_dir", "output")
    reports_subdir = config.get("reports_subdir", "reports")
    tables_subdir = config.get("tables_subdir", "tables")

    report_output_dir = os.path.join(base_output_dir, reports_subdir)
    table_output_dir = os.path.join(base_output_dir, tables_subdir)

    # Create directories if they don't exist
    create_report_directories(base_output_dir=base_output_dir)

    logger.info("📝 开始生成结构化报告元素...")
    # Ensure figures_list paths are adjusted if necessary (e.g. making them relative to a figures dir)
    # For now, assume paths in figures_list are suitable for embedding or direct use.
    # The composer might expect paths relative to where the final MD/HTML is, or absolute.
    # Let's assume 'output_figures_dir' is part of config for composer.
    
    # Update config for composer if needed, e.g., with image directory information
    composer_config = config.copy() # Avoid modifying original config dict directly if not intended
    # composer_config['output_figures_dir'] = os.path.join(base_output_dir, config.get('figures_subdir', 'figures'))


    report_elements = compose_report_elements(
        data=data,
        analysis_results=analysis_results, 
        figures_list=figures_list, 
        config=composer_config,
        data_cleaning_summary_md=data_cleaning_summary_md # Pass it here
    )
    logger.info(f"📄 结构化报告元素生成完毕，共 {len(report_elements)} 个元素。")

    # --- Generate Markdown Report ---
    md_filename = config.get("markdown_filename", "PWV数据分析综合报告.md")
    md_filepath = os.path.join(report_output_dir, md_filename)
    logger.info(f"🔄 正在生成Markdown报告: {md_filepath}...")
    try:
        render_elements_to_markdown(report_elements, md_filepath, config=config)
        logger.info(f"✅ Markdown报告已保存至: {md_filepath}")
        generated_paths.append(md_filepath)
    except Exception as e:
        logger.error(f"❌ 生成Markdown报告失败: {e}", exc_info=True)
        # Continue to other formats if possible

    # --- Extract Tables from Markdown to Excel ---
    excel_filename = config.get("excel_filename", "PWV数据分析报告_表格汇总.xlsx")
    excel_filepath = os.path.join(table_output_dir, excel_filename)
    logger.info(f"🔄 从 {md_filepath} 提取表格数据到Excel: {excel_filepath}...")
    try:
        # This utility would need to parse the generated markdown file
        extract_tables_from_markdown_to_excel(md_filepath, excel_filepath)
        logger.info(f"✅ 成功提取表格到: {excel_filepath}")
        generated_paths.append(excel_filepath)
    except Exception as e:
        logger.error(f"❌ 从Markdown提取表格到Excel失败: {e}", exc_info=True)

    # --- Generate HTML Report ---
    html_filename = config.get("html_filename", "PWV数据分析综合报告.html")
    html_filepath = os.path.join(report_output_dir, html_filename)
    logger.info(f"🔄 正在将结构化元素转换为HTML文档: {html_filepath}...")
    try:
        render_elements_to_html(report_elements, html_filepath, config=config)
        logger.info(f"✅ HTML报告已保存至: {html_filepath}")
        generated_paths.append(html_filepath)
    except Exception as e:
        logger.error(f"❌ 生成HTML报告失败: {e}", exc_info=True)

    # --- Generate Word Report ---
    word_filename = config.get("word_filename", "PWV数据分析综合报告.docx")
    word_filepath = os.path.join(report_output_dir, word_filename)
    logger.info(f"🔄 正在将结构化元素转换为Word文档: {word_filepath}...")
    try:
        render_elements_to_word(report_elements, word_filepath, config=config)
        logger.info(f"✅ Word报告已保存至: {word_filepath}")
        generated_paths.append(word_filepath)
    except Exception as e:
        logger.error(f"❌ 生成Word报告失败: {e}", exc_info=True)

    logger.info("============================== 所有报告生成完毕 ==============================")
    for path in generated_paths:
        logger.info(f"  -> {path}")
    
    return generated_paths # Return the list of paths


if __name__ == '__main__':
    # This is a basic test execution.
    # In a real scenario, analysis_results and figures_list would be populated by the analysis pipeline.
    
    print("运行 report_generator.py 作为独立脚本进行测试...")

    # Setup basic logging for the test
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Initialize analysis_results and figures_list
    mock_analysis_results = {}
    mock_figures_list = []

    # --- Run Data Missing and Cleaning Analysis ---
    data_cleaning_script_path = os.path.join("scripts", "temp_analysis", "data_missing_and_cleaning_analysis.py")
    data_cleaning_md_report_path = os.path.join("scripts", "temp_analysis", "data_missing_and_cleaning_analysis.md")
    data_prep_summary_path = os.path.join("output", "data", "data_preparation_summary.json")

    logger.info(f"执行数据清洗脚本: {data_cleaning_script_path}...")
    try:
        # Ensure the script is executable or run with python interpreter
        process = subprocess.run(['python3', data_cleaning_script_path], capture_output=True, text=True, check=True)
        logger.info("数据清洗脚本执行成功。")
        # print(process.stdout) # Optional: print script output
        if process.stderr:
            logger.warning(f"数据清洗脚本标准错误输出:\n{process.stderr}")

        # Load the generated Markdown content for data cleaning details
        if os.path.exists(data_cleaning_md_report_path):
            with open(data_cleaning_md_report_path, 'r', encoding='utf-8') as f:
                mock_analysis_results['data_missing_and_cleaning_details_md'] = f.read()
            logger.info(f"已加载数据清洗详情报告: {data_cleaning_md_report_path}")
        else:
            logger.warning(f"数据清洗详情报告未找到: {data_cleaning_md_report_path}")
            mock_analysis_results['data_missing_and_cleaning_details_md'] = "[数据清洗与缺失值分析的详细报告未生成或未找到]"

        # Load data preparation summary
        if os.path.exists(data_prep_summary_path):
            with open(data_prep_summary_path, 'r', encoding='utf-8') as f:
                mock_analysis_results['data_preparation_summary'] = json.load(f)
            logger.info(f"已加载数据准备摘要: {data_prep_summary_path}")
        else:
            logger.warning(f"数据准备摘要 JSON 文件未找到: {data_prep_summary_path}")
            mock_analysis_results['data_preparation_summary'] = {} # Provide empty dict as fallback

    except subprocess.CalledProcessError as e:
        logger.error(f"执行数据清洗脚本 {data_cleaning_script_path} 失败. 返回码: {e.returncode}")
        logger.error(f"标准输出:\n{e.stdout}")
        logger.error(f"标准错误:\n{e.stderr}")
        mock_analysis_results['data_missing_and_cleaning_details_md'] = "[数据清洗与缺失值分析脚本执行失败，详情请查看日志]"
        mock_analysis_results['data_preparation_summary'] = {}
    except FileNotFoundError:
        logger.error(f"数据清洗脚本 {data_cleaning_script_path} 未找到。请确保路径正确。")
        mock_analysis_results['data_missing_and_cleaning_details_md'] = "[数据清洗与缺失值分析脚本未找到]"
        mock_analysis_results['data_preparation_summary'] = {}
    except Exception as e:
        logger.error(f"加载数据清洗脚本输出时发生意外错误: {e}", exc_info=True)
        mock_analysis_results['data_missing_and_cleaning_details_md'] = "[加载数据清洗报告时发生未知错误]"
        mock_analysis_results['data_preparation_summary'] = {}

    # --- Run Subgroup Analysis ---
    subgroup_script_path = os.path.join("scripts", "temp_analysis", "subgroup_analysis.py")
    subgroup_summary_csv_path = os.path.join("output", "subgroup_analysis", "subgroup_analysis_summary.csv")
    subgroup_figures_base_dir = os.path.join("output", "subgroup_analysis")

    logger.info(f"执行亚组分析脚本: {subgroup_script_path}...")
    try:
        process = subprocess.run(['python3', subgroup_script_path], capture_output=True, text=True, check=True)
        logger.info("亚组分析脚本执行成功。")
        if process.stderr:
            logger.warning(f"亚组分析脚本标准错误输出:\n{process.stderr}")

        # Load subgroup analysis summary CSV
        if os.path.exists(subgroup_summary_csv_path):
            mock_analysis_results['subgroup_analysis_summary'] = pd.read_csv(subgroup_summary_csv_path)
            logger.info(f"已加载亚组分析摘要: {subgroup_summary_csv_path}")
        else:
            logger.warning(f"亚组分析摘要CSV未找到: {subgroup_summary_csv_path}")
            # mock_analysis_results['subgroup_analysis_summary'] = pd.DataFrame() # Fallback

        # Scan for subgroup analysis figures and add them to mock_figures_list
        figure_patterns = [
            os.path.join(subgroup_figures_base_dir, "cfPWV_速度m_s", "**", "boxplot_*.png"),
            os.path.join(subgroup_figures_base_dir, "baPWV_平均速度m_s", "**", "boxplot_*.png"),
            os.path.join(subgroup_figures_base_dir, "ABI_Overall", "**", "boxplot_*.png") # Add other targets if any
        ]
        
        found_subgroup_figures = 0
        for pattern in figure_patterns:
            for fig_path in glob.glob(pattern, recursive=True):
                # Construct a somewhat descriptive caption from the path
                # Path: output/subgroup_analysis/TARGET/GROUP/boxplot_TARGET_by_GROUP.png
                parts = fig_path.split(os.sep)
                try:
                    group_name = parts[-2]
                    target_variable_full = parts[-3]
                    # Clean up target variable name for caption, e.g., "cfPWV_速度m_s" -> "cfPWV 速度m/s"
                    target_variable_cleaned = target_variable_full.replace("_", " ") 
                    filename = os.path.basename(fig_path)
                    caption = f"{target_variable_cleaned} by {group_name} - {filename}"
                except IndexError:
                    caption = os.path.basename(fig_path) # Fallback caption
                
                # Ensure path is relative to project root for consistency if needed, though glob gives full/relative based on pattern
                # For now, assuming the paths from glob are what composer expects or can handle.
                # Convert to relative path from project root if fig_path is absolute for portability
                relative_fig_path = os.path.relpath(fig_path, os.getcwd()) # getcwd() is workspace root here
                mock_figures_list.append({'path': relative_fig_path, 'caption': caption})
                found_subgroup_figures += 1
        logger.info(f"扫描到 {found_subgroup_figures} 个亚组分析图表并已添加到 figures_list。")

    except subprocess.CalledProcessError as e:
        logger.error(f"执行亚组分析脚本 {subgroup_script_path} 失败. 返回码: {e.returncode}")
        logger.error(f"标准输出:\n{e.stdout}")
        logger.error(f"标准错误:\n{e.stderr}")
    except FileNotFoundError:
        logger.error(f"亚组分析脚本 {subgroup_script_path} 未找到。请确保路径正确。")
    except Exception as e:
        logger.error(f"加载亚组分析脚本输出时发生意外错误: {e}", exc_info=True)

    # --- Run Sample Power Analysis ---
    power_analysis_script_path = os.path.join("scripts", "temp_analysis", "sample_power_analysis.py")
    power_analysis_figures_dir = os.path.join("output", "figures", "other")
    power_analysis_excel_dir = os.path.join("output", "tables") # As defined in sample_power_analysis.py

    logger.info(f"执行样本量及功效分析脚本: {power_analysis_script_path}...")
    power_excel_path = None
    try:
        process = subprocess.run(['python3', power_analysis_script_path], capture_output=True, text=True, check=True)
        logger.info("样本量及功效分析脚本执行成功。")
        # Try to parse the Excel path from the script's stdout
        # Expected output line: "结果已保存到 output/tables/样本量分析结果_YYYYMMDD_HHMMSS.xlsx"
        for line in process.stdout.splitlines():
            if "结果已保存到" in line and ".xlsx" in line:
                power_excel_path = line.split("结果已保存到")[-1].strip()
                logger.info(f"从脚本输出中解析得到Excel文件路径: {power_excel_path}")
                break
        
        if process.stderr:
            logger.warning(f"样本量及功效分析脚本标准错误输出:\n{process.stderr}")

        if power_excel_path and os.path.exists(power_excel_path):
            mock_analysis_results['power_analysis_effect_sizes'] = pd.read_excel(power_excel_path, sheet_name='效应量计算结果')
            mock_analysis_results['power_analysis_sample_requirements'] = pd.read_excel(power_excel_path, sheet_name='样本量需求分析')
            logger.info(f"已加载样本量及功效分析Excel数据: {power_excel_path}")
            # Add the excel itself to analysis_results for potential direct linking or full table display if needed
            mock_analysis_results['power_analysis_excel_path'] = power_excel_path
        elif power_excel_path: # Path parsed but file not found
            logger.warning(f"样本量分析Excel文件 {power_excel_path} 未找到，尽管路径已从输出解析。")
        else:
            logger.warning("未能从样本量分析脚本输出中解析Excel文件路径。相关数据可能无法加载。")

        # Scan for power analysis figures
        power_figure_patterns = [
            os.path.join(power_analysis_figures_dir, "对比效应量的样本量需求_*.png"),
            os.path.join(power_analysis_figures_dir, "*_power_curve.png")
        ]
        found_power_figures = 0
        for pattern in power_figure_patterns:
            for fig_path in glob.glob(pattern):
                # filename_body = os.path.splitext(os.path.basename(fig_path))[0]
                # caption = filename_body.replace("_", " ").capitalize()
                # A more specific caption logic based on filename patterns:
                filename = os.path.basename(fig_path)
                caption = ""
                if "对比效应量的样本量需求_" in filename:
                    var_name = filename.replace("对比效应量的样本量需求_", "").replace(".png", "")
                    caption = f"样本量需求对比 - {var_name.replace('_', ' ').title()}"
                elif "_power_curve.png" in filename:
                    var_name = filename.replace("_power_curve.png", "")
                    caption = f"功效分析曲线 - {var_name.replace('_', ' ').title()}"
                else:
                    caption = os.path.splitext(filename)[0].replace("_", " ").capitalize() # Fallback

                relative_fig_path = os.path.relpath(fig_path, os.getcwd())
                mock_figures_list.append({'path': relative_fig_path, 'caption': caption})
                found_power_figures += 1
        logger.info(f"扫描到 {found_power_figures} 个样本量及功效分析图表并已添加到 figures_list。")

    except subprocess.CalledProcessError as e:
        logger.error(f"执行样本量及功效分析脚本 {power_analysis_script_path} 失败. 返回码: {e.returncode}")
        logger.error(f"标准输出:\n{e.stdout}")
        logger.error(f"标准错误:\n{e.stderr}")
    except FileNotFoundError:
        logger.error(f"样本量及功效分析脚本 {power_analysis_script_path} 未找到。请确保路径正确。")
    except Exception as e:
        logger.error(f"加载样本量及功效分析脚本输出时发生意外错误: {e}", exc_info=True)

    # --- Run Vascular Age Analysis ---
    vascular_age_script_path = os.path.join("scripts", "temp_analysis", "vascular_age_analysis.py")
    vascular_age_output_base_dir = os.path.join("output", "figures", "other") # As per script's default output

    logger.info(f"执行血管年龄分析脚本: {vascular_age_script_path}...")
    try:
        process = subprocess.run(['python3', vascular_age_script_path], capture_output=True, text=True, check=True)
        logger.info("血管年龄分析脚本执行成功。")
        if process.stderr:
            logger.warning(f"血管年龄分析脚本标准错误输出:\n{process.stderr}")
        # We might need to parse stdout for model performance if not saved to a file.
        # For now, focusing on figures and the Excel file.

        # Load vascular age statistics from Excel
        vascular_stats_excel_path = os.path.join(vascular_age_output_base_dir, "vascular_age_statistics.xlsx")
        if os.path.exists(vascular_stats_excel_path):
            excel_data = pd.read_excel(vascular_stats_excel_path, sheet_name=None) # Read all sheets
            if "年龄组统计" in excel_data:
                mock_analysis_results['vascular_age_by_age_group_stats'] = excel_data["年龄组统计"]
            if "血管状态分布" in excel_data:
                mock_analysis_results['vascular_age_status_distribution'] = excel_data["血管状态分布"]
            if "性别分组统计" in excel_data:
                mock_analysis_results['vascular_age_by_gender_stats'] = excel_data["性别分组统计"]
            mock_analysis_results['vascular_age_excel_path'] = vascular_stats_excel_path
            logger.info(f"已加载血管年龄统计Excel数据: {vascular_stats_excel_path}")
        else:
            logger.warning(f"血管年龄统计Excel文件 {vascular_stats_excel_path} 未找到。")

        # Scan for vascular age figures
        vascular_age_figure_names = [
            "vascular_age_feature_importance.png",
            "vascular_age_vs_real_age_scatter.png",
            "vascular_age_difference_distribution.png",
            "vascular_age_by_age_group.png"
        ]
        found_vascular_age_figures = 0
        for fig_name in vascular_age_figure_names:
            fig_path = os.path.join(vascular_age_output_base_dir, fig_name)
            if os.path.exists(fig_path):
                caption = fig_name.replace(".png", "").replace("_", " ").capitalize()
                relative_fig_path = os.path.relpath(fig_path, os.getcwd())
                mock_figures_list.append({'path': relative_fig_path, 'caption': caption})
                found_vascular_age_figures += 1
            else:
                logger.warning(f"血管年龄分析图表 {fig_path} 未找到。")
        logger.info(f"扫描到 {found_vascular_age_figures} 个血管年龄分析图表并已添加到 figures_list。")

    except subprocess.CalledProcessError as e:
        logger.error(f"执行血管年龄分析脚本 {vascular_age_script_path} 失败. 返回码: {e.returncode}")
        # logger.error(f"标准输出:\n{e.stdout}") # Potentially very long
        logger.error(f"标准错误:\n{e.stderr}")
    except FileNotFoundError:
        logger.error(f"血管年龄分析脚本 {vascular_age_script_path} 未找到。请确保路径正确。")
    except Exception as e:
        logger.error(f"加载血管年龄分析脚本输出时发生意外错误: {e}", exc_info=True)

    # --- Run Detailed Data Cleaning Analysis ---
    detailed_cleaning_script_path = os.path.join("scripts", "temp_analysis", "data_cleaning_analysis.py")
    detailed_cleaning_figures_dir = os.path.join("output", "figures", "other") 
    # Excel files are saved to output/tables by the script

    logger.info(f"执行详细数据清洗分析脚本: {detailed_cleaning_script_path}...")
    parsed_detailed_missing_excel = None
    parsed_tabular_report_excel = None
    try:
        process = subprocess.run(['python3', detailed_cleaning_script_path], capture_output=True, text=True, check=True)
        logger.info("详细数据清洗分析脚本执行成功。")
        if process.stderr:
            logger.warning(f"详细数据清洗分析脚本标准错误输出:\n{process.stderr}")
        
        # Parse stdout for the two Excel file paths
        # Expected: "缺失值分析结果: path/to/数据缺失与清洗分析_timestamp.xlsx"
        # Expected: "数据清洗报告: path/to/数据清洗报告_timestamp.xlsx"
        for line in process.stdout.splitlines():
            if "缺失值分析结果:" in line and ".xlsx" in line:
                parsed_detailed_missing_excel = line.split("缺失值分析结果:")[-1].strip()
                if parsed_detailed_missing_excel == '未生成': parsed_detailed_missing_excel = None
            elif "数据清洗报告:" in line and ".xlsx" in line:
                parsed_tabular_report_excel = line.split("数据清洗报告:")[-1].strip()
                if parsed_tabular_report_excel == '未生成': parsed_tabular_report_excel = None
        
        if parsed_detailed_missing_excel and os.path.exists(parsed_detailed_missing_excel):
            mock_analysis_results['detailed_missing_analysis_excel_path'] = parsed_detailed_missing_excel
            logger.info(f"详细缺失分析Excel已记录: {parsed_detailed_missing_excel}")
        else:
            logger.warning(f"详细缺失分析Excel路径 ({parsed_detailed_missing_excel}) 未找到或无法解析。")

        if parsed_tabular_report_excel and os.path.exists(parsed_tabular_report_excel):
            mock_analysis_results['tabular_cleaning_report_excel_path'] = parsed_tabular_report_excel
            logger.info(f"表格化清洗报告Excel已记录: {parsed_tabular_report_excel}")
        else:
            logger.warning(f"表格化清洗报告Excel路径 ({parsed_tabular_report_excel}) 未找到或无法解析。")

        # Scan for detailed cleaning figures
        detailed_cleaning_figure_patterns = [
            os.path.join(detailed_cleaning_figures_dir, "raw_data_missing_rate_comparison.png"),
            os.path.join(detailed_cleaning_figures_dir, "cleaning_effect_*.png"),
            os.path.join(detailed_cleaning_figures_dir, "top_missing_variables_heatmap.png")
        ]
        found_detailed_cleaning_figures = 0
        for pattern in detailed_cleaning_figure_patterns:
            for fig_path in glob.glob(pattern):
                filename = os.path.basename(fig_path)
                caption = filename.replace(".png", "").replace("_", " ").replace("detailed cleaning ", "").capitalize()
                relative_fig_path = os.path.relpath(fig_path, os.getcwd())
                mock_figures_list.append({'path': relative_fig_path, 'caption': caption})
                found_detailed_cleaning_figures += 1
        logger.info(f"扫描到 {found_detailed_cleaning_figures} 个详细清洗分析图表并已添加到 figures_list。")

    except subprocess.CalledProcessError as e:
        logger.error(f"执行详细数据清洗分析脚本 {detailed_cleaning_script_path} 失败. 返回码: {e.returncode}")
        logger.error(f"标准错误:\n{e.stderr}")
    except FileNotFoundError:
        logger.error(f"详细数据清洗分析脚本 {detailed_cleaning_script_path} 未找到。请确保路径正确。")
    except Exception as e:
        logger.error(f"加载详细数据清洗分析脚本输出时发生意外错误: {e}", exc_info=True)


    # --- Run Temporary PWV Analysis (temp_pwv_analysis.py) ---
    temp_pwv_script_path = os.path.join("scripts", "temp_analysis", "temp_pwv_analysis.py")
    temp_pwv_script_output_dir = os.path.join("scripts", "temp_analysis") # Script saves figures here
    final_temp_pwv_figures_dir = os.path.join("output", "figures", "pwv_internal_comparison")
    os.makedirs(final_temp_pwv_figures_dir, exist_ok=True)

    logger.info(f"执行临时PWV分析脚本: {temp_pwv_script_path}...")
    captured_temp_pwv_stats = {} # Placeholder for parsed stats
    try:
        process = subprocess.run(['python3', temp_pwv_script_path], capture_output=True, text=True, check=True)
        logger.info("临时PWV分析脚本执行成功。")
        if process.stderr:
            logger.warning(f"临时PWV分析脚本标准错误输出:\n{process.stderr}")

        # Attempt to parse key stats from stdout (this will be very basic and fragile)
        # Example: Wilcoxon p-value for baPWV R vs L, and Spearman r for cfPWV vs baPWV_avg
        # "Wilcoxon符号秩检验: 统计量=... P值=0.0020"
        # "  cfPWV-速度m/s vs baPWV-平均速度m/s: ... Spearman: r=0.750 (p=0.000)"
        for line in process.stdout.splitlines():
            if "Wilcoxon符号秩检验:" in line and "P值=" in line:
                try:
                    pval_str = line.split("P值=")[-1].strip()
                    captured_temp_pwv_stats['bapwv_rl_diff_pval'] = float(pval_str)
                except:
                    logger.warning(f"无法解析Wilcoxon P值从: {line}")
            if "cfPWV-速度m/s vs baPWV-平均速度m/s:" in process.stdout: # Check if the full line is in stdout
                 # More complex parsing needed here if we are to get the r value specifically for spearman
                 # This is a very simplified check
                 if "Spearman: r=" in process.stdout and "(p=" in process.stdout:
                    try:
                        # This is still too simplistic and likely to fail or grab wrong values
                        # Consider modifying the source script to output JSON for stats.
                        pass # captured_temp_pwv_stats['cfpwv_vs_bapwv_avg_spearman_r'] = ...
                    except:
                        logger.warning("无法从stdout解析cfPWV vs baPWV平均 Spearman r")
        
        if captured_temp_pwv_stats:
            mock_analysis_results['temp_pwv_analysis_stats'] = captured_temp_pwv_stats
            logger.info(f"已捕获部分临时PWV分析统计数据: {captured_temp_pwv_stats}")
        else:
            logger.info("未能从临时PWV分析脚本的stdout中捕获关键统计数据。")

        # Move figures and add to list
        temp_pwv_figure_names = [
            "pwv_distributions_boxplot.png",
            "cfpwv_vs_bapwv_scatter.png"
        ]
        found_temp_pwv_figures = 0
        for fig_name in temp_pwv_figure_names:
            source_fig_path = os.path.join(temp_pwv_script_output_dir, fig_name)
            dest_fig_path = os.path.join(final_temp_pwv_figures_dir, fig_name)
            if os.path.exists(source_fig_path):
                try:
                    import shutil # Import here as it's a standard library
                    shutil.move(source_fig_path, dest_fig_path)
                    caption = fig_name.replace(".png", "").replace("_", " ").capitalize()
                    relative_fig_path = os.path.relpath(dest_fig_path, os.getcwd())
                    mock_figures_list.append({'path': relative_fig_path, 'caption': caption})
                    found_temp_pwv_figures += 1
                    logger.info(f"已移动并记录图表: {dest_fig_path}")
                except Exception as e:
                    logger.error(f"移动图表 {source_fig_path} 到 {dest_fig_path} 失败: {e}") 
            else:
                logger.warning(f"临时PWV分析图表 {source_fig_path} 未找到。")
        logger.info(f"处理了 {found_temp_pwv_figures} 个临时PWV分析图表。")

    except subprocess.CalledProcessError as e:
        logger.error(f"执行临时PWV分析脚本 {temp_pwv_script_path} 失败. 返回码: {e.returncode}")
        logger.error(f"标准错误:\n{e.stderr}")
    except FileNotFoundError:
        logger.error(f"临时PWV分析脚本 {temp_pwv_script_path} 未找到。请确保路径正确。")
    except Exception as e:
        logger.error(f"加载临时PWV分析脚本输出时发生意外错误: {e}", exc_info=True)


    logger.info(f"最终 mock_analysis_results 包含键: {list(mock_analysis_results.keys())}")
    logger.info(f"最终 mock_figures_list 包含 {len(mock_figures_list)} 个图表.")

    # Dummy data for testing other parts - can be progressively replaced
    mock_analysis_results.update({
        'basic_stats': pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]}),
        'correlation_analysis': pd.DataFrame({'var1': ['A', 'B'], 'var2': ['C', 'D'], 'r': [0.5, -0.3]}),
        'pwv_stats_by_age_group': pd.DataFrame({
            '年龄组': ['<40', '40-60', '>60'],
            'PWV均值': [1000, 1250, 1500],
            'PWV标准差': [150, 180, 220]
        }),
    })
    # Placeholder for figures_list, will be populated by subgroup analysis next
    mock_figures_list.extend([
        {'path': 'output/figures/distribution/distribution_age.png', 'caption': '年龄分布图 (示例)'},
        {'path': 'output/figures/correlation/correlation_heatmap.png', 'caption': '相关性热图 (示例)'}
    ])
    
    # Create dummy figure files for testing if they don't exist
    # to avoid errors during rendering if image embedding is attempted.
    dummy_figure_dir = "output/figures/distribution"
    os.makedirs(dummy_figure_dir, exist_ok=True)
    dummy_corr_dir = "output/figures/correlation"
    os.makedirs(dummy_corr_dir, exist_ok=True)

    if not os.path.exists("output/figures/distribution/distribution_age.png"):
        with open("output/figures/distribution/distribution_age.png", "w") as f:
            f.write("dummy png content") # Not a real image, just for path existence
    if not os.path.exists("output/figures/correlation/correlation_heatmap.png"):
        with open("output/figures/correlation/correlation_heatmap.png", "w") as f:
            f.write("dummy png content")


    mock_config = {
        "report_title": "PWV数据分析综合报告 (测试)",
        "base_output_dir": "output",
        "reports_subdir": "reports",
        "tables_subdir": "tables",
        "figures_subdir": "figures", # Used by composer
        "markdown_filename": "Test_PWV_Report.md",
        "html_filename": "Test_PWV_Report.html",
        "word_filename": "Test_PWV_Report.docx",
        "excel_filename": "Test_PWV_Report_Tables.xlsx",
        "image_base_dir_markdown": "output/figures", # For markdown renderer, relative path to MD file
        "image_base_dir_html": "../figures",      # For HTML renderer, relative path to HTML file
                                                  # This might need adjustment based on actual output structure
    }
    
    # Ensure the base output and subdirectories exist for the test
    os.makedirs(os.path.join(mock_config["base_output_dir"], mock_config["reports_subdir"]), exist_ok=True)
    os.makedirs(os.path.join(mock_config["base_output_dir"], mock_config["tables_subdir"]), exist_ok=True)
    
    logger.info("开始测试 generate_all_reports...")
    generate_all_reports(
        data=mock_df, 
        analysis_results=mock_analysis_results, 
        figures_list=mock_figures_list, 
        config=mock_config,
        data_cleaning_summary_md=mock_data_cleaning_md # Pass the mock cleaning MD
    )
    logger.info("测试 generate_all_reports 完成。请检查 'output' 目录中的测试报告文件。") 