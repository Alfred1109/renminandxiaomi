#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PWV数据分析项目主脚本
执行完整的PWV数据分析流程：数据加载、分析、可视化、报告生成。
"""

import os
import sys
import datetime
import subprocess # Keep for now, but direct function calls are preferred
import pandas as pd

# 确保项目根目录和脚本目录在Python路径中
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
scripts_dir = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

# 导入所有模块
from data_processing import load_and_prepare_data
from data_analysis import analyze_pwv_data, save_analysis_results
from data_visualization import plot_all_visualizations, create_output_dirs as create_viz_dirs
from scripts.report_generator import generate_all_reports

# 导入高级分析模块
from advanced_analysis import run_advanced_analysis
from clinical_analysis import run_clinical_analysis
from enhanced_visualization import create_all_enhanced_visualizations
from risk_prediction import run_risk_prediction, target_definitions
from feature_engineering import create_derived_features

def main_orchestrator():
    """执行完整的数据分析流程，通过直接调用模块功能。"""
    start_time = datetime.datetime.now()
    print(f"🚀 PWV数据分析流程启动于: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 0. 切换到项目根目录 (如果脚本不是从根目录运行)
    # os.chdir(project_root) # 通常在IDE中运行时不需要，但命令行执行时可能有用
    print(f"ℹ️ 当前工作目录: {os.getcwd()}")
    print(f"ℹ️ 项目根目录: {project_root}")

    # 1. 创建所有必需的输出目录 (各模块也会自行创建，这里作为总览)
    print("\n创建输出目录...")
    output_base_dir = "output"
    image_dir = os.path.join(output_base_dir, "image")
    tables_dir = os.path.join(output_base_dir, "tables")
    reports_dir = os.path.join(output_base_dir, "reports")
    
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    print(f"✅ 主要输出目录已在 '{output_base_dir}' 下创建/确认。")

    # 2. 数据加载和预处理
    print("\n" + "-"*20 + " 步骤1: 数据加载与预处理 " + "-"*20)
    data_cleaning_report_md = "" # Initialize
    try:
        # 显式传递数据文件路径，如果需要的话。否则data_processing会尝试自动查找。
        # data_file_path = os.path.join("docs", "excel", "pwv数据采集表 -去公式.xlsx")
        # df, data_cleaning_report_md = load_and_prepare_data(data_file_path)
        df, data_cleaning_report_md = load_and_prepare_data() # 使用模块内的默认或查找逻辑
        if df is None or df.empty:
            print("❌ 数据加载和预处理失败，DataFrame为空。流程中止。")
            return False
        print("✅ 数据加载和预处理完成。")
        print(f"  加载数据形状: {df.shape}")
    except FileNotFoundError as e:
        print(f"❌ 数据文件未找到: {e}。流程中止。")
        return False
    except Exception as e:
        print(f"❌ 数据加载和预处理过程中发生错误: {e}。流程中止。")
        return False

    # 3. 数据分析
    print("\n" + "-"*20 + " 步骤2: 基础数据分析 " + "-"*20)
    analysis_results = None
    try:
        analysis_results = analyze_pwv_data(df)
        if not analysis_results:
            print("⚠️ 数据分析步骤未返回结果或返回空结果。")
        else:
            print("✅ 基础数据分析完成。")
            # 保存数值分析结果到Excel
            excel_results_path = os.path.join(tables_dir, "PWV_Core_Analysis_Results.xlsx")
            save_analysis_results(analysis_results, output_path=excel_results_path)
            print(f"  核心分析结果已保存到: {excel_results_path}")
    except Exception as e:
        print(f"❌ 基础数据分析过程中发生错误: {e}。部分后续步骤可能受影响。")
        # 即使分析出错，也尝试继续生成报告（如果有一些初步结果或图表）

    # 4. 高级数据分析
    print("\n" + "-"*20 + " 步骤3: 高级数据分析 " + "-"*20)
    advanced_results = None
    try:
        advanced_results, df = run_advanced_analysis(df)
        if not advanced_results:
            print("⚠️ 高级数据分析步骤未返回结果或返回空结果。")
        else:
            print("✅ 高级数据分析完成。")
            # 保存高级分析结果到Excel
            adv_excel_path = os.path.join(tables_dir, "PWV_Advanced_Analysis_Results.xlsx")
            # 将advanced_results的每个部分保存到Excel的不同工作表
            with pd.ExcelWriter(adv_excel_path, engine='openpyxl') as writer:
                for key, value in advanced_results.items():
                    if isinstance(value, pd.DataFrame):
                        value.to_excel(writer, sheet_name=key[:31])  # Excel工作表名长度限制
                    elif isinstance(value, dict) and 'summary' in value and isinstance(value['summary'], pd.DataFrame):
                        value['summary'].to_excel(writer, sheet_name=f"{key}_summary"[:31])
                    elif isinstance(value, dict) and 'summary_stats' in value and isinstance(value['summary_stats'], pd.DataFrame):
                        value['summary_stats'].to_excel(writer, sheet_name=f"{key}_stats"[:31])
            print(f"  高级分析结果已保存到: {adv_excel_path}")
    except Exception as e:
        print(f"❌ 高级数据分析过程中发生错误: {e}。部分后续步骤可能受影响。")

    # 5. 临床风险分析
    print("\n" + "-"*20 + " 步骤4: 临床风险分析 " + "-"*20)
    clinical_results = None
    try:
        df, clinical_results = run_clinical_analysis(df)
        if not clinical_results:
            print("⚠️ 临床风险分析步骤未返回结果或返回空结果。")
        else:
            print("✅ 临床风险分析完成。")
            # 保存临床分析结果到Excel
            clinical_excel_path = os.path.join(tables_dir, "PWV_Clinical_Analysis_Results.xlsx")
            # 将clinical_results的每个部分保存到Excel的不同工作表
            with pd.ExcelWriter(clinical_excel_path, engine='openpyxl') as writer:
                for key, value in clinical_results.items():
                    if isinstance(value, pd.DataFrame):
                        value.to_excel(writer, sheet_name=key[:31])
                    elif isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, pd.DataFrame):
                                sub_value.to_excel(writer, sheet_name=f"{key}_{sub_key}"[:31])
            print(f"  临床分析结果已保存到: {clinical_excel_path}")
    except Exception as e:
        print(f"❌ 临床风险分析过程中发生错误: {e}。部分后续步骤可能受影响。")

    # NEW STEP 5.5: Create common derived features for modeling from feature_engineering
    print("\n" + "-"*20 + " 步骤5.5: 创建通用衍生特征 (供风险预测使用) " + "-"*20)
    try:
        df = create_derived_features(df.copy()) # Ensure df is updated with these common features
        print("✅ 通用衍生特征创建完成。")
    except Exception as e:
        print(f"❌ 创建通用衍生特征过程中发生错误: {e}。风险预测可能受影响。")

    # 6. 基础数据可视化
    print("\n" + "-"*20 + " 步骤6: 基础数据可视化 " + "-"*20)
    figures_list = []
    try:
        # plot_all_visualizations 现在返回生成的图表文件名列表
        figures_list = plot_all_visualizations(df, analysis_results, save_dir=image_dir)
        if not figures_list:
            print("⚠️ 未生成任何基础图表，或图表列表为空。")
        else:
            print(f"✅ 基础数据可视化完成，共生成 {len(figures_list)} 个图表。")
            print(f"  图表保存在: {image_dir}")
    except Exception as e:
        print(f"❌ 基础数据可视化过程中发生错误: {e}。报告中的图表可能不完整。")

    # 7. 增强数据可视化
    print("\n" + "-"*20 + " 步骤7: 增强数据可视化 " + "-"*20)
    enhanced_figures = []
    try:
        # 执行增强可视化
        enhanced_figures = create_all_enhanced_visualizations(df, save_dir=image_dir)
        if not enhanced_figures:
            print("⚠️ 未生成任何增强图表，或图表列表为空。")
        else:
            print(f"✅ 增强数据可视化完成，共生成 {len(enhanced_figures)} 个图表。")
            print(f"  图表保存在: {image_dir}")
            
        # 合并所有图表列表
        figures_list.extend(enhanced_figures)
    except Exception as e:
        print(f"❌ 增强数据可视化过程中发生错误: {e}。报告中的增强图表可能不完整。")

    # 8. 风险预测分析
    print("\n" + "-"*20 + " 步骤8: 风险预测分析 " + "-"*20)
    all_risk_results_collection = {} # Store results for all risk types, renamed to avoid conflict
    collected_risk_figures = [] # Renamed to avoid conflict

    risk_targets_to_run = ['PWV超标风险', '高血压风险', '高综合风险']

    for target_key_to_run in risk_targets_to_run:
        print(f"\n--- 开始对 {target_key_to_run} 进行风险预测 ---")
        try:
            current_df_for_risk = df.copy() # Use a copy to be safe
            
            target_info_check = next((t for t in target_definitions if t['key'] == target_key_to_run), None)
            if target_info_check:
                required_col_for_risk = target_info_check['col']
                if required_col_for_risk not in current_df_for_risk.columns:
                    print(f"\u26A0 警告: 目标列 '{required_col_for_risk}' (为 '{target_key_to_run}' 定义) 在DataFrame中未找到。跳过此风险预测。")
                    print(f"  可用列: {current_df_for_risk.columns.tolist()}")
                    all_risk_results_collection[target_key_to_run] = {"error": f"Missing target column: {required_col_for_risk}"}
                    continue
            else:
                print(f"\u26A0 警告: 未找到 '{target_key_to_run}' 的目标定义。跳过此风险预测。")
                all_risk_results_collection[target_key_to_run] = {"error": f"Missing target definition for {target_key_to_run}"}
                continue

            individual_risk_result = run_risk_prediction(current_df_for_risk, 
                                                         target_key=target_key_to_run, 
                                                         output_dir=output_base_dir)
            
            if individual_risk_result:
                all_risk_results_collection[target_key_to_run] = individual_risk_result
                print(f"\u2705 {target_key_to_run} 的风险预测分析完成。")
                
                model_perf_plots = individual_risk_result.get('performance_plots', [])
                shap_plots_list = individual_risk_result.get('shap_plots', [])
                
                current_target_figures = 0
                if isinstance(model_perf_plots, list):
                    collected_risk_figures.extend(model_perf_plots)
                    current_target_figures += len(model_perf_plots)
                elif model_perf_plots:
                    collected_risk_figures.append(model_perf_plots)
                    current_target_figures += 1

                if isinstance(shap_plots_list, list):
                    collected_risk_figures.extend(shap_plots_list)
                    current_target_figures += len(shap_plots_list)
                elif shap_plots_list:
                    collected_risk_figures.append(shap_plots_list)
                    current_target_figures += 1
                
                print(f"  为 {target_key_to_run} 生成了 {current_target_figures} 个图表。")

                if 'predictions' in individual_risk_result and isinstance(individual_risk_result['predictions'], pd.DataFrame):
                    # Sanitize filename
                    safe_target_key = target_key_to_run.replace(' ', '_').replace('/', '_')
                    risk_excel_path = os.path.join(tables_dir, f"{safe_target_key}_Predictions.xlsx")
                    individual_risk_result['predictions'].to_excel(risk_excel_path, index=False)
                    print(f"  风险预测结果 ({target_key_to_run}) 已保存到: {risk_excel_path}")
            else:
                print(f"\u26A0 {target_key_to_run} 的风险预测分析未返回有效结果。")
                all_risk_results_collection[target_key_to_run] = {"error": "Prediction did not return results."}

        except Exception as e:
            print(f"\u274C 在为 {target_key_to_run} 进行风险预测分析过程中发生错误: {e}")
            all_risk_results_collection[target_key_to_run] = {"error": str(e)}

    if collected_risk_figures:
        figures_list.extend(collected_risk_figures)
        print(f"  风险预测总共生成 {len(collected_risk_figures)} 个图表。")
    
    # 9. 整合分析结果
    all_results = {}
    if analysis_results:
        all_results.update(analysis_results)
    if advanced_results:
        all_results['advanced_analysis'] = advanced_results # Changed key for clarity
    if clinical_results:
        all_results['clinical_analysis'] = clinical_results # Changed key for clarity
    
    if all_risk_results_collection:
        all_results['risk_prediction_detailed'] = all_risk_results_collection
        # For backward compatibility or if some part of report expects the old structure for PWV risk type:
        if 'PWV超标风险' in all_risk_results_collection and not all_risk_results_collection['PWV超标风险'].get("error"):
             all_results['risk_prediction'] = all_risk_results_collection['PWV超标风险']

    # 10. 报告生成
    print("\n" + "-"*20 + " 步骤9: 生成分析报告 " + "-"*20)
    try:
        # 确保即使前面步骤有错误，也能尝试生成报告
        print("尝试生成PWV数据分析综合报告...")
        # 如果figures_list为空，尝试从image目录获取图表列表
        if not figures_list:
            print("注意: 图表列表为空，尝试从图表目录获取图表...")
            if os.path.exists(image_dir):
                for root, _, files in os.walk(image_dir):
                    for file in files:
                        if file.endswith(('.png', '.jpg', '.jpeg', '.svg')):
                            rel_path = os.path.relpath(os.path.join(root, file), output_base_dir)
                            figures_list.append(rel_path)
                print(f"从图表目录获取到 {len(figures_list)} 个图表文件")
        
        # 如果all_results为空，创建一个基本结构
        if not all_results:
            all_results = {}
        
        # Define report configuration for the new generator
        report_config = {
            "report_title": "PWV数据分析综合报告",
            "base_output_dir": output_base_dir,
            "reports_subdir": "reports", # Default, can be overridden by run.sh if needed
            "tables_subdir": "tables",   # Default
            "figures_subdir": "figures", # Relative to base_output_dir, used by composer
            "markdown_filename": "PWV数据分析综合报告.md",
            "html_filename": "PWV数据分析综合报告.html",
            "word_filename": "PWV数据分析综合报告.docx",
            "excel_filename": "PWV数据分析报告_表格汇总.xlsx",
            # For markdown renderer, this is relative to where the MD file will be
            # if figures are in output/figures and MD is in output/reports, then path is ../figures
            "image_base_dir_markdown": "../figures", 
            # For HTML renderer, this is relative to where the HTML file will be
            "image_base_dir_html": "../figures",
            # Add other necessary config for composer/renderers if any
            "output_figures_dir": os.path.join(output_base_dir, "figures") # Absolute or relative to project for composer
        }
        
        # generate_all_reports 会处理Markdown, Excel (从MD), Word, HTML
        generated_report_paths = generate_all_reports(
            data=df,
            analysis_results=all_results,
            figures_list=figures_list, # Changed from figures=figures_list
            config=report_config,
            data_cleaning_summary_md=data_cleaning_report_md # Pass the cleaning report
        )
        if generated_report_paths:
            print("✅ 分析报告生成完成。")
    except Exception as e:
        print(f"❌ 报告生成过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        print("尽管报告生成出现错误，流程将继续尝试完成...")
        # 返回 False，但不中止流程

    # 流程完成
    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()
    print("\n" + "=" * 60)
    print(f"🎉 PWV数据分析流程全部完成! (总耗时: {duration:.2f}秒)")
    print("-" * 25 + " 查看产出 " + "-"*25)
    print(f"- Excel表格 (核心分析): {os.path.join(tables_dir, 'PWV_Core_Analysis_Results.xlsx')}")
    print(f"- Excel表格 (高级分析): {os.path.join(tables_dir, 'PWV_Advanced_Analysis_Results.xlsx')}")
    print(f"- Excel表格 (临床分析): {os.path.join(tables_dir, 'PWV_Clinical_Analysis_Results.xlsx')}")
    print(f"- Excel表格 (风险预测): {os.path.join(tables_dir, 'PWV_Risk_Predictions.xlsx')}")
    print(f"- Excel表格 (从报告提取): {os.path.join(tables_dir, 'PWV数据分析表格汇总.xlsx')}")
    print(f"- Markdown报告: {os.path.join(reports_dir, 'PWV数据分析综合报告.md')}")
    print(f"- Word报告: {os.path.join(reports_dir, 'PWV数据分析综合报告.docx')}")
    print(f"- HTML报告: {os.path.join(reports_dir, 'PWV数据分析综合报告.html')}")
    print(f"- 图表目录: {image_dir} (共 {len(figures_list)} 张图表)")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = main_orchestrator()
    if success:
        print("\n流程成功结束。")
    else:
        print("\n流程因错误而中止或未完全成功。")
        sys.exit(1) # 退出并返回错误码 