#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据处理模块：负责加载原始PWV数据、清洗和预处理
包含数据导入、异常值处理、特征工程等功能
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import glob
import warnings
import logging # Import logging module
import json # Import json module

warnings.filterwarnings('ignore')

# Setup basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s')
logger = logging.getLogger(__name__)

# Global constants for data processing (Moved from clean_data)
RENAME_DICT = {
    # 基本信息
    '基础信息-年龄': 'age',
    '受试者-性别': 'gender',
    '基础信息-身高': 'height',
    '基础信息-体重': 'weight',
    
    # PWV相关 - standardized to snake_case
    'pwv_值': 'pwv',
    'pwv值': 'pwv',
    'pwv_value': 'pwv',
    'pwv数值': 'pwv',
    '竞品信息-脉搏波传导速度': 'pwv',
    'cfPWV-速度m/s': 'cfpwv_speed',
    'baPWV-右侧-速度m/s': 'bapwv_right_speed',
    'baPWV-左侧-速度m/s': 'bapwv_left_speed',
    
    # 血压相关 - standardized to snake_case
    '收缩压(mmhg)': 'sbp',
    '舒张压(mmhg)': 'dbp',
    '收缩压': 'sbp', 
    '舒张压': 'dbp', 
    'sbp': 'sbp', 
    'dbp': 'dbp',

    # Additional selected columns for standardization
    'cfPWV-颈动脉-SI ': 'cfpwv_carotid_si', 
    'cfPWV-颈动脉-RI ': 'cfpwv_carotid_ri', 
    '竞品信息-心率变异性': 'hrv_index',
    '肌酐（umol/L）': 'creatinine_umol_l',
    '尿素（mmol/L）': 'urea_mmol_l', 
    'CRP（mg/L）': 'crp_mg_l',
    '射血分数（%）': 'ef_percent',
    'ABI-右侧-胫后指数': 'abi_right_pt_index',

    # New additions from last round
    'cfPWV-颈动脉-DAIX': 'cfpwv_carotid_daix',
    'cfPWV-时间间隔ms': 'cfpwv_interval_ms',
    'cfPWV-距离cm': 'cfpwv_distance_cm',
    'baPWV-右侧-距离cm': 'bapwv_right_distance_cm',
    'baPWV-左侧-时间间隔ms': 'bapwv_left_interval_ms',
    'ABI-右侧-肱指数': 'abi_right_brachial_index',
    '血流速度-颈部-平均速度m/s': 'bfv_carotid_mean_speed',
    'BNP（pg/ml）': 'bnp_pg_ml',
    'WBC(10^9)': 'wbc_10_9',
    'Hb（g/L）': 'hb_g_l'
}

NUMERIC_COLS_RAW = [
    'age', 'height', 'weight', 'pwv', 
    'sbp', 'dbp', 
    'cfpwv_speed', 'bapwv_right_speed', 'bapwv_left_speed',
    'cfpwv_carotid_si', 'cfpwv_carotid_ri',
    'hrv_index', 'creatinine_umol_l', 'urea_mmol_l', 'crp_mg_l', 
    'ef_percent', 'abi_right_pt_index',
    # New additions from last round
    'cfpwv_carotid_daix', 'cfpwv_interval_ms', 'cfpwv_distance_cm',
    'bapwv_right_distance_cm', 'bapwv_left_interval_ms',
    'abi_right_brachial_index', 'bfv_carotid_mean_speed',
    'bnp_pg_ml', 'wbc_10_9', 'hb_g_l',
    # Add standardized names for other ABI measures if they are expected from raw data
    'abi_left_pt_index', 'abi_left_dp_index', 'abi_right_dp_index', 'abi_left_brachial_index'
]

# Define clinical ranges for validation (Example values, adjust as needed)
CLINICAL_RANGES = {
    'sbp': {'min': 60, 'max': 300, 'unit': 'mmHg'},
    'dbp': {'min': 30, 'max': 200, 'unit': 'mmHg'},
    'pwv': {'min': 3, 'max': 25, 'unit': 'm/s'},
    'age': {'min': 1, 'max': 110, 'unit': 'years'},
    'height': {'min': 50, 'max': 250, 'unit': 'cm'},
    'weight': {'min': 10, 'max': 300, 'unit': 'kg'},
    'hrv_index': {'min': 0, 'max': 200, 'unit': ''}, # Example, adjust
    'creatinine_umol_l': {'min': 20, 'max': 1000, 'unit': 'umol/L'}, # Example, adjust
    'urea_mmol_l': {'min': 1, 'max': 50, 'unit': 'mmol/L'}, # Example, adjust
    'crp_mg_l': {'min': 0, 'max': 300, 'unit': 'mg/L'}, # Example, adjust
    'ef_percent': {'min': 10, 'max': 90, 'unit': '%'}, # Example, adjust
    'bnp_pg_ml': {'min': 0, 'max': 5000, 'unit': 'pg/ml'}, # Example, adjust
    'wbc_10_9': {'min': 1, 'max': 50, 'unit': '10^9/L'}, # Example, adjust
    'hb_g_l': {'min': 50, 'max': 250, 'unit': 'g/L'}, # Example, adjust
    'abi_value': {'min': 0.3, 'max': 1.8, 'unit': ''} # For derived ABI
}

def find_latest_data_file(data_dir="docs/excel", file_pattern="*pwv*.xlsx"):
    """
    在指定目录中查找最新的PWV数据文件
    
    参数:
        data_dir: 数据文件目录
        file_pattern: 文件名匹配模式
    
    返回:
        最新数据文件的完整路径或None
    """
    # 确保目录存在
    if not os.path.exists(data_dir):
        print(f"警告: 目录 {data_dir} 不存在，将尝试在其他位置查找数据文件")
        # 尝试在项目根目录查找
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        alt_data_dir = os.path.join(project_root, "docs/excel")
        if os.path.exists(alt_data_dir):
            data_dir = alt_data_dir
            print(f"找到替代目录: {data_dir}")
        else:
            # 尝试在当前工作目录及其子目录中查找
            current_dir = os.getcwd()
            print(f"在当前工作目录 {current_dir} 及其子目录中查找数据文件...")
            
            # 递归查找所有excel文件
            all_xlsx_files = []
            for root, dirs, files in os.walk(current_dir):
                for file in files:
                    if file.endswith('.xlsx') and 'pwv' in file.lower() and not file.startswith('~$'):
                        all_xlsx_files.append(os.path.join(root, file))
            
            if all_xlsx_files:
                # 按修改时间排序，返回最新的
                latest_file = max(all_xlsx_files, key=os.path.getmtime)
                print(f"找到最新数据文件: {latest_file}")
                return latest_file
            else:
                print("未找到任何PWV数据文件")
                return None
    
    # 获取匹配的文件列表
    file_pattern_path = os.path.join(data_dir, file_pattern)
    files = glob.glob(file_pattern_path)
    
    # 排除临时文件（以~$开头的文件）
    files = [f for f in files if not os.path.basename(f).startswith('~$')]
    
    if not files:
        print(f"在 {data_dir} 中未找到匹配 '{file_pattern}' 的文件")
        # 尝试使用更宽泛的匹配模式
        broader_pattern = os.path.join(data_dir, "*.xlsx")
        files = glob.glob(broader_pattern)
        files = [f for f in files if not os.path.basename(f).startswith('~$')]
        if files:
            print(f"找到 {len(files)} 个Excel文件，将使用其中最新的一个")
        else:
            return None
    
    # 按修改时间排序，返回最新的
    latest_file = max(files, key=os.path.getmtime)
    print(f"找到最新数据文件: {latest_file}")
    return latest_file

def load_data(file_path=None):
    """
    加载PWV数据文件
    
    参数:
        file_path: 数据文件路径，如果为None则自动查找
    
    返回:
        加载的原始数据DataFrame或None
    """
    if file_path is None:
        file_path = find_latest_data_file()
        if file_path is None:
            return None
    
    print(f"正在加载数据文件: {file_path}")
    try:
        # 尝试读取Excel文件
        df = pd.read_excel(file_path)
        print(f"成功加载数据: {df.shape[0]} 行, {df.shape[1]} 列")
        return df
    except Exception as e:
        print(f"加载数据文件时出错: {e}")
        return None

def handle_outliers_iqr(df, column_name, cleaning_summary, lower_multiplier=1.5, upper_multiplier=1.5, strategy='nan'):
    """
    Handles outliers in a specified column using the IQR method.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        column_name (str): The name of the column to handle outliers for.
        cleaning_summary (dict): A dictionary to store cleaning process statistics.
        lower_multiplier (float): Multiplier for IQR to determine the lower bound.
        upper_multiplier (float): Multiplier for IQR to determine the upper bound.
        strategy (str): Method to handle outliers ('nan' to replace with NaN).

    Returns:
        pd.DataFrame: The DataFrame with outliers handled.
    """
    if column_name not in df.columns or not pd.api.types.is_numeric_dtype(df[column_name]):
        logger.warning(f"[IQR Outlier] Column '{column_name}' not found or not numeric. Skipping IQR outlier handling.")
        if 'iqr_outliers' not in cleaning_summary:
            cleaning_summary['iqr_outliers'] = {}
        cleaning_summary['iqr_outliers'][column_name] = {'skipped': True, 'reason': 'Not found or not numeric'}
        return df

    col_data = df[column_name].dropna()
    if len(col_data) < 5: # Need a few data points to make IQR meaningful
        logger.info(f"[IQR Outlier] Column '{column_name}' has < 5 non-NaN values. Skipping IQR outlier handling.")
        if 'iqr_outliers' not in cleaning_summary:
            cleaning_summary['iqr_outliers'] = {}
        cleaning_summary['iqr_outliers'][column_name] = {'skipped': True, 'reason': 'Insufficient data points'}
        return df

    q1 = col_data.quantile(0.25)
    q3 = col_data.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - lower_multiplier * iqr
    upper_bound = q3 + upper_multiplier * iqr

    outliers_lower = df[column_name] < lower_bound
    outliers_upper = df[column_name] > upper_bound
    num_outliers_lower = outliers_lower.sum()
    num_outliers_upper = outliers_upper.sum()
    total_outliers = num_outliers_lower + num_outliers_upper

    logger.info(f"[IQR Outlier] Column '{column_name}': Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}. Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    if total_outliers > 0:
        logger.info(f"[IQR Outlier] Column '{column_name}': Found {num_outliers_lower} lower outliers and {num_outliers_upper} upper outliers.")
        if strategy == 'nan':
            df.loc[outliers_lower | outliers_upper, column_name] = np.nan
            logger.info(f"[IQR Outlier] Column '{column_name}': Replaced {total_outliers} outliers with NaN.")
        # Future strategies like 'clip' or 'remove_row' could be added here
    else:
        logger.info(f"[IQR Outlier] Column '{column_name}': No outliers found based on IQR method.")

    if 'iqr_outliers' not in cleaning_summary:
        cleaning_summary['iqr_outliers'] = {}
    cleaning_summary['iqr_outliers'][column_name] = {
        'q1': q1, 'q3': q3, 'iqr': iqr,
        'lower_bound': lower_bound, 'upper_bound': upper_bound,
        'num_lower_outliers': num_outliers_lower,
        'num_upper_outliers': num_outliers_upper,
        'total_outliers_handled': total_outliers,
        'strategy': strategy
    }
    return df

def generate_cleaning_report(cleaning_summary):
    """
    Generates a Markdown formatted string containing the cleaning report.

    Args:
        cleaning_summary (dict): Dictionary containing statistics from the cleaning process.

    Returns:
        str: A Markdown formatted string containing the cleaning report.
    """
    report_parts = ["# 数据清洗报告\n"]

    if 'initial_rows' in cleaning_summary:
        report_parts.append(f"- 原始数据行数: {cleaning_summary['initial_rows']}\n")
    if 'final_rows' in cleaning_summary:
        report_parts.append(f"- 清洗后数据行数: {cleaning_summary['final_rows']}\n")
    if 'rows_dropped_nan_pwv' in cleaning_summary:
        report_parts.append(f"- 因关键值缺失 (如PWV) 删除的行数: {cleaning_summary['rows_dropped_nan_pwv']}\n")
    if 'rows_dropped_duplicates' in cleaning_summary:
        report_parts.append(f"- 删除的重复行数: {cleaning_summary['rows_dropped_duplicates']}\n")

    report_parts.append("\n## 列处理详情:\n")
    report_parts.append("| 列名 (原始) | 列名 (标准化) | 类型转换 | NaN处理 (前) | NaN处理 (后) | NaN填充方法 | 范围检查 | IQR异常值 |\n")
    report_parts.append("|---|---|---|---|---|---|---|---|\n")

    if 'column_processing' in cleaning_summary:
        for col_name_std, details in cleaning_summary['column_processing'].items():
            orig_name = details.get('original_name', 'N/A')
            type_conversion = "成功" if details.get('type_conversion_successful', False) else (
                "失败" if 'type_conversion_successful' in details else "未尝试/不适用") # More descriptive
            nan_before = details.get('nan_count_before', 'N/A')
            nan_after = details.get('nan_count_after_imputation', 'N/A')
            imputation = details.get('imputation_method', '无')
            
            range_check_info = "无"
            if 'range_check' in details and details['range_check'].get('checked', False):
                rc = details['range_check']
                range_check_info = f"{rc.get('outside_min', 0)} 低 / {rc.get('outside_max', 0)} 高 (范围: {rc.get('min_val', 'N/A')}-{rc.get('max_val', 'N/A')})"
            elif 'range_check' in details and not details['range_check'].get('checked', True):
                range_check_info = "未检查"
            
            iqr_info = "未应用"
            if 'iqr_outliers' in cleaning_summary and col_name_std in cleaning_summary['iqr_outliers']:
                iqr = cleaning_summary['iqr_outliers'][col_name_std]
                if iqr.get('skipped'):
                    iqr_info = f"跳过 ({iqr.get('reason', '')})"
                else:
                    iqr_info = f"处理 {iqr.get('total_outliers_handled', 0)}"
            
            report_parts.append(f"| {orig_name} | {col_name_std} | {type_conversion} | {nan_before} | {nan_after} | {imputation} | {range_check_info} | {iqr_info} |\n")

    if 'gender_mapping' in cleaning_summary:
        gm = cleaning_summary['gender_mapping']
        report_parts.append("\n## 性别字段处理:\n")
        report_parts.append(f"- '男' (或变体) 映射为 1: {gm.get('mapped_male', 0)}次\n")
        report_parts.append(f"- '女' (或变体) 映射为 0: {gm.get('mapped_female', 0)}次\n")
        report_parts.append(f"- 性别字段映射前NaN数量: {gm.get('nan_before_gender_mapping', 'N/A')}\n")
        report_parts.append(f"- 性别字段映射后NaN数量: {gm.get('nan_after_gender_mapping', 'N/A')}\n")

    report_parts.append("\n--- End of Report ---\n")
    return "\n".join(report_parts)

def clean_data(df):
    """
    清洗PWV数据
    
    参数:
        df: 原始数据DataFrame
    
    返回:
        tuple: 清洗后的DataFrame 和 清洗过程摘要字典 (cleaning_summary)
    """
    cleaning_summary = {} # Initialize cleaning summary dictionary

    if df is None or df.empty:
        logger.warning("无数据可清洗 (Input df is None or empty)")
        cleaning_summary['status'] = 'No data to clean'
        return None, cleaning_summary
    
    logger.info(f"原始数据: {df.shape[0]} 行, {df.shape[1]} 列")
    cleaning_summary['initial_rows'] = df.shape[0]
    cleaning_summary['column_processing'] = {}

    # 0. 统一列名
    original_column_names_in_df = df.columns.tolist()
    df_renamed = df.rename(columns=RENAME_DICT)
    logger.info("列名已标准化")

    # Initialize column_processing for all columns present after renaming
    # And try to determine original name
    for col_name_std in df_renamed.columns:
        cleaning_summary['column_processing'][col_name_std] = {}
        original_name = col_name_std # Default if no specific mapping found
        # Check if this col_name_std was a result of a rename operation
        # by looking through RENAME_DICT's values.
        for rn_key, rn_val in RENAME_DICT.items():
            if rn_val == col_name_std and rn_key in original_column_names_in_df:
                original_name = rn_key
                break # Found the original name that was mapped to this std_name
        cleaning_summary['column_processing'][col_name_std]['original_name'] = original_name
        # Initialize type_conversion_successful to N/A or specific default
        cleaning_summary['column_processing'][col_name_std]['type_conversion_successful'] = '未尝试/不适用' 

    df = df_renamed # Use the renamed df from now on

    # 1. 检查关键列是否存在
    for col_name_std in df.columns:
        if col_name_std in NUMERIC_COLS_RAW:
            # Convert to numeric, coercing errors
            cleaning_summary['column_processing'][col_name_std]['nan_count_before'] = df[col_name_std].isnull().sum()
            original_dtype = df[col_name_std].dtype
            
            df[col_name_std] = pd.to_numeric(df[col_name_std], errors='coerce')
            
            if df[col_name_std].isnull().sum() > cleaning_summary['column_processing'][col_name_std]['nan_count_before']:
                logger.warning(f"列 '{col_name_std}' (原名: {cleaning_summary['column_processing'][col_name_std].get('original_name', '?')}) 在转换为数值类型时引入了新的NaN值 (原类型: {original_dtype}).")
            
            if pd.api.types.is_numeric_dtype(df[col_name_std]):
                cleaning_summary['column_processing'][col_name_std]['type_conversion_successful'] = True
            else: # Failed to convert to numeric fully, or was already non-numeric and stayed that way
                cleaning_summary['column_processing'][col_name_std]['type_conversion_successful'] = False

            # Clinical range check for specific columns
            if col_name_std in CLINICAL_RANGES:
                specs = CLINICAL_RANGES[col_name_std]
                min_val, max_val = specs['min'], specs['max']
                
                # Ensure column is numeric before comparison (it should be if type_conversion_successful is True)
                if cleaning_summary['column_processing'][col_name_std]['type_conversion_successful']:
                    numeric_col_data = df[col_name_std][df[col_name_std].notna()] # Use notna() on already numeric col
                    outside_min = (numeric_col_data < min_val).sum()
                    outside_max = (numeric_col_data > max_val).sum()
                    
                    if outside_min > 0:
                        logger.warning(f"列 '{col_name_std}' 中有 {outside_min} 个值低于临床范围最小值 {min_val} {specs['unit']}")
                    if outside_max > 0:
                        logger.warning(f"列 '{col_name_std}' 中有 {outside_max} 个值高于临床范围最大值 {max_val} {specs['unit']}")
                    
                    cleaning_summary['column_processing'][col_name_std]['range_check'] = {
                        'min_val': min_val, 'max_val': max_val, 'unit': specs['unit'],
                        'outside_min': outside_min, 'outside_max': outside_max,
                        'checked': True
                    }
                else:
                    cleaning_summary['column_processing'][col_name_std]['range_check'] = {'checked': False, 'reason': '列非数值型'}
            else:
                 cleaning_summary['column_processing'][col_name_std]['range_check'] = {'checked': False, 'reason': '无预设范围'}

        elif col_name_std == 'gender':
            initial_nan_gender = df[col_name_std].isnull().sum()
            cleaning_summary['column_processing'][col_name_std]['nan_count_before'] = initial_nan_gender
            gender_map_details = {
                'mapped_male': 0, 
                'mapped_female': 0, 
                'nan_before_gender_mapping': initial_nan_gender
            }
            
            temp_gender_series = df[col_name_std].astype(str).str.lower() # Work on a string copy

            male_mask = temp_gender_series.str.contains('男', na=False) | \
                          temp_gender_series.str.fullmatch('1') | \
                          temp_gender_series.str.fullmatch('m') | \
                          temp_gender_series.str.fullmatch('male')
            female_mask = temp_gender_series.str.contains('女', na=False) | \
                            temp_gender_series.str.fullmatch('0') | \
                            temp_gender_series.str.fullmatch('f') | \
                            temp_gender_series.str.fullmatch('female')
            
            gender_map_details['mapped_male'] = male_mask.sum()
            gender_map_details['mapped_female'] = female_mask.sum()

            # Apply mapping to the original DataFrame column
            df.loc[male_mask, col_name_std] = 1
            df.loc[female_mask, col_name_std] = 0
            
            # For values not matching male or female, set to NaN before final numeric conversion
            # This ensures that unmapped string values become NaN and don't cause to_numeric to fail silently for the whole column
            # or get coerced to something unexpected if errors='coerce' is used broadly later.
            df.loc[~(male_mask | female_mask), col_name_std] = np.nan
            
            df[col_name_std] = pd.to_numeric(df[col_name_std], errors='coerce') # Final conversion to numeric (0, 1, NaN)
            
            cleaning_summary['column_processing'][col_name_std]['type_conversion_successful'] = pd.api.types.is_numeric_dtype(df[col_name_std])
            nan_after_gender_mapping = df[col_name_std].isnull().sum()
            gender_map_details['nan_after_gender_mapping'] = nan_after_gender_mapping
            cleaning_summary['gender_mapping'] = gender_map_details

            logger.info(f"性别字段处理完成: 男性映射为1 ({gender_map_details['mapped_male']}), 女性映射为0 ({gender_map_details['mapped_female']}). Post-map NaNs: {nan_after_gender_mapping}")
        else:
             # For columns not in NUMERIC_COLS_RAW and not 'gender', record their initial NaN count
             # And assume type conversion is not applicable or successful by default if not processed.
             cleaning_summary['column_processing'][col_name_std]['nan_count_before'] = df[col_name_std].isnull().sum()
             cleaning_summary['column_processing'][col_name_std]['type_conversion_successful'] = '不适用' # Or True if we assume they are fine
             cleaning_summary['column_processing'][col_name_std]['range_check'] = {'checked': False, 'reason': '非目标数值列'}

    # 3. 处理特定列的异常值 (在类型转换之后，通用中位数填充之前)
    # ... existing code ...

    # 4. 处理缺失值
    missing_summary = {'imputation_details': {}}
    missing_summary['total_nans_before'] = df.isnull().sum().sum()
    logger.info(f"开始处理缺失值。当前总缺失值数量: {missing_summary['total_nans_before']}")
    
    if 'pwv' in df.columns:
        rows_before_dropna_pwv = df.shape[0]
        df = df.dropna(subset=['pwv'])
        rows_after_dropna_pwv = df.shape[0]
        removed_count = rows_before_dropna_pwv - rows_after_dropna_pwv
        logger.info(f"删除PWV值缺失的行: {removed_count} 行被删除。剩余 {rows_after_dropna_pwv} 行")
        missing_summary['pwv_dropna_rows_removed'] = removed_count
    
    for col in NUMERIC_COLS_RAW:
        if col != 'pwv' and col in df.columns and df[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                num_missing = df[col].isnull().sum()
                median_value = df[col].median()
                if pd.notna(median_value):
                    df[col] = df[col].fillna(median_value)
                    logger.info(f"列 '{col}' 中 {num_missing} 个缺失值已用中位数 {median_value:.2f} 填充")
                    missing_summary['imputation_details'][col] = {'num_missing_filled': num_missing, 'median_value': median_value}
                else:
                    logger.warning(f"列 '{col}' 中位数计算为NaN，无法填充 {num_missing} 个缺失值。")
                    missing_summary['imputation_details'][col] = {'num_missing_filled': 0, 'median_value': np.nan, 'skipped_reason': 'Median is NaN'}
            else:
                logger.warning(f"列 '{col}' 非数值型，跳过中位数填充。")
                missing_summary['imputation_details'][col] = {'num_missing_filled': 0, 'skipped_reason': 'Not numeric'}
    
    pwv_related_cols_generic = [col for col in df.columns if 'pwv' in str(col).lower() or 'cfpwv' in str(col).lower() or 'bapwv' in str(col).lower()]
    for col in pwv_related_cols_generic:
        if col not in NUMERIC_COLS_RAW and col in df.columns:
            logger.info(f"额外处理PWV相关列 '{col}' (不在主要数值列列表中)")
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce')
                logger.info(f"列 '{col}' 尝试转换为数值型.")
            
            if df[col].isnull().sum() > 0 and pd.api.types.is_numeric_dtype(df[col]):
                num_missing = df[col].isnull().sum()
                median_value = df[col].median()
                if pd.notna(median_value):
                    df[col] = df[col].fillna(median_value)
                    logger.info(f"PWV相关列 '{col}' 中 {num_missing} 个缺失值已用中位数 {median_value:.2f} 填充")
                    missing_summary['imputation_details'][col] = {'num_missing_filled': num_missing, 'median_value': median_value, 'is_pwv_related_extra': True}
                else:
                    logger.warning(f"PWV相关列 '{col}' 中位数计算为NaN，无法填充 {num_missing} 个缺失值。")
                    missing_summary['imputation_details'][col] = {'num_missing_filled': 0, 'median_value': np.nan, 'skipped_reason': 'Median is NaN', 'is_pwv_related_extra': True}

    missing_summary['total_nans_after'] = df.isnull().sum().sum()
    cleaning_summary['missing_values'] = missing_summary
    logger.info(f"缺失值处理完成：总缺失值 {missing_summary['total_nans_before']} -> {missing_summary['total_nans_after']}")
    
    # 4. 标准化性别编码
    gender_summary = {}
    if 'gender' in df.columns:
        logger.info("标准化性别编码...")
        gender_map = {
            '女': 0, '0': 0, 0: 0, 'female': 0, 'f': 0, '女性': 0,
            '男': 1, '1': 1, 1: 1, 'male': 1, 'm': 1, '男性': 1
        }
        df['gender_original_str_lc'] = df['gender'].astype(str).str.lower()
        df['gender_mapped'] = df['gender_original_str_lc'].map(gender_map)
        
        unmapped_values = df[df['gender_mapped'].isnull()]['gender_original_str_lc'].unique()
        gender_summary['unmapped_values_found'] = list(unmapped_values)
        if len(unmapped_values) > 0 :
            logger.warning(f"性别编码中存在未映射的值: {list(unmapped_values)}. 这些将变为NaN。")

        df['gender'] = pd.to_numeric(df['gender_mapped'], errors='coerce')
        
        if pd.api.types.is_numeric_dtype(df['gender']):
            gender_counts = df['gender'].value_counts(dropna=False)
            gender_summary['final_distribution_str'] = str(gender_counts)
            logger.info(f"性别分布 (0=女, 1=男, NaN=未识别/缺失): \n{gender_counts}")
            
            unknown_gender_count = df['gender'].isnull().sum()
            if unknown_gender_count > 0:
                logger.warning(f"发现 {unknown_gender_count} 个未识别/缺失的性别值。")
                if not df['gender'].dropna().empty:
                    mode_gender = df['gender'].mode()[0]
                    df['gender'].fillna(mode_gender, inplace=True)
                    logger.info(f"已将未识别/缺失的性别值替换为众数: {mode_gender}")
                    gender_summary['nans_filled_with_mode'] = unknown_gender_count
                    gender_summary['mode_used'] = mode_gender
        else:
            logger.warning("性别列全为NaN或无法识别，无法用众数填充。")
        df.drop(columns=['gender_original_str_lc', 'gender_mapped'], inplace=True, errors='ignore')
    cleaning_summary['gender_standardization'] = gender_summary
    
    # 5. 创建BMI列
    bmi_summary = {}
    if 'height' in df.columns and 'weight' in df.columns:
        if pd.api.types.is_numeric_dtype(df['height']) and pd.api.types.is_numeric_dtype(df['weight']):
            logger.info("计算BMI...")
            height_mean = df['height'].mean()
            height_m = df['height'] / 100 if height_mean > 10 else df['height']
            
            df['bmi'] = df['weight'] / (height_m ** 2)
            bmi_summary['initial_mean_bmi'] = df['bmi'].mean()
            logger.info(f"BMI计算完成，平均BMI: {bmi_summary['initial_mean_bmi']:.2f} (初步)")
            
            bmi_low, bmi_high = 10, 50
            bmi_summary['low_threshold'], bmi_summary['high_threshold'] = bmi_low, bmi_high
            bmi_outliers_low = df['bmi'] < bmi_low
            bmi_outliers_high = df['bmi'] > bmi_high
            num_bmi_outliers = bmi_outliers_low.sum() + bmi_outliers_high.sum()
            bmi_summary['num_outliers_found'] = num_bmi_outliers

            if num_bmi_outliers > 0:
                logger.warning(f"检测到 {num_bmi_outliers} 个BMI值超出合理范围 ({bmi_low}-{bmi_high}). 将替换为NaN后用中位数填充。")
                df.loc[bmi_outliers_low | bmi_outliers_high, 'bmi'] = np.nan
            
            if df['bmi'].isnull().sum() > 0:
                bmi_median = df['bmi'].median()
                if pd.notna(bmi_median):
                    df['bmi'].fillna(bmi_median, inplace=True)
                    logger.info(f"BMI中的NaN值 (包括转换的异常值) 已用中位数 {bmi_median:.2f} 填充.")
                    bmi_summary['median_filled_value'] = bmi_median
                else:
                    logger.warning("BMI中位数计算为NaN，无法填充。")
            bmi_summary['status'] = 'created'
        else:
            logger.warning("身高或体重列非数值型，无法计算BMI。")
            bmi_summary['status'] = 'skipped'
            bmi_summary['reason'] = 'Height or Weight not numeric'
    else:
        bmi_summary['status'] = 'skipped'
        bmi_summary['reason'] = 'Height or Weight column missing'
    cleaning_summary['feature_creation_bmi'] = bmi_summary

    # 6. 处理血压数据和临床范围检查
    bp_handling_summary = {}
    SBP_PLAUSIBLE_MIN, SBP_PLAUSIBLE_MAX = 50, 300
    DBP_PLAUSIBLE_MIN, DBP_PLAUSIBLE_MAX = 30, 200
    
    for bp_col in ['sbp', 'dbp']:
        bp_handling_summary[bp_col] = {}
        if bp_col in df.columns and pd.api.types.is_numeric_dtype(df[bp_col]):
            plausible_min = SBP_PLAUSIBLE_MIN if bp_col == 'sbp' else DBP_PLAUSIBLE_MIN
            plausible_max = SBP_PLAUSIBLE_MAX if bp_col == 'sbp' else DBP_PLAUSIBLE_MAX
            bp_handling_summary[bp_col]['plausible_min'] = plausible_min
            bp_handling_summary[bp_col]['plausible_max'] = plausible_max
            
            outside_plausible_low = df[bp_col] < plausible_min
            outside_plausible_high = df[bp_col] > plausible_max
            num_outside_plausible = outside_plausible_low.sum() + outside_plausible_high.sum()
            bp_handling_summary[bp_col]['num_outside_plausible'] = num_outside_plausible

            if num_outside_plausible > 0:
                logger.warning(f"列 '{bp_col}': {num_outside_plausible} 个值超出临床合理范围 ({plausible_min}-{plausible_max}). Min_val: {df.loc[outside_plausible_low, bp_col].min() if outside_plausible_low.sum() > 0 else 'N/A'}, Max_val: {df.loc[outside_plausible_high, bp_col].max() if outside_plausible_high.sum() > 0 else 'N/A'}")

            specific_low = 80 if bp_col == 'sbp' else 40
            specific_high = 200 if bp_col == 'sbp' else 120
            bp_handling_summary[bp_col]['specific_low'] = specific_low
            bp_handling_summary[bp_col]['specific_high'] = specific_high
            
            outliers_specific_low = df[bp_col] < specific_low
            outliers_specific_high = df[bp_col] > specific_high
            num_specific_outliers = outliers_specific_low.sum() + outliers_specific_high.sum()
            bp_handling_summary[bp_col]['num_specific_outliers_nan'] = num_specific_outliers

            if num_specific_outliers > 0:
                logger.info(f"列 '{bp_col}': {num_specific_outliers} 个值被识别为特定范围异常值 (<{specific_low} or >{specific_high})，将设为NaN。")
                df.loc[outliers_specific_low | outliers_specific_high, bp_col] = np.nan
            
            missing_count = df[bp_col].isnull().sum()
            bp_handling_summary[bp_col]['num_nans_before_final_fill'] = missing_count
            if missing_count > 0:
                bp_median = df[bp_col].median()
                bp_handling_summary[bp_col]['median_filled_value'] = bp_median
                if pd.notna(bp_median):
                    df[bp_col].fillna(bp_median, inplace=True)
                    logger.info(f"列 '{bp_col}' 中的 {missing_count} 个NaN值 (含转换的异常值) 已用中位数 {bp_median:.2f} 填充。")
                    bp_handling_summary[bp_col]['num_nans_filled'] = missing_count
                else:
                    logger.warning(f"列 '{bp_col}' 中位数计算为NaN，无法填充 {missing_count} 个NaN值。")
                    bp_handling_summary[bp_col]['num_nans_filled'] = 0
            else:
                bp_handling_summary[bp_col]['num_nans_filled'] = 0


        elif bp_col in df.columns:
             logger.warning(f"列 '{bp_col}' 非数值型，跳过血压处理和范围检查。")
             bp_handling_summary[bp_col]['skipped_reason'] = 'Not numeric'
        else:
             bp_handling_summary[bp_col]['skipped_reason'] = 'Column not found'
    cleaning_summary['bp_outlier_handling'] = bp_handling_summary


    # 7. PWV 临床范围检查
    pwv_range_summary = {}
    PWV_PLAUSIBLE_MIN, PWV_PLAUSIBLE_MAX = 2, 30 
    pwv_range_summary['min_val'], pwv_range_summary['max_val'] = PWV_PLAUSIBLE_MIN, PWV_PLAUSIBLE_MAX
    if 'pwv' in df.columns and pd.api.types.is_numeric_dtype(df['pwv']):
        outside_pwv_low = df['pwv'] < PWV_PLAUSIBLE_MIN
        outside_pwv_high = df['pwv'] > PWV_PLAUSIBLE_MAX
        num_outside_pwv = outside_pwv_low.sum() + outside_pwv_high.sum()
        pwv_range_summary['num_outside_range'] = num_outside_pwv
        if num_outside_pwv > 0:
            logger.warning(f"列 'pwv': {num_outside_pwv} 个值超出临床合理范围 ({PWV_PLAUSIBLE_MIN}-{PWV_PLAUSIBLE_MAX}). Min_val: {df.loc[outside_pwv_low, 'pwv'].min() if outside_pwv_low.sum() > 0 else 'N/A'}, Max_val: {df.loc[outside_pwv_high, 'pwv'].max() if outside_pwv_high.sum() > 0 else 'N/A'}")
    else:
        pwv_range_summary['skipped_reason'] = 'PWV column not found or not numeric'
    cleaning_summary['pwv_range_check'] = pwv_range_summary

    # 8. 基于收缩压和舒张压创建脉压差列
    pp_summary = {}
    if 'sbp' in df.columns and 'dbp' in df.columns and \
       pd.api.types.is_numeric_dtype(df['sbp']) and pd.api.types.is_numeric_dtype(df['dbp']):
        df['脉压差'] = df['sbp'] - df['dbp']
        pp_summary['status'] = 'created'
        pp_summary['mean_value'] = df['脉压差'].mean()
        logger.info(f"已创建脉压差列. 平均值: {pp_summary['mean_value']:.2f}")
    else:
        logger.warning("无法创建脉压差，sbp或dbp列缺失或非数值型。")
        pp_summary['status'] = 'skipped'
        pp_summary['reason'] = 'sbp or dbp missing or not numeric'
    if 'derived_features' not in cleaning_summary: cleaning_summary['derived_features'] = {}
    cleaning_summary['derived_features']['pulse_pressure'] = pp_summary
    
    # Ensure 'pwv_specific_range_check_summary' is populated if pwv has range check info from the main loop.
    # This information is already part of cleaning_summary['column_processing']['pwv']['range_check'] due to the generic check.
    # This new key provides easier top-level access in the JSON summary for PWV's specific range check outcome.
    if ('pwv' in CLINICAL_RANGES and 
        'column_processing' in cleaning_summary and 
        'pwv' in cleaning_summary['column_processing'] and 
        'range_check' in cleaning_summary['column_processing']['pwv'] and 
        cleaning_summary['column_processing']['pwv']['range_check'].get('checked', False)):
        cleaning_summary['pwv_specific_range_check_summary'] = cleaning_summary['column_processing']['pwv']['range_check']
        logger.info("PWV range check details (from CLINICAL_RANGES) are available in the summary.")

    # 9. 最终数据检查和清理
    all_nan_cols_summary = {}
    empty_cols_before_drop = df.shape[1]
    all_nan_cols_list = df.columns[df.isnull().all()].tolist()
    df = df.dropna(axis=1, how='all')
    empty_cols_dropped_count = empty_cols_before_drop - df.shape[1]
    all_nan_cols_summary['count'] = empty_cols_dropped_count
    all_nan_cols_summary['columns'] = all_nan_cols_list
    if empty_cols_dropped_count > 0:
        logger.info(f"已删除 {empty_cols_dropped_count} 个全为NaN的列: {all_nan_cols_list}")
    cleaning_summary['all_nan_cols_dropped'] = all_nan_cols_summary
    
    cleaning_summary['final_rows'] = df.shape[0]
    cleaning_summary['final_cols'] = df.shape[1]
    clean_rows = df.shape[0]
    if df.shape[0] > 0 :
        retention_pct = (clean_rows / df.shape[0] * 100)
        cleaning_summary['retention_percentage'] = retention_pct
        logger.info(f"数据清洗完成: {df.shape[0]} 行 -> {clean_rows} 行 ({retention_pct:.1f}% 保留)")
    else:
        cleaning_summary['retention_percentage'] = 0.0
        logger.info(f"数据清洗完成: {df.shape[0]} 行 -> {clean_rows} 行")

    # Save cleaning_summary to JSON
    try:
        output_tables_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", "tables")
        os.makedirs(output_tables_dir, exist_ok=True)
        summary_file_path = os.path.join(output_tables_dir, f"data_cleaning_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(summary_file_path, 'w', encoding='utf-8') as f:
            json.dump(cleaning_summary, f, ensure_ascii=False, indent=4, default=str) # Use default=str for non-serializable
        logger.info(f"数据清洗摘要已保存到: {summary_file_path}")
    except Exception as e:
        logger.error(f"保存数据清洗摘要JSON文件时出错: {e}")

    return df, cleaning_summary

def create_derived_features(df):
    """
    创建派生特征，计算新指标
    
    参数:
        df: 清洗后的数据DataFrame
    
    返回:
        添加了派生特征的DataFrame
    """
    if df is None or df.empty:
        print("无数据，无法创建派生特征")
        return df
    
    print("\n开始创建派生特征...")
    
    # 1. 计算BMI（如果还没有）
    if 'bmi' not in df.columns and 'height' in df.columns and 'weight' in df.columns:
        # 检查身高单位
        height_mean = df['height'].mean()
        if height_mean > 10:  # 假设高于10的为厘米单位
            height_m = df['height'] / 100  # 转换为米
        else:
            height_m = df['height']  # 已经是米
        
        # 计算BMI = 体重(kg) / 身高(m)²
        df['bmi'] = df['weight'] / (height_m ** 2)
        print(f"已创建BMI特征，平均值: {df['bmi'].mean():.2f}")
        
        # 创建BMI分类
        df['bmi_category'] = pd.cut(
            df['bmi'],
            bins=[0, 18.5, 24, 28, 100],
            labels=['偏瘦', '正常', '超重', '肥胖']
        )
        print("已创建BMI分类特征")
    
    # 2. 计算年龄段
    if 'age' in df.columns:
        df['age_group'] = pd.cut(
            df['age'],
            bins=[0, 18, 40, 60, 75, 200],
            labels=['青少年', '青年', '中年', '老年', '高龄']
        )
        print("已创建年龄段特征")
    
    # 3. 计算脉压差（如果已有收缩压和舒张压）
    if 'sbp' in df.columns and 'dbp' in df.columns and '脉压差' not in df.columns:
        df['脉压差'] = df['sbp'] - df['dbp']
        print(f"已创建脉压差特征，平均值: {df['脉压差'].mean():.2f}")
    
    # 4. 计算血压分类（按照国际标准）
    if 'sbp' in df.columns and 'dbp' in df.columns:
        # 定义血压分类函数
        def bp_category(row):
            sbp_val = row['sbp'] # Use standardized name
            dbp_val = row['dbp'] # Use standardized name
            
            if sbp_val >= 180 or dbp_val >= 110:
                return '3级高血压'
            elif sbp_val >= 160 or dbp_val >= 100:
                return '2级高血压'
            elif sbp_val >= 140 or dbp_val >= 90:
                return '1级高血压'
            elif sbp_val >= 130 or dbp_val >= 85:
                return '高值血压'
            elif sbp_val >= 120 or dbp_val >= 80:
                return '正常高值'
            elif sbp_val >= 90 and dbp_val >= 60:
                return '正常血压'
            else:
                return '低血压'
        
        # 应用函数创建血压分类
        df['bp_category'] = df.apply(bp_category, axis=1)
        bp_counts = df['bp_category'].value_counts()
        print(f"已创建血压分类特征，分布: \n{bp_counts}")
    
    # 5. 计算PWV分类（根据临床PWV标准）
    if 'pwv' in df.columns:
        # 按年龄段划分PWV参考值
        # PWV的正常范围随年龄增加而增加
        
        # 定义PWV分类函数（简化版）
        def pwv_category(row):
            pwv = row['pwv']
            age = row['age'] if 'age' in row else 60  # 默认60岁
            
            # 依据年龄划分PWV参考范围
            if age < 30:
                if pwv < 6.5:
                    return '正常'
                elif pwv < 7.5:
                    return '边缘'
                else:
                    return '偏高'
            elif age < 40:
                if pwv < 7.0:
                    return '正常'
                elif pwv < 8.0:
                    return '边缘'
                else:
                    return '偏高'
            elif age < 50:
                if pwv < 7.5:
                    return '正常'
                elif pwv < 8.5:
                    return '边缘'
                else:
                    return '偏高'
            elif age < 60:
                if pwv < 8.0:
                    return '正常'
                elif pwv < 9.0:
                    return '边缘'
                else:
                    return '偏高'
            elif age < 70:
                if pwv < 9.0:
                    return '正常'
                elif pwv < 10.0:
                    return '边缘'
                else:
                    return '偏高'
            else:  # 70岁及以上
                if pwv < 10.0:
                    return '正常'
                elif pwv < 11.0:
                    return '边缘'
                else:
                    return '偏高'
        
        # 只有当age列存在时才应用函数
        if 'age' in df.columns:
            df['pwv_category'] = df.apply(pwv_category, axis=1)
            pwv_counts = df['pwv_category'].value_counts()
            print(f"已创建PWV分类特征，分布: \n{pwv_counts}")
    
    # 6. 创建复合风险指标
    risk_factors = []
    
    # 检查已有的风险因素列
    if 'age' in df.columns:
        df['age_risk'] = (df['age'] > 65).astype(int)
        risk_factors.append('age_risk')
    
    if 'sbp' in df.columns:
        df['sbp_risk'] = (df['sbp'] >= 140).astype(int)
        risk_factors.append('sbp_risk')
    
    if 'gender' in df.columns:
        df['gender_risk'] = (df['gender'] == 1).astype(int)  # 假设男性=1为风险因素
        risk_factors.append('gender_risk')
    
    if 'bmi' in df.columns:
        df['bmi_risk'] = (df['bmi'] >= 28).astype(int)
        risk_factors.append('bmi_risk')
    
    if 'pwv' in df.columns:
        # 不同年龄段PWV风险阈值不同
        if 'age' in df.columns:
            # 创建年龄相关的PWV风险
            conditions = [
                (df['age'] < 30) & (df['pwv'] >= 7.5),
                (df['age'] < 40) & (df['pwv'] >= 8.0),
                (df['age'] < 50) & (df['pwv'] >= 8.5),
                (df['age'] < 60) & (df['pwv'] >= 9.0),
                (df['age'] < 70) & (df['pwv'] >= 10.0),
                (df['pwv'] >= 11.0)
            ]
            df['pwv_risk'] = np.select(conditions, [1] * len(conditions), default=0)
        else:
            # 无年龄信息时使用固定阈值
            df['pwv_risk'] = (df['pwv'] >= 10).astype(int)
        
        risk_factors.append('pwv_risk')
    
    # 计算总风险得分
    if risk_factors:
        df['risk_score'] = df[risk_factors].sum(axis=1)
        print(f"已创建总风险得分，平均值: {df['risk_score'].mean():.2f}")
        
        # 创建风险等级
        df['risk_level'] = pd.cut(
            df['risk_score'],
            bins=[-1, 0, 1, 2, 100],
            labels=['低风险', '中低风险', '中高风险', '高风险']
        )
        print("已创建风险等级特征")
    
    # 7. 添加额外列标识不同类型的PWV测量
    has_cfpwv = any('cfpwv' in str(col).lower() for col in df.columns)
    has_bapwv = any('bapwv' in str(col).lower() for col in df.columns)
    
    if has_cfpwv or has_bapwv:
        if has_cfpwv:
            df['has_cfpwv'] = 1
            print("已添加cfPWV标记列")
        else:
            df['has_cfpwv'] = 0
        
        if has_bapwv:
            df['has_bapwv'] = 1
            print("已添加baPWV标记列")
        else:
            df['has_bapwv'] = 0
    
    # 8. Create ABI risk category (if ABI data is available)
    # Using 'abi_right_pt_index' as an example. This can be adapted if a different or combined ABI metric is preferred.
    abi_col_to_use = None
    if 'abi_right_pt_index' in df.columns:
        abi_col_to_use = 'abi_right_pt_index'
    elif 'abi_right_brachial_index' in df.columns: # Fallback to another ABI if primary is not present
        abi_col_to_use = 'abi_right_brachial_index'
    # Add more fallbacks if necessary

    if abi_col_to_use and pd.api.types.is_numeric_dtype(df[abi_col_to_use]):
        print(f"创建 ABI 风险分类 (基于 {abi_col_to_use})...")
        # Define ABI categories and labels
        # Ref: ACC/AHA guidelines often use <0.9 for PAD, 0.9-0.99 borderline, 1.0-1.4 normal, >1.4 non-compressible
        # Simplified for this example, can be made more granular
        bins = [-float('inf'), 0.40, 0.70, 0.90, 1.30, float('inf')]
        labels = ['重度PAD', '中度PAD', '轻度PAD', '正常/临界', '血管不可压缩'] # Order matches bins from low to high
        
        df['abi_risk_category'] = pd.cut(df[abi_col_to_use], bins=bins, labels=labels, right=True)
        
        abi_cat_counts = df['abi_risk_category'].value_counts().sort_index()
        print(f"ABI风险分类分布: \n{abi_cat_counts}")
    else:
        print(f"警告: 未找到合适的ABI数值列 ({abi_col_to_use if abi_col_to_use else 'specified ABI column'}) 用于创建风险分类，或列非数值型。")
    
    print(f"派生特征创建完成，现有特征总数: {df.shape[1]}")
    return df

def generate_data_summary(df):
    """
    生成数据摘要信息
    
    参数:
        df: 数据DataFrame
    
    返回:
        数据摘要字典
    """
    if df is None or df.empty:
        return {"error": "无数据可分析"}
    
    summary = {}
    
    # 基本信息
    summary["shape"] = df.shape
    summary["rows"] = df.shape[0]
    summary["columns"] = df.shape[1]
    
    # 数据类型分布
    dtypes = df.dtypes.value_counts()
    summary["dtypes"] = dtypes.to_dict()
    
    # 缺失值统计
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if not missing_cols.empty:
        missing_pct = missing_cols / len(df) * 100
        missing_info = pd.DataFrame({
            'missing_count': missing_cols,
            'missing_pct': missing_pct
        }).sort_values('missing_count', ascending=False)
        summary["missing"] = missing_info.to_dict()
    else:
        summary["missing"] = "无缺失值"
    
    # 主要特征统计
    key_features = ['age', 'gender', 'height', 'weight', 'bmi', 'pwv']
    key_stats = {}
    
    for feature in key_features:
        if feature in df.columns:
            # 数值型特征
            if pd.api.types.is_numeric_dtype(df[feature]):
                stats = df[feature].describe()
                key_stats[feature] = {
                    'count': stats['count'],
                    'mean': stats['mean'],
                    'std': stats['std'],
                    'min': stats['min'],
                    'q1': stats['25%'],
                    'median': stats['50%'],
                    'q3': stats['75%'],
                    'max': stats['max']
                }
            # 分类型特征
            else:
                value_counts = df[feature].value_counts()
                key_stats[feature] = value_counts.to_dict()
    
    summary["key_stats"] = key_stats
    
    # PWV统计信息（如果存在）
    if 'pwv' in df.columns:
        pwv_stats = df['pwv'].describe()
        summary["pwv_stats"] = {
            'count': pwv_stats['count'],
            'mean': pwv_stats['mean'],
            'std': pwv_stats['std'],
            'min': pwv_stats['min'],
            'q1': pwv_stats['25%'],
            'median': pwv_stats['50%'],
            'q3': pwv_stats['75%'],
            'max': pwv_stats['max']
        }
        
        # 按性别统计PWV
        if 'gender' in df.columns:
            if 0 in df['gender'].unique() and 1 in df['gender'].unique():
                pwv_by_gender = df.groupby('gender')['pwv'].describe()
                summary["pwv_by_gender"] = {
                    'female': {
                        'count': pwv_by_gender.loc[0, 'count'],
                        'mean': pwv_by_gender.loc[0, 'mean'],
                        'std': pwv_by_gender.loc[0, 'std']
                    },
                    'male': {
                        'count': pwv_by_gender.loc[1, 'count'],
                        'mean': pwv_by_gender.loc[1, 'mean'],
                        'std': pwv_by_gender.loc[1, 'std']
                    }
                }
    
    # 年龄分布
    if 'age' in df.columns:
        age_bins = [0, 18, 30, 40, 50, 60, 70, 100]
        age_labels = ['<18', '18-29', '30-39', '40-49', '50-59', '60-69', '>=70']
        age_groups = pd.cut(df['age'], bins=age_bins, labels=age_labels)
        age_dist = age_groups.value_counts().sort_index()
        summary["age_distribution"] = age_dist.to_dict()
        
        # 按年龄组统计PWV
        if 'pwv' in df.columns:
            pwv_by_age = df.groupby(age_groups)['pwv'].agg(['count', 'mean', 'std'])
            summary["pwv_by_age"] = pwv_by_age.to_dict()
    
    # 血压分布
    if 'sbp' in df.columns and 'dbp' in df.columns:
        bp_stats = {
            'sbp': df['sbp'].describe().to_dict(),
            'dbp': df['dbp'].describe().to_dict()
        }
        summary["bp_stats"] = bp_stats
        
        # 血压分类（如果存在）
        if 'bp_category' in df.columns:
            bp_dist = df['bp_category'].value_counts()
            summary["bp_distribution"] = bp_dist.to_dict()
    
    return summary

def load_and_prepare_data(data_file_path=None):
    """
    加载和准备PWV数据，包括清洗和特征工程
    
    参数:
        data_file_path (str, optional): 指定的数据文件路径。如果为None，则自动查找最新文件。
    
    返回:
        pd.DataFrame: 处理后的数据，如果出错则为None
    """
    logger.info("===== 开始加载和准备数据 =====")
    
    # 加载数据
    df = load_data(file_path=data_file_path)
    if df is None:
        logger.error("数据加载失败，无法继续处理。")
        return None
    
    logger.info(f"原始数据加载成功: {df.shape[0]} 行, {df.shape[1]} 列")
    
    # 清洗数据
    cleaned_df, cleaning_summary = clean_data(df) # Capture cleaning_summary
    if cleaned_df is None:
        logger.error("数据清洗失败或返回空数据框。")
        # Generate report even if cleaning failed partially
        if cleaning_summary:
            report_str = generate_cleaning_report(cleaning_summary)
            logger.info("\n" + report_str)
        return None
    
    # 生成并打印清洗报告
    report_str = generate_cleaning_report(cleaning_summary)
    logger.info("\n" + report_str) # Print the report to logs

    # 创建衍生特征
    final_df = create_derived_features(cleaned_df)
    if final_df is None:
        logger.error("衍生特征创建失败。")
        return None # Or return cleaned_df if derived features are optional

    # 生成数据摘要（可选，根据需要）
    # data_summary_table = generate_data_summary(final_df)
    # if data_summary_table is not None:
    #     print("\n生成的数据摘要:")
    #     print(data_summary_table)
    # else:
    #     print("数据摘要生成失败")
        
    logger.info(f"数据准备完成。最终数据: {final_df.shape[0]} 行, {final_df.shape[1]} 列")
    logger.info("===== 数据加载和准备结束 =====")
    return final_df, report_str # Return both DataFrame and cleaning report string

if __name__ == "__main__":
    # 如果作为独立脚本运行，执行默认数据加载和预处理
    df, cleaning_report = load_and_prepare_data() # Capture both outputs
    
    if df is not None:
        # 基本检查
        print("\n数据类型检查:")
        print(df.dtypes)
        
        # 查看处理后的前几行
        print("\n处理后的数据示例:")
        print(df.head())
        
        # 可选：保存处理后的数据
        try:
            output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", "tables")
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
            df.to_excel(output_file, index=False)
            print(f"\n处理后的数据已保存到: {output_file}")
        except Exception as e:
            print(f"保存处理后的数据时出错: {e}") 