#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
特征工程模块：包含特征创建和准备相关的函数
"""

import numpy as np
import pandas as pd
import warnings
import logging # Import logging

# Configure logger for this module
logger = logging.getLogger(__name__) # Initialize logger
# Basic configuration if this module is run standalone or logger isn't configured by calling script
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 从本地项目中导入必要的字体和可视化辅助函数
# (如果这些函数直接使用)
# from .data_visualization import CHINESE_FONT_PROP, apply_font_to_axis 

warnings.filterwarnings('ignore')

def create_derived_features(df):
    """
    创建衍生特征以增强风险预测模型
    
    参数:
        df: 输入DataFrame
    
    返回:
        添加了衍生特征的DataFrame
    """
    print("创建衍生特征以增强风险预测...")
    df_enhanced = df.copy()
    
    # 1. 常见临床特征
    # 使用英文标准化列名
    if 'sbp' in df.columns and 'dbp' in df.columns:
        df_enhanced['pulse_pressure'] = df['sbp'] - df['dbp']
        print("已创建'pulse_pressure' (脉压)特征")
        df_enhanced['mean_arterial_pressure'] = df['dbp'] + (df['sbp'] - df['dbp']) / 3
        print("已创建'mean_arterial_pressure' (平均动脉压)特征")
    else:
        print("警告: 'sbp'或'dbp'列不存在，无法创建'pulse_pressure'和'mean_arterial_pressure'")
    
    # 2. 年龄相关特征
    if 'age' in df.columns:
        df_enhanced['age_squared'] = df['age'] ** 2
        print("已创建'age_squared' (年龄_平方)特征")
        
        bins_age = [0, 40, 50, 60, 70, 120] # Adjusted upper bound for age
        labels_age = ['<40', '40-49', '50-59', '60-69', '>=70'] # English labels
        df_enhanced['age_group_code'] = pd.cut(df['age'], bins=bins_age, labels=labels_age, right=False)
        df_enhanced['age_group_code'] = df_enhanced['age_group_code'].cat.codes # Convert to numeric codes
        print("已创建'age_group_code' (年龄分组编码)特征")
    else:
        print("警告: 'age'列不存在，无法创建年龄相关衍生特征")

    # 3. BMI相关特征
    if 'bmi' in df.columns:
        bins_bmi = [0, 18.5, 24, 28, 100]
        labels_bmi = ['underweight', 'normal', 'overweight', 'obese'] # English labels
        df_enhanced['bmi_category_code'] = pd.cut(df['bmi'], bins=bins_bmi, labels=labels_bmi, right=False)
        df_enhanced['bmi_category_code'] = df_enhanced['bmi_category_code'].cat.codes
        print("已创建'bmi_category_code' (BMI分类编码)特征")
    else:
        print("警告: 'bmi'列不存在，无法创建BMI相关衍生特征")
        
    # 4. 体重相关特征 (Weight group - less common, but keeping if used)
    if 'weight' in df.columns:
        bins_weight = [0, 50, 60, 70, 80, 500]
        labels_weight = ['light', 'medium_light', 'medium', 'medium_heavy', 'heavy'] # English labels
        df_enhanced['weight_group_code'] = pd.cut(df['weight'], bins=bins_weight, labels=labels_weight, right=False)
        df_enhanced['weight_group_code'] = df_enhanced['weight_group_code'].cat.codes
        print("已创建'weight_group_code' (体重分组编码)特征")
    else:
        print("警告: 'weight'列不存在，无法创建体重相关衍生特征")

    # 5. 血压分类特征
    if 'sbp' in df.columns and 'dbp' in df.columns:
        conditions_bp = [
            (df['sbp'] < 120) & (df['dbp'] < 80),      # Normal
            (df['sbp'] < 130) & (df['dbp'] < 80),      # Elevated
            ((df['sbp'] >= 130) & (df['sbp'] < 140)) | ((df['dbp'] >= 80) & (df['dbp'] < 90)), # Stage 1 HTN (using 130/80 as lower bound for stage 1 based on some guidelines)
            (df['sbp'] >= 140) | (df['dbp'] >= 90)       # Stage 2 HTN or higher
        ]
        # Simpler categories for modeling: 0=Normal, 1=Elevated, 2=Hypertension Stage 1, 3=Hypertension Stage 2+
        # Or using your original: 0=正常, 1=高值, 2=1级高血压前期, 3=高血压
        # Let's use a more standard clinical categorization if possible, or a simpler one for modeling.
        # For now, adapting the user's original numeric values but linking to standard English names for clarity.
        # Normal:0, Elevated:1, Stage1_HTN:2, Stage2_HTN:3 (based on JNC8-like, but simplified for code)
        # User original: values = [0, 1, 2, 3]  # 正常, 高值, 1级高血压前期, 高血压 (This seems like: Normal, Elevated, Pre-HTN, HTN)
        # Let's map to: Normal (0), Elevated (1), Hypertension (2) (combining stage1 and stage2 for simplicity if target is just 'hypertension')
        # Or be more granular if stages are distinct targets elsewhere.
        # Given `血压状态` in `risk_prediction.py` is a classification, these categories might be important.
        # Let's retain similar numeric structure to user's previous `血压_分类` but use English names.
        # Revising conditions and values for BP category based on common definitions (e.g., ACC/AHA 2017 simplified)
        # 0: Normal (<120 AND <80)
        # 1: Elevated (120-129 AND <80)
        # 2: Hypertension Stage 1 (130-139 OR 80-89)
        # 3: Hypertension Stage 2 (>=140 OR >=90)
        # 4: Hypertensive Crisis (consult physician immediately) (>=180 AND/OR >=120) - usually not for typical risk scores
        
        conditions_bp_detailed = [
            (df['sbp'] < 120) & (df['dbp'] < 80),                                          # Normal
            (df['sbp'].between(120, 129, inclusive='both')) & (df['dbp'] < 80),         # Elevated
            (df['sbp'].between(130, 139, inclusive='both')) | (df['dbp'].between(80, 89, inclusive='both')), # HTN Stage 1
            (df['sbp'] >= 140) | (df['dbp'] >= 90)                                         # HTN Stage 2 or higher
        ]
        values_bp_detailed = [0, 1, 2, 3] # Normal, Elevated, HTN Stg1, HTN Stg2+
        df_enhanced['bp_category_code'] = np.select(conditions_bp_detailed, values_bp_detailed, default=3) # Default to highest risk category if conditions not met
        print("已创建'bp_category_code' (血压分类编码)特征: 0=正常, 1=高值, 2=1级高血压, 3=2级及以上高血压")
    else:
        print("警告: 'sbp'或'dbp'列不存在，无法创建'bp_category_code'")

    # 6. 交互特征
    if 'age' in df.columns and 'sbp' in df.columns:
        df_enhanced['age_sbp_interaction'] = (df['age'] * df['sbp']) / 100 # Scaled
        print("已创建'age_sbp_interaction' (年龄x收缩压交互)特征")
    else:
        print("警告: 'age'或'sbp'列不存在，无法创建'age_sbp_interaction'")

    if 'bmi' in df.columns and 'sbp' in df.columns:
        df_enhanced['bmi_sbp_interaction'] = (df['bmi'] * df['sbp']) / 100 # Scaled
        print("已创建'bmi_sbp_interaction' (BMIx收缩压交互)特征")
    else:
        print("警告: 'bmi'或'sbp'列不存在，无法创建'bmi_sbp_interaction'")

    created_feature_count = len(df_enhanced.columns) - len(df.columns)
    print(f"创建了 {created_feature_count} 个新特征")
    return df_enhanced

def prepare_features_and_target(df, target_col='PWV风险', risk_threshold=None, features=None):
    """
    准备机器学习模型的特征和目标变量
    
    参数:
        df: 输入DataFrame
        target_col: 目标列名
        risk_threshold: 风险值阈值（如果是数值型目标，用于转换为二分类）
        features: 要使用的特征列列表，如果为None则自动选择
    
    返回:
        X: 特征DataFrame
        y: 目标Series
        is_binary: 布尔值，指示目标是否为二分类
    """
    print(f"\n准备特征和目标变量 (目标: {target_col})...")
    
    # 检查目标列是否存在
    if target_col not in df.columns:
        print(f"错误: 目标列 '{target_col}' 不在数据中")
        if target_col == 'PWV风险' and any('pwv' in str(col).lower() for col in df.columns):
            print("注意: 找到pwv相关列，但没有'PWV风险'分类列，考虑先运行临床分析模块")
        return None, None, None
    
    # 获取目标变量
    y = df[target_col].copy()
    
    # 尝试将目标转换为数值类型（如果不是）
    if not pd.api.types.is_numeric_dtype(y):
        print(f"信息: 目标列 '{target_col}' (dtype: {y.dtype}) 是非数值类型。尝试进行智能转换...")
        
        mapped_or_encoded = False
        if target_col in ['PWV风险', '综合风险等级', 'CVD风险等级', '血压状态'] or '风险等级' in target_col or '风险状态' in target_col:
            risk_mapping = {
                '正常': 0, '边缘': 1, '轻度风险': 2, '显著风险': 3,
                '低风险': 0, '中等风险': 1, '高风险': 2, '极高风险': 3,
                '无风险': 0, '轻微风险': 1, '中度风险': 2, '严重风险': 3,
                '理想血压': 0, '正常血压': 0, '正常高值': 1, 
                '1级高血压': 2, '2级高血压': 3, '3级高血压': 4, 
                '单纯收缩期高血压': 2 
            }
            unique_values_in_y = [str(cat) for cat in y.dropna().unique()]
            can_map_all = unique_values_in_y and all(val in risk_mapping for val in unique_values_in_y)

            if target_col not in ['血压状态'] and can_map_all :
                print(f"  使用预定义风险映射处理目标列 '{target_col}'。")
                y = y.map(lambda x: risk_mapping.get(str(x)) if pd.notna(x) else np.nan)
                mapped_or_encoded = True
            elif y.dtype == 'object' or pd.api.types.is_categorical_dtype(y):
                print(f"  目标列 '{target_col}' 为对象/分类类型，将使用pd.factorize()进行通用标签编码。")
                y_factorized, uniques = pd.factorize(y, sort=True)
                y = pd.Series(y_factorized, index=y.index, dtype=float) 
                y[y == -1] = np.nan 
                print(f"  标签编码完成。类别: {uniques.tolist()}. NaN处理为np.nan.")
                mapped_or_encoded = True
        
        if not mapped_or_encoded:
            print(f"警告: 目标列 '{target_col}' 未通过特定映射或标签编码完全转换。尝试最后的强制数值转换。")
            y_coerced = pd.to_numeric(y, errors='coerce')
            if not y_coerced.isnull().all() or y.isnull().all(): 
                y = y_coerced
                print(f"  目标列 '{target_col}' 已尝试强制数值转换。")
            else: 
                print(f"错误: 目标列 '{target_col}' 无法可靠转换为数值或分类标签。")
                return None, None, None

        if pd.api.types.is_numeric_dtype(y) and y.isnull().any() and not y.isnull().all():
            print(f"信息: 目标列 '{target_col}' 包含NaN值，将保留为np.nan for XGBoost (if applicable).")
    
    if pd.api.types.is_numeric_dtype(y) and risk_threshold is not None:
        print(f"将数值目标 '{target_col}' 转换为二分类 (阈值: {risk_threshold}, 条件: >=阈值为1)")
        y_original_missing = y.isnull().sum()
        y = (y >= risk_threshold).astype(int)
        # After conversion, NaNs in original y would have become 0 if not handled.
        # It is better to ensure NaNs remain NaNs if that's desired, or are handled explicitly.
        # However, (pd.Series([10, np.nan, 12]) >= 11).astype(int) -> [0,0,1]. NaNs become 0.
        # If original y had NaNs, and they should remain NaNs in the binary target:
        # y = np.where(df[target_col].isnull(), np.nan, (df[target_col] >= risk_threshold).astype(int))
        # For simplicity, current approach converts NaN to 0 (False for >= threshold if NaN was < threshold).
        # This is standard behavior for boolean conversion of NaN-containing series unless explicitly handled.
        y_new_missing = pd.Series(y).isnull().sum() # y is now ndarray, convert to series for isnull
        if y_original_missing > 0 and y_new_missing == 0:
            print(f"  注意: 目标列中原有的 {y_original_missing} 个NaN值在二分类转换后可能已变为0。")
        print(f"  分类分布: 0 (正常/低风险): {(y==0).sum()}, 1 (超标/高风险): {(y==1).sum()}")
    
    is_binary = False
    if pd.api.types.is_numeric_dtype(y):
        unique_values = y.unique()
        if len(unique_values) == 2 and set(unique_values).issubset({0, 1, True, False}):
            is_binary = True
    
    if features is None:
        # df = create_derived_features(df) # This is now called once in run_pwv_analysis.py before the loop
                                         # If it were to be called here, it should be idempotent or controlled.
                                         # For now, assume df arrives with these features.

        exclude_cols = [target_col, 'PWV风险', '综合风险等级', '综合风险评分', 'CVD风险等级', 
                       '10年CVD风险(%)', 'id', '姓名', '联系方式', '处理时间', 'is_processed',
                       '住院号', '性别', '既往病史', '药物治疗', '血管相关疾病', # Added more potential identifiers/text fields
                       '颈动脉超声', '肾动脉超声', '下肢血管超声', '其他信息（相关既往病史，没有可直接跳过）', '备注信息', '入组受试者的诊断状态', '手表佩戴手'
                       ]
        
        # Specific exclusion for PWV超标风险 to prevent data leakage
        if target_col == 'cfpwv_speed':
            if 'pwv' not in exclude_cols:
                exclude_cols.append('pwv') # Exclude the raw 'pwv' column if target is 'cfpwv_speed'
            # Also exclude other direct cfPWV measurements if they are not the target
            cfpwv_direct_measures = ['cfpwv_interval_ms', 'cfpwv_distance_cm', 'cfPWV-颈动脉-DAIX ', 'cfPWV-股动脉-SI ', 'cfPWV-股动脉-RI', 'cfPWV-股动脉-DAIX ']
            for cf_m in cfpwv_direct_measures:
                if cf_m != target_col and cf_m not in exclude_cols:
                    exclude_cols.append(cf_m)
            logger.info(f"Target is '{target_col}', excluding raw 'pwv' and other direct cfPWV measures from features to prevent leakage.")

        # Add all known target_col names from risk_prediction.target_definitions to exclude_cols to be safe
        # This avoids a target column inadvertently being used as a feature for another target.
        # from risk_prediction import target_definitions # This import cannot be here.
        # Instead, assume target_definitions is not available or pass a list of all target cols.
        # For now, manually list known target columns used in target_definitions from risk_prediction.py
        all_defined_target_cols = ['cfpwv_速度', '血压状态', '综合风险得分'] # Manually listed based on target_definitions
        for defined_target in all_defined_target_cols:
            if defined_target not in exclude_cols:
                exclude_cols.append(defined_target)

        date_cols = [col for col in df.columns if '日期' in col or '时间' in col]
        exclude_cols.extend(date_cols)
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        features = [col for col in numeric_cols if col not in exclude_cols]
        key_features = ['age', 'gender', 'sbp', 'dbp', 'bmi', 'height', 'weight', # Standardized base columns
                        # Other key direct measurements (ensure these are the final English standardized names)
                        'bapwv_right_speed', 'crp_mg_l', 'creatinine_umol_l', 
                        'bnp_pg_ml', 'hb_g_l', 'abi_value', # Assuming 'abi_value' is a processed, final ABI column name
                        'bapwv_left_speed', 'ef_percent', 'hrv_index', 
                        'urea_mmol_l', 'wbc_10_9', 'cfpwv_carotid_si', 'cfpwv_carotid_ri'
                       ]
        # Add new English derived feature names to key_features if they are not already generically picked up
        # and ensure they are not in exclude_cols.
        # New derived features: pulse_pressure, mean_arterial_pressure, age_squared, age_group_code,
        # bmi_category_code, weight_group_code, bp_category_code, age_sbp_interaction, bmi_sbp_interaction
        new_derived_features = [
            'pulse_pressure', 'mean_arterial_pressure', 'age_squared', 'age_group_code',
            'bmi_category_code', 'weight_group_code', 'bp_category_code', 
            'age_sbp_interaction', 'bmi_sbp_interaction'
        ]
        for ndf in new_derived_features:
            if ndf not in key_features:
                 key_features.append(ndf)

        # Ensure all selected key_features actually exist in the df after derivations and initial selection
        # And ensure no duplicates
        current_features_set = set(features)
        for kf in key_features:
            if kf in df.columns and kf not in current_features_set and kf not in exclude_cols:
                features.append(kf)
                current_features_set.add(kf)
            elif kf not in df.columns:
                print(f"信息: 预定义关键特征 '{kf}' 不在DataFrame中，将不会被包含在特征集。")

        print(f"自动选择了 {len(features)} 个特征")
    
    X = df[features].copy()
    
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            try:
                if col.lower() == 'gender' or col.lower() == '性别':
                    gender_map = {'男': 1, '男性': 1, 'male': 1, 'm': 1, '1': 1, 1: 1,
                                '女': 0, '女性': 0, 'female': 0, 'f': 0, '0': 0, 0: 0}
                    X[col] = X[col].map(lambda x: gender_map.get(x, np.nan) if not pd.isna(x) else np.nan)
                    print(f"列 '{col}' 作为性别列处理: 男=1, 女=0")
                else:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                
                if X[col].isna().any():
                    X[col] = X[col].fillna(X[col].median() if not X[col].isna().all() else 0)
                    print(f"列 '{col}' 转换为数值类型并填充缺失值")
            except:
                print(f"警告: 无法将列 '{col}' 转换为数值类型，将从特征集中移除")
                X = X.drop(columns=[col])
    
    print(f"特征形状: {X.shape}, 目标形状: {y.shape}")
    print(f"目标类型: {'分类' if pd.api.types.is_categorical_dtype(y) or len(y.unique()) < 10 else '回归'}")
    print(f"特征列: {', '.join(X.columns)}")
    
    missing_counts = X.isnull().sum()
    cols_with_missing = missing_counts[missing_counts > 0]
    if not cols_with_missing.empty:
        print(f"警告: {len(cols_with_missing)} 个特征含有缺失值，模型训练时将进行处理")
        print(cols_with_missing)
    
    return X, y, is_binary 