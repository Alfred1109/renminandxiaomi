import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager
from matplotlib.font_manager import FontProperties
import os
from statsmodels.stats.multicomp import pairwise_tukeyhsd
# from statsmodels.formula.api import ols # Not strictly needed for current anova_lm usage
# from statsmodels.stats.anova import anova_lm # Not strictly needed for current f_oneway usage

# --- Configuration ---
DATA_FILE_PATH = 'docs/excel/pwv数据采集表 -去公式.xlsx'
OUTPUT_DIR = 'output/subgroup_analysis/' # For original plots and summary CSV
OTHER_FIGURES_DIR = 'output/figures/other/' # For new, richer charts

# Columns needed for target variables
COL_CFPWV = 'cfPWV-速度m/s'
COL_BAPWV_R = 'baPWV-右侧-速度m/s'
COL_BAPWV_L = 'baPWV-左侧-速度m/s'
COL_ABI_R_DP = 'ABI-右侧-足背指数'
COL_ABI_R_PT = 'ABI-右侧-胫后指数'
COL_ABI_L_DP = 'ABI-左侧-足背指数'
COL_ABI_L_PT = 'ABI-左侧-胫后指数'

# Assumed columns for grouping variables
COL_AGE = '基础信息-年龄'
COL_SEX = '性别'
COL_HEIGHT = '基础信息-身高'
COL_WEIGHT = '基础信息-体重'
COL_HYPERTENSION = '高血压病史'
COL_DIABETES = '糖尿病病史'

# Derived column names
TARGET_COL_BAPWV_AVG = 'baPWV-平均速度m/s'
TARGET_COL_ABI_OVERALL = 'ABI_Overall'
GROUP_COL_AGE_GROUP = '年龄组'
GROUP_COL_BMI = 'BMI'
GROUP_COL_BMI_GROUP = 'BMI_组'
GROUP_COL_HYPERTENSION_GROUP = '高血压状态'
GROUP_COL_DIABETES_GROUP = '糖尿病状态'


# Physiological ranges for cleaning (similar to temp_pwv_analysis.py)
BAPWV_LOWER_BOUND = 2.0
BAPWV_UPPER_BOUND = 35.0
ABI_LOWER_BOUND = 0.2
ABI_UPPER_BOUND = 2.5

# --- Font Setup ---
def setup_chinese_font():
    """Attempts to set up a Chinese font for Matplotlib."""
    chinese_font_paths = {
        'Noto Sans CJK SC': '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        'WenQuanYi Micro Hei': '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        'Source Han Sans SC': '/home/alfred/.fonts/SourceHanSansSC-Regular.otf',
        'Source Han Sans SC': '/usr/share/fonts/opentype/source-han-sans/SourceHanSansSC-Regular.otf',
        'WenQuanYi Zen Hei': '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
    }
    
    for font_name, font_path in chinese_font_paths.items():
        if os.path.exists(font_path):
            try:
                loaded_font_prop = FontProperties(fname=font_path)
                font_name_for_rc = loaded_font_prop.get_name() # Get the actual name Matplotlib recognizes

                # Prepend the loaded font name to the list of sans-serif fonts
                # This makes it the first choice when 'sans-serif' family is used.
                current_sans_serif = plt.rcParams['font.sans-serif']
                if font_name_for_rc not in current_sans_serif:
                    plt.rcParams['font.sans-serif'].insert(0, font_name_for_rc)
                
                # Set the default font family to use the sans-serif list (which now prioritizes our font)
                plt.rcParams['font.family'] = 'sans-serif'

                print(f"成功加载中文字体: {font_name} ({font_path}). Matplotlib将尝试使用 '{font_name_for_rc}' 作为优先sans-serif字体。")
                plt.rcParams['axes.unicode_minus'] = False
                return loaded_font_prop # Return the specific FontProperties object for direct use if needed
            except Exception as e:
                print(f"加载字体 {font_name} 失败: {e}")
    
    print("警告: 未找到可用的中文字体，图表中的中文可能显示为方块。将使用Matplotlib的默认后备字体。")
    plt.rcParams['axes.unicode_minus'] = False
    return FontProperties() # Return a default, empty FontProperties object

CHINESE_FONT = setup_chinese_font()

def set_chinese_font_properties(text_obj):
    """Applies the loaded Chinese font to a Matplotlib text object if available."""
    # Only apply if CHINESE_FONT is not the generic fallback FontProperties()
    if hasattr(CHINESE_FONT, 'get_file') and CHINESE_FONT.get_file() is not None:
         text_obj.set_fontproperties(CHINESE_FONT)
    # Otherwise, assume plt.rcParams['font.family'] will handle the font rendering.
    return text_obj

# --- Utility Functions ---
def create_output_directories(base_dir, target_cols, group_cols):
    """Creates necessary output directories."""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    for tc in target_cols:
        tc_safe_name = tc.replace('/', '_').replace('-', '_')
        path = os.path.join(base_dir, tc_safe_name)
        if not os.path.exists(path):
            os.makedirs(path)
        for gc in group_cols:
            gc_safe_name = gc.replace('/', '_').replace('-', '_')
            path_gc = os.path.join(path, gc_safe_name)
            if not os.path.exists(path_gc):
                os.makedirs(path_gc)
    print(f"输出目录已准备好: {base_dir}")

def clean_pwv_abi_value(series, lower_bound, upper_bound):
    """Cleans PWV or ABI series based on physiological bounds."""
    # Ensure the input is a Series, not a DataFrame column that might trigger CoW issues on direct assignment
    # Create a new Series from the input to ensure we are working with a copy
    series_copy = pd.Series(series.values).copy() # Ensures it's a new Series based on values
    series_numeric = pd.to_numeric(series_copy, errors='coerce')
    valid_before = series_numeric.notna().sum()
    series_numeric.loc[(series_numeric < lower_bound) | (series_numeric > upper_bound)] = np.nan
    valid_after = series_numeric.notna().sum()
    print(f"  列 {series.name}: {valid_before - valid_after} 个值因超出生理范围 ({lower_bound}-{upper_bound}) 被设为NaN。有效值: {valid_after}/{len(series)}")
    return series_numeric

# --- Data Loading and Preprocessing ---
def load_and_preprocess_data(file_path):
    """Loads and preprocesses the data."""
    print(f"正在加载数据从: {file_path}")
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"错误: 数据文件 '{file_path}' 未找到。请检查路径。")
        return None
    
    print("数据加载完成。开始预处理...")
    df_processed = df.copy()

    # Calculate baPWV-平均速度m/s
    if COL_BAPWV_R in df.columns and COL_BAPWV_L in df.columns:
        df_processed[COL_BAPWV_R] = clean_pwv_abi_value(df_processed[COL_BAPWV_R], BAPWV_LOWER_BOUND, BAPWV_UPPER_BOUND)
        df_processed[COL_BAPWV_L] = clean_pwv_abi_value(df_processed[COL_BAPWV_L], BAPWV_LOWER_BOUND, BAPWV_UPPER_BOUND)
        df_processed[TARGET_COL_BAPWV_AVG] = df_processed[[COL_BAPWV_R, COL_BAPWV_L]].mean(axis=1)
        print(f"已计算 '{TARGET_COL_BAPWV_AVG}'.")
    else:
        print(f"警告: 计算 '{TARGET_COL_BAPWV_AVG}' 所需的列 ({COL_BAPWV_R}, {COL_BAPWV_L}) 未全部找到。")
        df_processed[TARGET_COL_BAPWV_AVG] = np.nan

    # Calculate ABI_Overall
    abi_cols_present = all(col in df.columns for col in [COL_ABI_R_DP, COL_ABI_R_PT, COL_ABI_L_DP, COL_ABI_L_PT])
    if abi_cols_present:
        for col in [COL_ABI_R_DP, COL_ABI_R_PT, COL_ABI_L_DP, COL_ABI_L_PT]:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce') # Ensure numeric before max/min

        df_processed['ABI_R_temp'] = df_processed[[COL_ABI_R_DP, COL_ABI_R_PT]].max(axis=1)
        df_processed['ABI_L_temp'] = df_processed[[COL_ABI_L_DP, COL_ABI_L_PT]].max(axis=1)
        df_processed[TARGET_COL_ABI_OVERALL] = df_processed[['ABI_R_temp', 'ABI_L_temp']].min(axis=1)
        df_processed[TARGET_COL_ABI_OVERALL] = clean_pwv_abi_value(df_processed[TARGET_COL_ABI_OVERALL], ABI_LOWER_BOUND, ABI_UPPER_BOUND)
        df_processed.drop(columns=['ABI_R_temp', 'ABI_L_temp'], inplace=True)
        print(f"已计算 '{TARGET_COL_ABI_OVERALL}'.")
    else:
        print(f"警告: 计算 '{TARGET_COL_ABI_OVERALL}' 所需的ABI组件列未全部找到。")
        df_processed[TARGET_COL_ABI_OVERALL] = np.nan

    # Clean cfPWV
    if COL_CFPWV in df.columns:
         # Assuming cfPWV might share similar broad physiological range as baPWV for outlier cleaning, adjust if specific range known
        df_processed[COL_CFPWV] = clean_pwv_abi_value(df_processed[COL_CFPWV], BAPWV_LOWER_BOUND, BAPWV_UPPER_BOUND) # Using BAPWV bounds as placeholder
    else:
        print(f"警告: 目标列 '{COL_CFPWV}' 未在数据中找到。")
        df_processed[COL_CFPWV] = np.nan


    # Create 年龄组 (Age Group)
    if COL_AGE in df.columns:
        df_processed[COL_AGE] = pd.to_numeric(df_processed[COL_AGE], errors='coerce')
        age_bins = [0, 39, 49, 59, 69, np.inf]
        age_labels = ['<40岁', '40-49岁', '50-59岁', '60-69岁', '≥70岁']
        df_processed[GROUP_COL_AGE_GROUP] = pd.cut(df_processed[COL_AGE], bins=age_bins, labels=age_labels, right=True)
        print(f"已创建 '{GROUP_COL_AGE_GROUP}'.")
    else:
        print(f"警告: 年龄列 '{COL_AGE}' 未找到，无法创建年龄组。")
        df_processed[GROUP_COL_AGE_GROUP] = "N/A"

    # Handle 性别 (Sex)
    if COL_SEX in df.columns:
        # Standardize sex column if necessary, e.g., map 1/2 to 男/女
        # For now, assume it's usable as is or requires minimal mapping.
        # Example mapping: df_processed[COL_SEX] = df_processed[COL_SEX].map({1: '男', 2: '女'}).fillna('未知')
        print(f"性别列 '{COL_SEX}' 将用于分组。")
    else:
        print(f"警告: 性别列 '{COL_SEX}' 未找到。")
        df_processed[COL_SEX] = "N/A" # Placeholder if missing


    # Create BMI and BMI_组 (BMI Group)
    if COL_HEIGHT in df.columns and COL_WEIGHT in df.columns:
        df_processed[COL_HEIGHT] = pd.to_numeric(df_processed[COL_HEIGHT], errors='coerce')
        df_processed[COL_WEIGHT] = pd.to_numeric(df_processed[COL_WEIGHT], errors='coerce')
        # Assuming height is in cm, convert to meters for BMI calculation
        height_m = df_processed[COL_HEIGHT] / 100
        df_processed[GROUP_COL_BMI] = df_processed[COL_WEIGHT] / (height_m ** 2)
        bmi_bins = [0, 18.5, 23.9, 27.9, np.inf] # Chinese BMI categories
        bmi_labels = ['偏瘦 (<18.5)', '正常 (18.5-23.9)', '超重 (24-27.9)', '肥胖 (≥28)']
        df_processed[GROUP_COL_BMI_GROUP] = pd.cut(df_processed[GROUP_COL_BMI], bins=bmi_bins, labels=bmi_labels, right=False) # right=False: e.g. [0, 18.5)
        print(f"已计算 '{GROUP_COL_BMI}' 和 '{GROUP_COL_BMI_GROUP}'.")
    elif GROUP_COL_BMI in df.columns: # If BMI column already exists
        df_processed[GROUP_COL_BMI] = pd.to_numeric(df_processed[GROUP_COL_BMI], errors='coerce')
        bmi_bins = [0, 18.5, 23.9, 27.9, np.inf] # Chinese BMI categories
        bmi_labels = ['偏瘦 (<18.5)', '正常 (18.5-23.9)', '超重 (24-27.9)', '肥胖 (≥28)']
        df_processed[GROUP_COL_BMI_GROUP] = pd.cut(df_processed[GROUP_COL_BMI], bins=bmi_bins, labels=bmi_labels, right=False)
        print(f"已使用现有的 '{GROUP_COL_BMI}' 列创建 '{GROUP_COL_BMI_GROUP}'.")
    else:
        print(f"警告: 计算 BMI 所需的列 ({COL_HEIGHT}, {COL_WEIGHT}) 或直接的 BMI 列未全部找到。")
        df_processed[GROUP_COL_BMI_GROUP] = "N/A"

    # Handle 高血压状态 (Hypertension Status)
    if COL_HYPERTENSION in df.columns:
        # Example: df_processed[GROUP_COL_HYPERTENSION_GROUP] = df_processed[COL_HYPERTENSION].map({'是': '高血压', '否': '非高血压', 1: '高血压', 0: '非高血压'}).fillna('未知')
        df_processed[GROUP_COL_HYPERTENSION_GROUP] = df_processed[COL_HYPERTENSION].astype(str) # Simplistic, adjust mapping
        print(f"已处理 '{GROUP_COL_HYPERTENSION_GROUP}'.")
    else:
        print(f"警告: 高血压史列 '{COL_HYPERTENSION}' 未找到。")
        df_processed[GROUP_COL_HYPERTENSION_GROUP] = "N/A"

    # Handle 糖尿病状态 (Diabetes Status)
    if COL_DIABETES in df.columns:
        # Example: df_processed[GROUP_COL_DIABETES_GROUP] = df_processed[COL_DIABETES].map({'是': '糖尿病', '否': '非糖尿病', 1: '糖尿病', 0: '非糖尿病'}).fillna('未知')
        df_processed[GROUP_COL_DIABETES_GROUP] = df_processed[COL_DIABETES].astype(str) # Simplistic, adjust mapping
        print(f"已处理 '{GROUP_COL_DIABETES_GROUP}'.")
    else:
        print(f"警告: 糖尿病史列 '{COL_DIABETES}' 未找到。")
        df_processed[GROUP_COL_DIABETES_GROUP] = "N/A"

    print("数据预处理完成。")
    return df_processed

# --- Analysis Functions ---
def get_descriptive_stats(df, group_col, target_col):
    """Calculates descriptive statistics for target_col grouped by group_col."""
    if df[group_col].nunique() == 0 or df[target_col].isnull().all():
        return pd.DataFrame()
    
    # Explicitly set observed=False to maintain current behavior and silence FutureWarning
    # This is relevant if group_col is categorical (e.g., from pd.cut)
    if isinstance(df[group_col].dtype, pd.CategoricalDtype):
        desc_stats = df.groupby(group_col, observed=False)[target_col].agg(
            N='count',
            Mean='mean',
            Std='std',
            Median='median',
            Q1=lambda x: x.quantile(0.25),
            Q3=lambda x: x.quantile(0.75),
            Min='min',
            Max='max'
        ).reset_index()
    else:
        desc_stats = df.groupby(group_col)[target_col].agg(
            N='count',
            Mean='mean',
            Std='std',
            Median='median',
            Q1=lambda x: x.quantile(0.25),
            Q3=lambda x: x.quantile(0.75),
            Min='min',
            Max='max'
        ).reset_index()
    return desc_stats

def perform_statistical_tests(df, group_col, target_col):
    """Performs appropriate statistical tests for group differences."""
    print(f"\n--- {target_col} by {group_col} ---")
    data_to_test = df[[group_col, target_col]].dropna()
    if data_to_test.empty or data_to_test[group_col].nunique() < 2:
        print("  数据不足或分组少于2个，无法进行统计检验。")
        return None, "N/A"

    groups = [data_to_test[target_col][data_to_test[group_col] == g] for g in data_to_test[group_col].unique()]
    groups = [g for g in groups if len(g) > 1] # Remove groups with less than 2 samples for some tests

    if len(groups) < 2:
        print("  有效分组少于2个，无法进行统计检验。")
        return None, "N/A"

    num_groups = len(groups)
    test_used = ""
    p_value = np.nan
    test_result_summary = "无法执行检验"

    # Normality check (Shapiro-Wilk) for all groups
    normality_results = [stats.shapiro(g).pvalue > 0.05 for g in groups if len(g) >=3] # Shapiro needs at least 3 samples
    all_normal = all(normality_results) if normality_results else False # Default to false if no groups eligible for shapiro

    if num_groups == 2:
        if all_normal:
            # Check for homogeneity of variances (Levene test)
            levene_stat, levene_p = stats.levene(groups[0], groups[1])
            if levene_p > 0.05: # Variances are equal
                test_stat, p_value = stats.ttest_ind(groups[0], groups[1], equal_var=True)
                test_used = "Independent T-test (equal variances)"
                test_result_summary = f"T-statistic={test_stat:.3f}, P-value={p_value:.3f}"
            else: # Variances are not equal
                test_stat, p_value = stats.ttest_ind(groups[0], groups[1], equal_var=False)
                test_used = "Welch's T-test (unequal variances)"
                test_result_summary = f"T-statistic={test_stat:.3f}, P-value={p_value:.3f}"
        else:
            test_stat, p_value = stats.mannwhitneyu(groups[0], groups[1], alternative='two-sided')
            test_used = "Mann-Whitney U test"
            test_result_summary = f"U-statistic={test_stat:.3f}, P-value={p_value:.3f}"
    elif num_groups > 2:
        if all_normal:
             # Check for homogeneity of variances (Levene test)
            levene_stat, levene_p = stats.levene(*groups)
            if levene_p > 0.05: # Variances are equal
                test_stat, p_value = stats.f_oneway(*groups)
                test_used = "One-way ANOVA"
                test_result_summary = f"F-statistic={test_stat:.3f}, P-value={p_value:.3f}"
                if p_value < 0.05:
                    try:
                        tukey_results = pairwise_tukeyhsd(data_to_test[target_col], data_to_test[group_col], alpha=0.05)
                        print("\n  Tukey HSD Post-hoc Test:")
                        print(tukey_results)
                    except Exception as e:
                        print(f"  Tukey HSD 执行失败: {e}")
            else: # Variances are not equal - Welch's ANOVA (not directly in scipy, Kruskal as fallback)
                print("  数据正态但方差不齐，执行Kruskal-Wallis作为替代。")
                test_stat, p_value = stats.kruskal(*groups)
                test_used = "Kruskal-Wallis test (due to unequal variances in normal data)"
                test_result_summary = f"H-statistic={test_stat:.3f}, P-value={p_value:.3f}"

        else: # Not all normal
            test_stat, p_value = stats.kruskal(*groups)
            test_used = "Kruskal-Wallis test"
            test_result_summary = f"H-statistic={test_stat:.3f}, P-value={p_value:.3f}"
        
        if p_value < 0.05 and (not all_normal or (all_normal and levene_p <= 0.05)): # If Kruskal was used or ANOVA with unequal var fallback
             # Dunn's post-hoc for Kruskal-Wallis (requires scikit-posthocs or manual implementation)
             # For simplicity, we'll just report the main test here.
             # Advanced: from scikit_posthocs import posthoc_dunn
             # dunn_results = posthoc_dunn(data_to_test, val_col=target_col, group_col=group_col, p_adjust='bonferroni')
             # print("\n  Dunn's Post-hoc Test (Bonferroni corrected):")
             # print(dunn_results)
             print(f"  Kruskal-Wallis P-value < 0.05. 建议进行Dunn's post-hoc检验以确定具体组间差异。")


    print(f"  使用的检验: {test_used}")
    print(f"  检验结果: {test_result_summary}")
    return test_used, p_value, test_result_summary


def generate_visualizations(df, group_col, target_col, output_path_prefix, p_value=None):
    """Generates and saves boxplot and violinplot."""
    if df[group_col].nunique() == 0 or df[target_col].isnull().all() or df[group_col].nunique() < 2:
        print(f"  数据不足或分组不适用，跳过 {target_col} by {group_col} 的可视化。")
        return

    safe_target_col = target_col.replace('/', '_').replace('-', '_')
    safe_group_col = group_col.replace('/', '_').replace('-', '_')
    
    plot_title = f'{target_col} 按 {group_col} 分布'
    if p_value is not None and not np.isnan(p_value):
        plot_title += f' (P={p_value:.3f})'

    # Boxplot
    plt.figure(figsize=(10, 7))
    sns.boxplot(x=group_col, y=target_col, data=df, palette="viridis")
    # Overlay individual data points
    sns.stripplot(x=group_col, y=target_col, data=df, color='.3', jitter=True, size=3, alpha=0.5)
    title_obj = plt.title(plot_title, fontsize=16)
    set_chinese_font_properties(title_obj)
    set_chinese_font_properties(plt.xlabel(group_col, fontsize=14))
    set_chinese_font_properties(plt.ylabel(target_col, fontsize=14))
    
    # Apply font properties to tick labels explicitly
    ax = plt.gca() # Get current axes
    for tick_label in ax.get_xticklabels():
        set_chinese_font_properties(tick_label)
    for tick_label in ax.get_yticklabels():
        set_chinese_font_properties(tick_label)
    plt.xticks(rotation=45, ha='right') # Keep other properties like rotation
    # plt.yticks() # No specific properties other than font needed for yticks usually

    plt.tight_layout()
    boxplot_path = os.path.join(output_path_prefix, f"boxplot_{safe_target_col}_by_{safe_group_col}.png")
    plt.savefig(boxplot_path)
    print(f"  箱线图已保存: {boxplot_path}")
    plt.close()

    # Violinplot
    plt.figure(figsize=(10, 7))
    sns.violinplot(x=group_col, y=target_col, data=df, palette="viridis", inner="quartile")
    # Overlay individual data points
    sns.stripplot(x=group_col, y=target_col, data=df, color='.3', jitter=True, size=3, alpha=0.5)
    title_obj = plt.title(plot_title, fontsize=16)
    set_chinese_font_properties(title_obj)
    set_chinese_font_properties(plt.xlabel(group_col, fontsize=14))
    set_chinese_font_properties(plt.ylabel(target_col, fontsize=14))
    
    # Apply font properties to tick labels explicitly
    ax = plt.gca() # Get current axes
    for tick_label in ax.get_xticklabels():
        set_chinese_font_properties(tick_label)
    for tick_label in ax.get_yticklabels():
        set_chinese_font_properties(tick_label)
    plt.xticks(rotation=45, ha='right') # Keep other properties like rotation
    # plt.yticks() # No specific properties other than font needed for yticks usually

    plt.tight_layout()
    violinplot_path = os.path.join(output_path_prefix, f"violinplot_{safe_target_col}_by_{safe_group_col}.png")
    plt.savefig(violinplot_path)
    print(f"  小提琴图已保存: {violinplot_path}")
    plt.close()

# --- New Richer Visualization Functions ---
def generate_faceted_plot(df, x_group_col, y_target_col, facet_group_col, output_dir, plot_kind='box'):
    """Generates and saves a faceted plot (e.g., boxplot or violinplot)."""
    if df[x_group_col].nunique() < 1 or df[facet_group_col].nunique() < 1 or df[y_target_col].isnull().all() or df[x_group_col].dtype.name == 'category' and df[x_group_col].cat.categories.empty or df[facet_group_col].dtype.name == 'category' and df[facet_group_col].cat.categories.empty:
        print(f"  数据不足、分组无效或分类变量没有类别，跳过 {y_target_col} by {x_group_col} faceted by {facet_group_col} 的可视化。")
        return

    # Drop rows where any of the crucial columns for this plot are NaN
    plot_df = df[[x_group_col, y_target_col, facet_group_col]].dropna()
    if plot_df.empty or plot_df[x_group_col].nunique() < 1 or plot_df[facet_group_col].nunique() < 1:
        print(f"  去除缺失值后数据不足或分组无效，跳过 {y_target_col} by {x_group_col} faceted by {facet_group_col} 的可视化。")
        return

    safe_target_col = y_target_col.replace('/', '_').replace('-', '_')
    safe_x_group_col = x_group_col.replace('/', '_').replace('-', '_')
    safe_facet_group_col = facet_group_col.replace('/', '_').replace('-', '_')

    plot_filename = f"faceted_{plot_kind}_{safe_target_col}_by_{safe_x_group_col}_faceted_{safe_facet_group_col}.png"
    output_path = os.path.join(output_dir, plot_filename)

    # plt.figure(figsize=(12, 8)) # catplot creates its own figure, so managing it directly
    g = sns.catplot(x=x_group_col, y=y_target_col, col=facet_group_col, 
                    data=plot_df, kind=plot_kind, palette="viridis", 
                    sharey=False, # Allow y-axes to vary if ranges are very different
                    height=4, aspect=1.2, # Control size of each facet
                    col_wrap=3 if plot_df[facet_group_col].nunique() > 3 else None) # Wrap facets if many (e.g. >3)
    
    # Set the suptitle text
    g.fig.suptitle(f'{y_target_col} 按 {x_group_col} (分面: {facet_group_col})', y=1.03, fontsize=16)
    # Apply font properties to the actual suptitle Text object
    if hasattr(g.fig, '_suptitle') and g.fig._suptitle is not None:
        set_chinese_font_properties(g.fig._suptitle)
    
    # Overlay stripplot on each facet if it's a box or violin
    if plot_kind in ['box', 'violin']:
        g.map_dataframe(sns.stripplot, x=x_group_col, y=y_target_col, color='.3', jitter=True, size=3, alpha=0.5, dodge=True)

    for ax in g.axes.flatten():
        if ax.get_xlabel(): 
            ax.set_xlabel(ax.get_xlabel(), fontsize=12)
            set_chinese_font_properties(ax.xaxis.label)
        if ax.get_ylabel(): 
            ax.set_ylabel(ax.get_ylabel(), fontsize=12)
            set_chinese_font_properties(ax.yaxis.label)
        if ax.get_title(): 
            ax.set_title(ax.get_title(), fontsize=14)
            set_chinese_font_properties(ax.title)
        
        # Apply font properties to tick labels
        for tick_label in ax.get_xticklabels():
            set_chinese_font_properties(tick_label)
        for tick_label in ax.get_yticklabels():
            set_chinese_font_properties(tick_label)
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust layout to make space for suptitle
    plt.savefig(output_path)
    print(f"  分面图已保存: {output_path}")
    plt.close(g.fig) # Close the figure associated with catplot
    # plt.close('all') # Avoid closing other figures if any are managed differently

def generate_interaction_plot(df, x_group_col, y_target_col, hue_group_col, output_dir):
    """Generates and saves an interaction plot (pointplot)."""
    if df[x_group_col].nunique() < 1 or df[hue_group_col].nunique() < 1 or df[y_target_col].isnull().all() or df[x_group_col].dtype.name == 'category' and df[x_group_col].cat.categories.empty or df[hue_group_col].dtype.name == 'category' and df[hue_group_col].cat.categories.empty:
        print(f"  数据不足、分组无效或分类变量没有类别，跳过 {y_target_col} by {x_group_col} with hue {hue_group_col} 的交互图。")
        return

    # Drop rows where any of the crucial columns for this plot are NaN
    plot_df = df[[x_group_col, y_target_col, hue_group_col]].dropna()
    if plot_df.empty or plot_df[x_group_col].nunique() < 1 or plot_df[hue_group_col].nunique() < 1:
        print(f"  去除缺失值后数据不足或分组无效，跳过 {y_target_col} by {x_group_col} with hue {hue_group_col} 的交互图。")
        return

    safe_target_col = y_target_col.replace('/', '_').replace('-', '_')
    safe_x_group_col = x_group_col.replace('/', '_').replace('-', '_')
    safe_hue_group_col = hue_group_col.replace('/', '_').replace('-', '_')

    plot_filename = f"interaction_{safe_target_col}_by_{safe_x_group_col}_hue_{safe_hue_group_col}.png"
    output_path = os.path.join(output_dir, plot_filename)

    plt.figure(figsize=(12, 8))
    ax = sns.pointplot(x=x_group_col, y=y_target_col, hue=hue_group_col, 
                       data=plot_df, palette="viridis", dodge=True, errorbar=('ci', 95))
    
    title_text = f'{y_target_col} 按 {x_group_col} 与 {hue_group_col} 的交互效应'
    title_obj = plt.title(title_text, fontsize=16)
    set_chinese_font_properties(title_obj)
    
    xlabel_obj = plt.xlabel(x_group_col, fontsize=14)
    set_chinese_font_properties(xlabel_obj)
    ylabel_obj = plt.ylabel(y_target_col, fontsize=14)
    set_chinese_font_properties(ylabel_obj)
    
    # Apply font properties to tick labels
    for tick_label in ax.get_xticklabels():
        set_chinese_font_properties(tick_label)
    for tick_label in ax.get_yticklabels():
        set_chinese_font_properties(tick_label)
    ax.tick_params(axis='x', rotation=45) # Use ax.tick_params for rotation with pointplot's ax
    
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels: # Ensure legend is not empty
      legend_title_obj = None
      if CHINESE_FONT.get_family()[0] != 'sans-serif':
        legend_obj = plt.legend(handles=handles, labels=labels, title=hue_group_col, prop=CHINESE_FONT)
        if legend_obj.get_title(): 
            legend_title_obj = legend_obj.get_title()
            set_chinese_font_properties(legend_title_obj)
      else: # Default font for legend if no specific Chinese font
        legend_obj = plt.legend(handles=handles, labels=labels, title=hue_group_col)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"  交互图已保存: {output_path}")
    plt.close()

# --- Main Execution ---
def main():
    """Main function to run the subgroup analysis."""
    
    target_analysis_cols = [COL_CFPWV, TARGET_COL_BAPWV_AVG, TARGET_COL_ABI_OVERALL]
    # Ensure derived group column names are used here
    grouping_cols = [GROUP_COL_AGE_GROUP, COL_SEX, GROUP_COL_BMI_GROUP, GROUP_COL_HYPERTENSION_GROUP, GROUP_COL_DIABETES_GROUP]

    create_output_directories(OUTPUT_DIR, target_analysis_cols, grouping_cols)

    # Ensure the new directory for richer figures is created
    if not os.path.exists(OTHER_FIGURES_DIR):
        os.makedirs(OTHER_FIGURES_DIR)
        print(f"已创建目录: {OTHER_FIGURES_DIR}")

    df_data = load_and_preprocess_data(DATA_FILE_PATH)

    if df_data is None:
        print("数据加载失败，无法继续分析。")
        return

    summary_results = []

    for target_col in target_analysis_cols:
        if target_col not in df_data.columns or df_data[target_col].isnull().all():
            print(f"\n目标列 '{target_col}' 数据缺失或不存在，跳过其所有分组分析。")
            continue
        
        print(f"\n===== 分析目标变量: {target_col} =====")
        
        for group_col in grouping_cols:
            print(f"\n--- 按 '{group_col}' 分组进行分析 ---")
            
            if group_col not in df_data.columns or df_data[group_col].nunique() < 2 or df_data[group_col].astype(str).isin(["N/A", "nan"]).all() :
                print(f"  分组列 '{group_col}' 数据缺失、无效或分组少于2个，跳过此分组。")
                continue

            # Filter out rows where group_col or target_col is NaN for this specific analysis
            analysis_df = df_data[[group_col, target_col]].dropna()
            if analysis_df.empty or analysis_df[group_col].nunique() < 2 :
                 print(f"  在列 '{group_col}' 和 '{target_col}'去除缺失值后，数据不足或分组少于2个，跳过此分析。")
                 continue
            
            # Descriptive Statistics
            desc_stats = get_descriptive_stats(analysis_df, group_col, target_col)
            if not desc_stats.empty:
                print(f"\n描述性统计 ({target_col} by {group_col}):")
                print(desc_stats.round(2))
            else:
                print(f"  无法为 {target_col} by {group_col} 生成描述性统计。")


            # Statistical Tests
            test_used, p_value, test_summary = perform_statistical_tests(analysis_df, group_col, target_col)
            summary_results.append({
                'Target': target_col,
                'Group': group_col,
                'TestUsed': test_used,
                'P_Value': p_value,
                'TestSummary': test_summary
            })

            # Visualizations
            tc_safe_name = target_col.replace('/', '_').replace('-', '_')
            gc_safe_name = group_col.replace('/', '_').replace('-', '_')
            figure_output_path = os.path.join(OUTPUT_DIR, tc_safe_name, gc_safe_name)
            
            generate_visualizations(analysis_df, group_col, target_col, figure_output_path, p_value)
            
    print("\n\n===== 分析结果汇总 =====")
    summary_df = pd.DataFrame(summary_results)
    print(summary_df)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "subgroup_analysis_summary.csv"), index=False, encoding='utf-8-sig')
    print(f"\n统计检验汇总已保存到: {os.path.join(OUTPUT_DIR, 'subgroup_analysis_summary.csv')}")

    # --- Generate and Save Richer Charts ---
    print("\n\n===== 生成更丰富的图表 (保存于 output/figures/other/) =====")
    # Ensure df_data is not None and has data
    if df_data is None or df_data.empty:
        print("数据为空，跳过生成更丰富的图表。")
    else:
        for target_col in target_analysis_cols:
            if target_col not in df_data.columns or df_data[target_col].isnull().all():
                print(f"\n目标列 '{target_col}' 数据缺失或不存在，跳过其所有丰富图表分析。")
                continue
            print(f"\n--- 为目标变量 '{target_col}' 生成丰富图表 ---")
            
            # Iterate through unique pairs of grouping columns for x-axis and hue/facet
            # Ensure grouping_cols list is populated and has at least two distinct valid columns
            valid_grouping_cols_for_interaction = [gc for gc in grouping_cols 
                                                 if gc in df_data.columns and 
                                                    df_data[gc].nunique() >= 1 and 
                                                    not (df_data[gc].dtype.name == 'category' and df_data[gc].cat.categories.empty) and 
                                                    not df_data[gc].astype(str).isin(["N/A", "nan"]).all()]
            
            if len(valid_grouping_cols_for_interaction) < 2:
                print(f"  为 '{target_col}' 生成丰富图表至少需要2个有效的分组列，当前有效列数: {len(valid_grouping_cols_for_interaction)}。跳过。")
                continue

            for i in range(len(valid_grouping_cols_for_interaction)):
                for j in range(i + 1, len(valid_grouping_cols_for_interaction)):
                    x_group = valid_grouping_cols_for_interaction[i]
                    hue_or_facet_group = valid_grouping_cols_for_interaction[j]

                    print(f"  图表: {target_col} by {x_group} with {hue_or_facet_group}")

                    # Faceted plots (Box)
                    generate_faceted_plot(df_data, x_group_col=x_group, y_target_col=target_col, 
                                          facet_group_col=hue_or_facet_group, output_dir=OTHER_FIGURES_DIR, plot_kind='box')
                    
                    # Faceted plots (Violin)
                    generate_faceted_plot(df_data, x_group_col=x_group, y_target_col=target_col, 
                                          facet_group_col=hue_or_facet_group, output_dir=OTHER_FIGURES_DIR, plot_kind='violin')

                    # Interaction plots
                    generate_interaction_plot(df_data, x_group_col=x_group, y_target_col=target_col, 
                                              hue_group_col=hue_or_facet_group, output_dir=OTHER_FIGURES_DIR)
                    
                    # Optional: Swap roles of x_group and hue_or_facet_group for completeness if desired
                    # This would generate more plots, e.g., Faceted by A with X=B, then Faceted by B with X=A
                    print(f"  图表 (角色互换): {target_col} by {hue_or_facet_group} with {x_group}")
                    generate_faceted_plot(df_data, x_group_col=hue_or_facet_group, y_target_col=target_col, 
                                          facet_group_col=x_group, output_dir=OTHER_FIGURES_DIR, plot_kind='box')
                    generate_faceted_plot(df_data, x_group_col=hue_or_facet_group, y_target_col=target_col, 
                                          facet_group_col=x_group, output_dir=OTHER_FIGURES_DIR, plot_kind='violin')
                    generate_interaction_plot(df_data, x_group_col=hue_or_facet_group, y_target_col=target_col, 
                                              hue_group_col=x_group, output_dir=OTHER_FIGURES_DIR)

    print(f"\n全部分析完成。标准图表和汇总已保存到目录: {OUTPUT_DIR}")
    print(f"更丰富的图表已保存到目录: {OTHER_FIGURES_DIR}")

if __name__ == "__main__":
    main() 