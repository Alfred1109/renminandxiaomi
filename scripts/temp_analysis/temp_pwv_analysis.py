import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager
from matplotlib.font_manager import FontProperties
import os

# 设置中文字体
# 1. 先尝试找到系统中可用的中文字体
chinese_font_paths = {
    'Noto Sans CJK SC': '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    'WenQuanYi Micro Hei': '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
    'Source Han Sans SC': '/home/alfred/.fonts/SourceHanSansSC-Regular.otf',
    'WenQuanYi Zen Hei': '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
}

# 2. 尝试加载可用的字体文件
chinese_font = None
for font_name, font_path in chinese_font_paths.items():
    if os.path.exists(font_path):
        try:
            chinese_font = FontProperties(fname=font_path)
            print(f"成功加载中文字体: {font_name} ({font_path})")
            # 将此字体添加到Matplotlib的字体管理器中
            matplotlib.font_manager.fontManager.ttflist.append(chinese_font)
            # 找到一个有效字体后跳出循环
            break
        except Exception as e:
            print(f"加载字体 {font_name} 失败: {e}")

# 3. 如果找不到任何中文字体，给出警告
if chinese_font is None:
    print("警告: 未找到可用的中文字体，图表中的中文可能显示为方块。")
    # 尝试使用sans-serif字体作为后备
    chinese_font = FontProperties(family='sans-serif')

# 4. 设置全局字体参数
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 简单函数，用于设置图表中的中文字体
def set_chinese_font(text_obj):
    """为matplotlib文本对象设置中文字体"""
    if chinese_font is not None:
        text_obj.set_fontproperties(chinese_font)
    return text_obj

# 加载数据
try:
    df = pd.read_excel('docs/excel/pwv数据采集表 -去公式.xlsx')
except FileNotFoundError:
    print("错误: 原始数据文件 'docs/excel/pwv数据采集表 -去公式.xlsx' 未找到。")
    exit()

# 选择相关的PWV列
pwv_cols = ['cfPWV-速度m/s', 'baPWV-右侧-速度m/s', 'baPWV-左侧-速度m/s']
missing_cols = [col for col in pwv_cols if col not in df.columns]
if missing_cols:
    print(f"错误: 以下期望的PWV列在原始数据中未找到: {', '.join(missing_cols)}")
    exit()

df_pwv = df[pwv_cols].copy()

print('\n=== PWV列原始数据类型检查 ===\n')
print(df_pwv.dtypes)

# 转换为数值型，无法转换的设为NaN
print('\n=== 转换为数值型并检查非空值数量 (转换前) ===\n')
for col in pwv_cols:
    # 先记录转换前的有效值数量
    initial_valid_count = pd.to_numeric(df_pwv[col], errors='coerce').notna().sum()
    df_pwv[col] = pd.to_numeric(df_pwv[col], errors='coerce')
    print(f"列 '{col}' 初始有效值数量: {initial_valid_count} / {len(df_pwv)}")

print('\n=== 应用生理范围过滤 baPWV 值 ===\n')
# 定义生理范围
bapwv_lower_bound = 2.0
bapwv_upper_bound = 35.0

for col in ['baPWV-右侧-速度m/s', 'baPWV-左侧-速度m/s']:
    if col in df_pwv.columns:
        original_valid_count = df_pwv[col].notna().sum()
        # 标记超出范围的值
        outside_range_mask = (df_pwv[col] < bapwv_lower_bound) | (df_pwv[col] > bapwv_upper_bound)
        num_outside_range = outside_range_mask.sum()
        
        df_pwv.loc[outside_range_mask, col] = np.nan
        print(f"列 '{col}': {num_outside_range} 个值因超出生理范围 ({bapwv_lower_bound}-{bapwv_upper_bound} m/s) 被设为NaN。")
        print(f"  处理后有效值数量: {df_pwv[col].notna().sum()} / {original_valid_count} (原有效值) / {len(df_pwv)} (总行数)")


print('\n=== 描述性统计结果 (异常值处理后) ===\n')
print(df_pwv.describe())

print('\n=== 每列的缺失值数量 (异常值处理后) ===\n')
print(df_pwv.isnull().sum())

print('\n=== 清除所有缺失值后的描述性统计结果 (用于后续比较分析) ===\n')
df_pwv_cleaned = df_pwv.dropna(subset=pwv_cols) # dropna on all specified pwv_cols
print(f"原始数据 {len(df)} 行，经过异常值处理并清除含任一PWV缺失值的行后剩余 {len(df_pwv_cleaned)} 行。")

if len(df_pwv_cleaned) < 2:
    print("\n警告: 清除缺失值和异常值后数据过少，可能无法进行可靠的后续比较分析。")
    if not df_pwv_cleaned.empty:
         print(df_pwv_cleaned.describe())
else:
    print(df_pwv_cleaned.describe())
    # 初步比较：左右两侧baPWV差异的均值
    if 'baPWV-右侧-速度m/s' in df_pwv_cleaned.columns and 'baPWV-左侧-速度m/s' in df_pwv_cleaned.columns:
        diff_baPWV = df_pwv_cleaned['baPWV-右侧-速度m/s'] - df_pwv_cleaned['baPWV-左侧-速度m/s']
        print('\n=== 右侧与左侧 baPWV 差值 (右 - 左) 的描述性统计 (异常值处理后) ===\n')
        print(diff_baPWV.describe())

# --- 统计检验与相关性分析 ---
print('\n\n=== 统计检验与相关性分析 ===')

# 1. 左右侧 baPWV 比较
ba_right_col = 'baPWV-右侧-速度m/s'
ba_left_col = 'baPWV-左侧-速度m/s'

diff_baPWV = df_pwv_cleaned[ba_right_col] - df_pwv_cleaned[ba_left_col]
print('\n--- 1. 左右侧 baPWV 比较 ---')
print('\n右侧与左侧 baPWV 差值 (右 - 左) 的描述性统计:')
print(diff_baPWV.describe())

# 正态性检验 (Shapiro-Wilk test for normality)
shapiro_stat, shapiro_p = stats.shapiro(diff_baPWV)
print(f'\nShapiro-Wilk检验 (差值正态性): 统计量={shapiro_stat:.4f}, P值={shapiro_p:.4f}')

alpha = 0.05
if shapiro_p > alpha:
    print('差值数据近似正态分布 (p > 0.05)，将使用配对t检验。')
    ttest_stat, ttest_p = stats.ttest_rel(df_pwv_cleaned[ba_right_col], df_pwv_cleaned[ba_left_col])
    print(f'配对t检验: t统计量={ttest_stat:.4f}, P值={ttest_p:.4f}')
    if ttest_p < alpha:
        print('结论: 左右侧baPWV存在统计学上的显著差异。')
    else:
        print('结论: 左右侧baPWV无统计学上的显著差异。')
else:
    print('差值数据非正态分布 (p <= 0.05)，将使用Wilcoxon符号秩检验。')
    wilcoxon_stat, wilcoxon_p_sides = stats.wilcoxon(df_pwv_cleaned[ba_right_col], df_pwv_cleaned[ba_left_col])
    print(f'Wilcoxon符号秩检验: 统计量={wilcoxon_stat:.4f}, P值={wilcoxon_p_sides:.4f}')
    if wilcoxon_p_sides < alpha:
        print('结论: 左右侧baPWV存在统计学上的显著差异。')
    else:
        print('结论: 左右侧baPWV无统计学上的显著差异。')

# 2. cfPWV 与 baPWV (处理后) 的相关性
print('\n--- 2. cfPWV 与 baPWV 相关性分析 ---')
cf_col = 'cfPWV-速度m/s'

# 创建左右baPWV平均值列 (如果需要)
df_pwv_cleaned.loc[:, 'baPWV-平均速度m/s'] = df_pwv_cleaned[[ba_right_col, ba_left_col]].mean(axis=1)

pwv_for_corr = [cf_col, ba_right_col, ba_left_col, 'baPWV-平均速度m/s']

print('\n将计算以下列之间的Pearson和Spearman相关性:')
print(pwv_for_corr)

# 检查数据是否近似正态分布，以决定主要报告哪个相关系数
# (为简化，此处我们同时计算两者，报告中可根据具体分布选择)

correlation_results = {}
for col1 in pwv_for_corr:
    for col2 in pwv_for_corr:
        if col1 != col2 and (col2, col1) not in correlation_results: #避免重复和自身相关
            # Pearson
            pearson_r, pearson_p = stats.pearsonr(df_pwv_cleaned[col1], df_pwv_cleaned[col2])
            # Spearman
            spearman_r, spearman_p = stats.spearmanr(df_pwv_cleaned[col1], df_pwv_cleaned[col2])
            correlation_results[(col1, col2)] = {
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p
            }

print('\n相关性分析结果 (Pearson r, p-value; Spearman r, p-value):')
for (col1, col2), res in correlation_results.items():
    print(f"  {col1} vs {col2}:")
    print(f"    Pearson : r={res['pearson_r']:.3f} (p={res['pearson_p']:.3f})")
    print(f"    Spearman: r={res['spearman_r']:.3f} (p={res['spearman_p']:.3f})")

# 为方便报告，可以生成一个相关性矩阵的DataFrame
corr_matrix_pearson = df_pwv_cleaned[pwv_for_corr].corr(method='pearson')
corr_matrix_spearman = df_pwv_cleaned[pwv_for_corr].corr(method='spearman')

print('\nPearson 相关系数矩阵:')
print(corr_matrix_pearson.round(3))

print('\nSpearman 相关系数矩阵:')
print(corr_matrix_spearman.round(3))

# --- 图表生成 ---
print('\n\n=== 图表生成 ===')

# 图1: 不同类型PWV值的分布比较 (箱线图)
plt.figure(figsize=(10, 7))
sns.boxplot(data=df_pwv_cleaned[[cf_col, ba_right_col, ba_left_col]])
plt.title('不同类型PWV值的分布比较 (N={})'.format(len(df_pwv_cleaned)), fontsize=16, fontproperties=set_chinese_font(plt.gca().title))
plt.ylabel('PWV (m/s)', fontsize=14, fontproperties=set_chinese_font(plt.gca().yaxis.label))
plt.xticks(ticks=range(3), labels=['cfPWV', '右侧 baPWV', '左侧 baPWV'], fontsize=12, fontproperties=set_chinese_font(plt.gca().get_xticklabels()))
# 添加显著性标记 (基于Wilcoxon P值 for baPWV R vs L)
if wilcoxon_p_sides < 0.05:
    y_max = df_pwv_cleaned[[ba_right_col, ba_left_col]].max().max() # 找到两侧baPWV的最大值作为参考
    # 调整h和text_y确保标记在箱体上方且可见
    h = y_max * 0.05 # 升高一点距离
    text_y = y_max + h * 1.5
    plt.plot([1, 1, 2, 2], [y_max + h, text_y, text_y, y_max + h], lw=1.5, c='black')
    star_text = '***' if wilcoxon_p_sides < 0.001 else '**' if wilcoxon_p_sides < 0.01 else '*'
    plt.text(1.5, text_y, star_text, ha='center', va='bottom', color='black', fontsize=15, fontproperties=set_chinese_font(plt.gca().texts[-1]))

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
try:
    plt.savefig('temp_analysis/pwv_distributions_boxplot.png')
    print("图1: 'temp_analysis/pwv_distributions_boxplot.png' 已保存。")
except Exception as e:
    print(f"错误: 保存图1失败: {e}")
plt.close()

# 图2: cfPWV与baPWV平均值的散点图
plt.figure(figsize=(10, 7))
sns.scatterplot(x=cf_col, y='baPWV-平均速度m/s', data=df_pwv_cleaned, alpha=0.6)
# 添加回归线 (OLS)
try:
    m, b = np.polyfit(df_pwv_cleaned[cf_col], df_pwv_cleaned['baPWV-平均速度m/s'], 1)
    plt.plot(df_pwv_cleaned[cf_col], m*df_pwv_cleaned[cf_col] + b, color='red', linestyle='--', linewidth=2)
except Exception as e:
    print(f"警告: 添加回归线失败: {e}")

# 获取相关系数和P值 (Spearman)
spearman_r_cf_ba_avg, spearman_p_cf_ba_avg = stats.spearmanr(df_pwv_cleaned[cf_col], df_pwv_cleaned['baPWV-平均速度m/s'])

plt.title(f'cfPWV 与 baPWV 平均值关系 (N={len(df_pwv_cleaned)})', fontsize=16, fontproperties=set_chinese_font(plt.gca().title))
plt.xlabel('cfPWV (m/s)', fontsize=14, fontproperties=set_chinese_font(plt.gca().xaxis.label))
plt.ylabel('baPWV 平均值 (m/s)', fontsize=14, fontproperties=set_chinese_font(plt.gca().yaxis.label))
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
try:
    plt.savefig('temp_analysis/cfpwv_vs_bapwv_scatter.png')
    print("图2: 'temp_analysis/cfpwv_vs_bapwv_scatter.png' 已保存。")
except Exception as e:
    print(f"错误: 保存图2失败: {e}")
plt.close()

print("\n--- PWV 分析部分结束 ---")


# ==============================================================================
# === ABI (踝臂指数) 分析 ===
# ==============================================================================
print('\n\n===============================')
print('=== 开始 ABI (踝臂指数) 分析 ===')
print('===============================\n')

# 1. 选择相关的ABI列
abi_cols_defs = {
    'abi_r_dp': 'ABI-右侧-足背指数',
    'abi_r_pt': 'ABI-右侧-胫后指数',
    'abi_l_dp': 'ABI-左侧-足背指数',
    'abi_l_pt': 'ABI-左侧-胫后指数'
}
abi_cols_to_extract = list(abi_cols_defs.values())

# 检查列是否存在
missing_abi_cols = [col for col in abi_cols_to_extract if col not in df.columns]
if missing_abi_cols:
    print(f"错误: 以下期望的ABI列在原始数据中未找到: {', '.join(missing_abi_cols)}")
    # 如果关键列缺失，可能无法继续ABI分析，但此处选择继续，后续步骤会处理NaN
else:
    print("所有预期的ABI指数列均已找到。")

# 2. 从原始DataFrame 'df' 提取数据并转换为数值型
# 使用 .copy() 避免 SettingWithCopyWarning
df_abi = df[abi_cols_to_extract].copy()

print('\n=== ABI列原始数据类型检查 ===')
print(df_abi.dtypes)

print('\n=== ABI列转换为数值型并检查非空值数量 (转换前) ===')
for col_key, col_name in abi_cols_defs.items():
    if col_name in df_abi.columns:
        # 先记录转换前的有效值数量
        initial_valid_count = pd.to_numeric(df_abi[col_name], errors='coerce').notna().sum()
        df_abi[col_name] = pd.to_numeric(df_abi[col_name], errors='coerce')
        print(f"列 '{col_name}' ({col_key}) 初始有效值数量: {initial_valid_count} / {len(df_abi)}")
    else:
        print(f"警告: 列 '{col_name}' ({col_key}) 未在提取的df_abi中找到，将跳过处理。")
        df_abi[col_name] = np.nan # 确保列存在，即使原始数据没有，填充NaN


print('\n=== ABI列原始描述性统计 (转换后, 清理前) ===')
print(df_abi[abi_cols_to_extract].describe())

# 3. 定义ABI_R, ABI_L, ABI_Overall
# ABI_R = max(右侧足背指数, 右侧胫后指数)
df_abi['ABI_R'] = df_abi[[abi_cols_defs['abi_r_dp'], abi_cols_defs['abi_r_pt']]].max(axis=1)
# ABI_L = max(左侧足背指数, 左侧胫后指数)
df_abi['ABI_L'] = df_abi[[abi_cols_defs['abi_l_dp'], abi_cols_defs['abi_l_pt']]].max(axis=1)
# ABI_Overall = min(ABI_R, ABI_L) - 这是常用的临床筛查ABI
df_abi['ABI_Overall'] = df_abi[['ABI_R', 'ABI_L']].min(axis=1)

print('\n=== 计算得到的 ABI_R, ABI_L, ABI_Overall 描述性统计 (清理前) ===')
print(df_abi[['ABI_R', 'ABI_L', 'ABI_Overall']].describe())

# 4. ABI 值清理
# ABI 合理范围大致在 0.1 到 2.5 之间。小于0.1通常不太可能，大于2.0可能提示血管钙化或测量错误。
# 这里我们先观察，如果需要严格清理，可以设定阈值。
# 例如，将超出 (0.2, 2.5) 范围的值设为 NaN
abi_lower_bound = 0.2
abi_upper_bound = 2.5
columns_to_clean = ['ABI_R', 'ABI_L', 'ABI_Overall']

print(f'\n=== 应用生理范围过滤 ABI 值 ({abi_lower_bound} - {abi_upper_bound}) ===\n')
for col in columns_to_clean:
    if col in df_abi.columns:
        original_valid_count = df_abi[col].notna().sum()
        outside_range_mask = (df_abi[col] < abi_lower_bound) | (df_abi[col] > abi_upper_bound)
        num_outside_range = outside_range_mask.sum()
        
        df_abi.loc[outside_range_mask, col] = np.nan
        print(f"列 '{col}': {num_outside_range} 个值因超出生理范围 ({abi_lower_bound}-{abi_upper_bound}) 被设为NaN。")
        print(f"  处理后有效值数量: {df_abi[col].notna().sum()} / {original_valid_count} (原有效值) / {len(df_abi)} (总行数)")

print('\n=== 计算得到的 ABI_R, ABI_L, ABI_Overall 描述性统计 (清理后) ===')
print(df_abi[['ABI_R', 'ABI_L', 'ABI_Overall']].describe())

# 用于分析的数据框：移除 ABI_Overall 为 NaN 的行
df_abi_cleaned = df_abi.dropna(subset=['ABI_Overall']).copy()
print(f"\n用于ABI分析的数据: {len(df_abi_cleaned)} 行 (在ABI_Overall上有有效值)。")

if len(df_abi_cleaned) < 2:
    print("警告: 清理和移除缺失ABI_Overall后数据过少，部分后续分析可能无法进行或不可靠。")
else:
    # 5. 临床分类 (基于 ABI_Overall 清理后的值)
    print('\n--- 5. ABI 临床分类 (基于 ABI_Overall 清理后) ---')
    bins = [0, 0.90, 0.99, 1.30, abi_upper_bound + 0.1] # abi_upper_bound + 0.1 确保上界包含
    labels = ['PAD (≤0.90)', '临界值 (0.91-0.99)', '正常 (1.00-1.30)', '不可压/疑钙化 (>1.30)']
    
    # 确保 abi_upper_bound 足够大以包含所有清理后的数据，或者调整bins/labels逻辑
    # 如果 abi_upper_bound 是 2.5, 那么 >1.30 类别会捕获到 2.5
    # 如果有值恰好等于abi_upper_bound，它会落入最后一个bin
    
    df_abi_cleaned.loc[:, 'ABI_Category'] = pd.cut(df_abi_cleaned['ABI_Overall'], bins=bins, labels=labels, right=True, include_lowest=True)
    
    category_counts = df_abi_cleaned['ABI_Category'].value_counts().sort_index()
    category_percentages = df_abi_cleaned['ABI_Category'].value_counts(normalize=True).sort_index() * 100
    
    print("ABI_Overall 临床分类统计:")
    abi_category_summary = pd.DataFrame({'例数': category_counts, '百分比 (%)': category_percentages.round(2)})
    print(abi_category_summary)

    # 6. 左右侧 ABI 比较 (基于 df_abi_cleaned 中左右ABI均有效的部分)
    print('\n--- 6. 左右侧 ABI 比较 ---')
    # 需要左右ABI均不为空的子集进行比较
    lr_compare_df = df_abi_cleaned[['ABI_R', 'ABI_L']].dropna()
    if len(lr_compare_df) >= 2:
        print(f"用于左右ABI比较的样本数: {len(lr_compare_df)}")
        diff_abi_rl = lr_compare_df['ABI_R'] - lr_compare_df['ABI_L']
        print('\n右侧与左侧 ABI 差值 (右 - 左) 的描述性统计:')
        print(diff_abi_rl.describe())

        shapiro_stat_abi, shapiro_p_abi = stats.shapiro(diff_abi_rl)
        print(f'\nShapiro-Wilk检验 (ABI差值正态性): 统计量={shapiro_stat_abi:.4f}, P值={shapiro_p_abi:.4f}')
        
        alpha = 0.05
        wilcoxon_p_abi_sides = np.nan # 初始化
        if shapiro_p_abi > alpha:
            print('ABI差值数据近似正态分布 (p > 0.05)，将使用配对t检验。')
            ttest_stat_abi, ttest_p_abi = stats.ttest_rel(lr_compare_df['ABI_R'], lr_compare_df['ABI_L'])
            print(f'配对t检验: t统计量={ttest_stat_abi:.4f}, P值={ttest_p_abi:.4f}')
            wilcoxon_p_abi_sides = ttest_p_abi # Store p-value for plot annotation
            if ttest_p_abi < alpha:
                print('结论: 左右侧ABI存在统计学上的显著差异。')
            else:
                print('结论: 左右侧ABI无统计学上的显著差异。')
        else:
            print('ABI差值数据非正态分布 (p <= 0.05)，将使用Wilcoxon符号秩检验。')
            wilcoxon_stat_abi, wilcoxon_p_abi = stats.wilcoxon(lr_compare_df['ABI_R'], lr_compare_df['ABI_L'])
            print(f'Wilcoxon符号秩检验: 统计量={wilcoxon_stat_abi:.4f}, P值={wilcoxon_p_abi:.4f}')
            wilcoxon_p_abi_sides = wilcoxon_p_abi # Store p-value for plot annotation
            if wilcoxon_p_abi < alpha:
                print('结论: 左右侧ABI存在统计学上的显著差异。')
            else:
                print('结论: 左右侧ABI无统计学上的显著差异。')
    else:
        print("左右ABI均有效的样本过少，无法进行比较。")
        wilcoxon_p_abi_sides = np.nan # 确保已定义

    # 7. ABI 图表生成
    print('\n--- 7. ABI 图表生成 ---')

    # 图3: ABI 值分布比较 (箱线图)
    plt.figure(figsize=(10, 7))
    plot_data_abi = df_abi_cleaned[['ABI_R', 'ABI_L', 'ABI_Overall']].dropna() # Ensure no NaNs for plotting these three
    if not plot_data_abi.empty:
        sns.boxplot(data=plot_data_abi)
        plt.title(f'不同ABI值的分布比较 (N={len(plot_data_abi)})', fontsize=16, fontproperties=set_chinese_font(plt.gca().title))
        plt.ylabel('ABI值', fontsize=14, fontproperties=set_chinese_font(plt.gca().yaxis.label))
        plt.xticks(ticks=range(3), labels=['右侧 ABI (ABI_R)', '左侧 ABI (ABI_L)', '总体 ABI (ABI_Overall)'], fontsize=12, fontproperties=set_chinese_font(plt.gca().get_xticklabels()))

        # 添加左右ABI比较的显著性标记 (基于上面计算的 wilcoxon_p_abi_sides 或 ttest_p_abi)
        if not np.isnan(wilcoxon_p_abi_sides) and wilcoxon_p_abi_sides < 0.05 and len(lr_compare_df) >=2 : # lr_compare_df ensures we have data for comparison
            # Determine y-position for the significance bar
            # Use the max of ABI_R and ABI_L from the data used for the test (lr_compare_df)
            y_max_rl_abi = lr_compare_df[['ABI_R', 'ABI_L']].max().max() 
            h_abi = y_max_rl_abi * 0.05 
            text_y_abi = y_max_rl_abi + h_abi * 1.5
            # Plot significance line between ABI_R (index 0) and ABI_L (index 1)
            plt.plot([0, 0, 1, 1], [y_max_rl_abi + h_abi, text_y_abi, text_y_abi, y_max_rl_abi + h_abi], lw=1.5, c='black')
            star_text_abi = '***' if wilcoxon_p_abi_sides < 0.001 else '**' if wilcoxon_p_abi_sides < 0.01 else '*'
            plt.text(0.5, text_y_abi, star_text_abi, ha='center', va='bottom', color='black', fontsize=15, fontproperties=set_chinese_font(plt.gca().texts[-1]))

        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        try:
            plt.savefig('temp_analysis/abi_distributions_boxplot.png')
            print("图3: 'temp_analysis/abi_distributions_boxplot.png' 已保存。")
        except Exception as e:
            print(f"错误: 保存图3失败: {e}")
        plt.close()
    else:
        print("警告: 清理后用于ABI分布图的数据为空，跳过图3生成。")

    # 图4: ABI 临床分类分布 (柱状图)
    if 'ABI_Category' in df_abi_cleaned.columns and not df_abi_cleaned['ABI_Category'].empty:
        plt.figure(figsize=(10, 7))
        category_counts.plot(kind='bar', color=sns.color_palette("viridis", len(category_counts)))
        plt.title(f'ABI_Overall 临床分类分布 (N={len(df_abi_cleaned)})', fontsize=16, fontproperties=set_chinese_font(plt.gca().title))
        plt.xlabel('ABI 分类', fontsize=14, fontproperties=set_chinese_font(plt.gca().xaxis.label))
        plt.ylabel('例数', fontsize=14, fontproperties=set_chinese_font(plt.gca().yaxis.label))
        plt.xticks(rotation=45, ha='right', fontsize=12, fontproperties=set_chinese_font(plt.gca().get_xticklabels()))
        plt.yticks(fontsize=12, fontproperties=set_chinese_font(plt.gca().get_yticklabels()))
        # 在柱状图顶部显示具体数值
        for i, count in enumerate(category_counts):
            plt.text(i, count + (category_counts.max()*0.01), str(count), ha='center', va='bottom', fontsize=11, fontproperties=set_chinese_font(plt.gca().texts[-1]))
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        try:
            plt.savefig('temp_analysis/abi_clinical_categories_barchart.png')
            print("图4: 'temp_analysis/abi_clinical_categories_barchart.png' 已保存。")
        except Exception as e:
            print(f"错误: 保存图4失败: {e}")
        plt.close()
    else:
        print("警告: ABI临床分类数据为空或不存在，跳过图4生成。")

print("\n--- ABI 分析部分结束 ---")
print("\n脚本执行完毕。") 