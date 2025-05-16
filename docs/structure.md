# 项目文件结构

本文档描述了PWV数据分析工具项目的整体文件结构。

## 目录结构

```
pwv_analysis/                    # 项目根目录
├── docs/                        # 项目文档目录
│   ├── excel/                   # 存放原始Excel数据文件
│   ├── markdown/                # Markdown格式文档
│   ├── pdf/                     # PDF格式文档
│   ├── word/                    # Word格式文档
│   └── structure.md             # 项目结构说明文档（本文件）
├── metadata/                    # 元数据文件目录
├── output/                      # 输出文件目录
│   ├── figures/                 # 图表输出目录
│   ├── image/                   # 图像输出目录
│   ├── reports/                 # 生成的报告目录
│   └── tables/                  # 生成的数据表目录
├── scripts/                     # 项目核心脚本目录
│   ├── advanced_analysis.py     # 高级数据分析模块
│   ├── clinical_analysis.py     # 临床分析模块
│   ├── convert_md_to_docx.py    # Markdown转Word文档工具
│   ├── data_analysis.py         # 数据分析模块
│   ├── data_processing.py       # 数据处理模块
│   ├── data_visualization.py    # 数据可视化模块
│   ├── enhanced_visualization.py# 增强数据可视化模块
│   ├── orgnization/             # 组织相关脚本目录
│   ├── regenerate_report.py     # 重新生成报告工具
│   ├── rename_risk_images.py    # 风险图像重命名工具
│   ├── report_generator.py      # 报告生成模块
│   ├── risk_prediction.py       # 风险预测模块
│   └── run_pwv_analysis.py      # 主控脚本
├── venv/                        # Python虚拟环境目录
├── install.bat                  # Windows安装脚本
├── install.sh                   # Linux/macOS安装脚本
├── README.md                    # 项目说明文档
└── setup.py                     # 安装配置文件
```

## 核心模块

### 1. 数据处理模块 (data_processing.py)

- 功能：加载原始PWV数据、数据清洗和预处理
- 主要函数：
  - `find_latest_data_file()`: 查找最新的PWV数据文件
  - `load_data()`: 加载数据文件
  - `clean_data()`: 清洗数据
  - `create_derived_features()`: 创建派生特征
  - `load_and_prepare_data()`: 整合数据加载和准备流程

### 2. 数据分析模块 (data_analysis.py)

- 功能：实现基础统计分析和相关性分析
- 包含各种统计方法和数据分析函数

### 3. 高级分析模块 (advanced_analysis.py)

- 功能：实现高级统计分析和模型分析
- 包含多元分析和机器学习模型分析

### 4. 临床分析模块 (clinical_analysis.py)

- 功能：执行临床相关的分析和风险评估
- 基于医学标准进行风险分层

### 5. 风险预测模块 (risk_prediction.py)

- 功能：进行机器学习建模和风险预测
- 构建和评估预测模型

### 6. 数据可视化模块

- 基础可视化 (data_visualization.py)：创建基础数据图表
- 增强可视化 (enhanced_visualization.py)：创建高级可视化图表

### 7. 报告生成模块 (report_generator.py)

- 功能：生成综合分析报告
- 支持多种格式：Markdown, Word, HTML

### 8. 主控脚本 (run_pwv_analysis.py)

- 功能：协调整个分析流程
- 调用各个模块完成端到端的数据分析和报告生成

## 数据流向

1. 数据首先从`docs/excel/`目录中的Excel文件加载
2. 通过`data_processing.py`进行预处理和清洗
3. 经过`data_analysis.py`、`advanced_analysis.py`和`clinical_analysis.py`模块进行分析
4. 使用`data_visualization.py`和`enhanced_visualization.py`生成可视化图表
5. 通过`risk_prediction.py`进行风险预测建模
6. 最后由`report_generator.py`整合结果生成报告
7. 输出结果保存到`output/`目录的相应子目录中

## 安装与运行

- 使用`install.sh`(Linux/macOS)或`install.bat`(Windows)安装项目
- 运行`scripts/run_pwv_analysis.py`执行完整的分析流程 