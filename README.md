# PWV数据分析工具

基于小米可穿戴设备进行动脉硬化筛查与脑卒中风险预测研究的数据分析工具

## 项目简介

本项目是基于小米可穿戴设备收集的数据，针对动脉硬化筛查与脑卒中风险预测进行分析的工具集。主要功能包括：

1. **数据处理**：加载、清洗和预处理PWV测量数据
2. **数据分析**：执行基础统计分析、相关性分析、回归分析等
3. **高级分析**：进行机器学习建模、风险分层和预测
4. **临床分析**：基于临床标准进行风险评估和分类
5. **数据可视化**：生成各类图表展示分析结果
6. **报告生成**：自动生成分析报告（Markdown, Word, HTML格式）

## 系统要求

- Python 3.7或更高版本
- 支持Windows、macOS和Linux系统

## 安装指南

### Linux/macOS安装

1. 克隆或下载本仓库到本地
2. 打开终端，进入项目根目录
3. 运行安装脚本：

```bash
chmod +x install.sh
./install.sh
```

### Windows安装

1. 克隆或下载本仓库到本地
2. 双击`install.bat`运行安装脚本
3. 按照屏幕提示操作

### 手动安装

如果安装脚本不适用于您的系统，可以手动安装：

```bash
# 创建并激活虚拟环境（可选）
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate.bat  # Windows

# 安装依赖
pip install -e .
```

## 使用方法

### 数据准备

1. 将PWV测量数据Excel文件放入`docs/excel/`目录中
2. 确保Excel文件包含必要的列（PWV值、年龄、性别、血压等）

### 运行分析

```bash
# 如果使用了虚拟环境，先激活环境
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate.bat  # Windows

# 运行主脚本
python scripts/run_pwv_analysis.py
```

### 查看结果

分析完成后，可以在以下位置查看生成的结果：

- 数值分析结果：`output/tables/`目录下的Excel文件
- 图表：`output/image/`目录下的PNG文件
- Markdown报告：`output/reports/PWV_Analysis_Report.md`
- Word报告：`output/reports/PWV_Analysis_Report.docx`
- HTML报告：`output/reports/PWV_Analysis_Report.html`

## 项目结构

```
pwv_analysis/
├── docs/                    # 文档目录
│   ├── excel/              # 原始Excel数据文件
│   ├── markdown/           # Markdown文档
│   ├── pdf/                # PDF文档
│   └── word/               # Word文档
├── images/                  # 项目相关图片
├── metadata/                # 元数据文件
├── output/                  # 输出目录
│   ├── figures/            # 生成的图表
│   ├── image/              # 生成的图像
│   ├── reports/            # 生成的报告
│   └── tables/             # 生成的数据表
├── scripts/                 # 脚本目录
│   ├── advanced_analysis.py        # 高级分析模块
│   ├── clinical_analysis.py        # 临床分析模块
│   ├── data_analysis.py            # 数据分析模块
│   ├── data_processing.py          # 数据处理模块
│   ├── data_visualization.py       # 数据可视化模块
│   ├── enhanced_visualization.py   # 增强可视化模块
│   ├── report_generator.py         # 报告生成模块
│   └── run_pwv_analysis.py         # 主控脚本
├── install.bat              # Windows安装脚本
├── install.sh               # Linux/macOS安装脚本
├── setup.py                 # 安装配置
└── README.md                # 项目说明文档
```

## 主要功能模块

- **data_processing.py**：负责数据加载、清洗和特征工程
- **data_analysis.py**：实现基础统计分析和回归分析
- **data_visualization.py**：生成基础数据可视化图表
- **advanced_analysis.py**：实现高级统计分析和机器学习模型
- **clinical_analysis.py**：执行临床相关的分析和评估
- **enhanced_visualization.py**：生成高级可视化图表
- **report_generator.py**：生成分析报告
- **run_pwv_analysis.py**：主控脚本，协调整个分析流程

## 常见问题

**Q: 如何自定义分析参数？**  
A: 可以在各个分析模块中修改相应的参数设置。

**Q: 我的Excel文件格式与要求不符怎么办？**  
A: 可以手动修改Excel文件以符合要求，或者调整`data_processing.py`中的数据加载和处理逻辑。

**Q: 如何添加新的分析方法？**  
A: 可以在现有模块中添加新函数，或者创建新的分析模块并在`run_pwv_analysis.py`中引入。

## 许可证

本项目仅供人民医院小米项目研究使用，未经许可不得用于其他目的。

## 联系方式

有关本项目的问题或建议，请联系项目负责人。

# 中文字体显示功能

本项目支持在图表中正确显示中文字符，包括标题、标签和坐标轴文本。我们实现了一套健壮的字体检测和应用机制，确保在各种环境下都能正确显示中文。

## 字体查找顺序

系统会按照以下顺序查找并使用中文字体：

1. WSL环境：优先使用Windows系统字体（如SimHei黑体、Microsoft YaHei微软雅黑等）
2. 用户下载字体：检查`~/.fonts/`目录中是否有SimHei.ttf等中文字体
3. 系统字体：搜索系统字体目录中的中文字体
4. 自动下载：如果以上方式都找不到合适的字体，会自动从GitHub下载SimHei或其他中文字体

## 在代码中使用

要在自己的图表中使用中文字体显示功能，只需导入并使用以下函数：

```python
from scripts.data_visualization import fix_chinese_font_display, apply_font_to_axis, apply_font_to_figure

# 创建图表
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 5, 6])
ax.set_title('中文标题')
ax.set_xlabel('中文横轴')
ax.set_ylabel('中文纵轴')

# 应用中文字体到坐标轴
apply_font_to_axis(ax)

# 或者应用到整个图表
# apply_font_to_figure(fig)

plt.savefig('chinese_chart.png')
```

## 自定义字体

如果需要使用特定的中文字体，可以将字体文件（如.ttf格式）放在`~/.fonts/`目录下，系统会自动检测并使用。

## 常见问题

1. **中文显示为方块**：如果中文仍然显示为方块，请确保使用了`apply_font_to_axis()`函数，特别是对于坐标轴刻度标签。

2. **在Docker容器中运行**：在Docker容器中运行时，请确保安装了相应的字体包：
   ```
   # Ubuntu/Debian
   RUN apt-get update && apt-get install -y fonts-wqy-microhei
   
   # CentOS/RHEL
   RUN yum install -y wqy-microhei-fonts
   ```

3. **字体质量问题**：如果对默认字体的显示效果不满意，可以尝试安装更高质量的字体，如思源黑体（Source Han Sans）或Noto Sans CJK。