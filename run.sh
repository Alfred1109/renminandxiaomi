#!/bin/bash

# PWV数据分析工具运行脚本
# 设置日志文件
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/pwv_analysis_$(date +"%Y%m%d_%H%M%S").log"

# 日志函数
log_info() {
    local message="[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $1"
    echo "$message" | tee -a "$LOG_FILE"
}

log_error() {
    local message="[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - $1"
    echo "$message" | tee -a "$LOG_FILE"
}

log_success() {
    local message="[SUCCESS] $(date '+%Y-%m-%d %H:%M:%S') - $1"
    echo "$message" | tee -a "$LOG_FILE"
}

# 开始记录日志
log_info "=============================================="
log_info "PWV数据分析工具 - 运行脚本"
log_info "基于小米可穿戴设备进行动脉硬化筛查与脑卒中风险预测研究"
log_info "=============================================="

# 检查当前目录是否有venv
if [ -d "venv" ] && [ -f "venv/bin/activate" ]; then
    use_venv=true
    log_info "检测到虚拟环境venv"
else
    use_venv=false
    log_info "未检测到虚拟环境，将使用系统Python"
fi

# 使用pyenv切换到Python 3.12.9
if command -v pyenv &> /dev/null; then
    log_info "检测到pyenv，切换到Python 3.12.9..."
    {
        export PYENV_ROOT="$HOME/.pyenv"
        export PATH="$PYENV_ROOT/bin:$PATH"
        eval "$(pyenv init -)"
        pyenv shell 3.12.9
    } >> "$LOG_FILE" 2>&1
    
    python_version=$(python --version 2>&1)
    if [[ $? -ne 0 ]]; then
        log_error "Python 3.12.9切换失败"
        log_info "尝试使用系统Python..."
    else
        log_success "使用Python: $python_version"
    fi
else
    log_info "未检测到pyenv，将使用系统Python"
fi

# 检查Python版本
python_version=$(python --version 2>&1)
if [[ $? -ne 0 ]]; then
    log_error "未找到Python，请确保Python已安装"
    exit 1
fi
log_success "使用Python: $python_version"

# 激活虚拟环境（如果存在）
if [ "$use_venv" = true ]; then
    log_info "激活虚拟环境..."
    {
        source venv/bin/activate
    } >> "$LOG_FILE" 2>&1
    
    if [[ $? -ne 0 ]]; then
        log_error "虚拟环境激活失败，将尝试使用系统Python"
        use_venv=false
    else
        log_success "虚拟环境已激活"
        # 记录虚拟环境中的包信息
        log_info "记录虚拟环境包信息..."
        {
            pip freeze >> "$LOG_FILE" 2>&1
        } >> "$LOG_FILE" 2>&1
    fi
fi

# 检查是否存在分析脚本
if [ ! -f "scripts/run_pwv_analysis.py" ]; then
    log_error "未找到分析脚本: scripts/run_pwv_analysis.py"
    log_error "请确认当前目录是否为项目根目录"
    exit 1
fi
log_info "分析脚本检查通过"

# 检查输出目录
if [ ! -d "output" ]; then
    log_info "警告: 未找到输出目录，将创建必要的目录结构..."
    {
        mkdir -p output/figures/{distribution,boxplots,correlation,gender_comparison,age_analysis,regression}
        mkdir -p output/tables output/reports
        ln -sf figures output/image
    } >> "$LOG_FILE" 2>&1
    log_success "输出目录已创建"
fi

# 运行分析脚本
log_info "=============================================="
log_info "开始运行PWV数据分析流程..."
log_info "注意: 这个过程可能需要几分钟时间，请耐心等待..."
log_info "=============================================="

# 捕获分析脚本的标准输出和错误输出
{
    python scripts/run_pwv_analysis.py
    analysis_status=$?
} >> "$LOG_FILE" 2>&1

if [[ $analysis_status -ne 0 ]]; then
    log_error "分析脚本运行失败，退出代码: $analysis_status"
    log_error "请检查日志文件 $LOG_FILE 查看详细错误信息"
    echo "❌ 分析脚本运行失败，退出代码: $analysis_status"
    echo "请检查日志文件 $LOG_FILE 查看详细错误信息"
    exit 1
fi

log_success "=============================================="
log_success "PWV数据分析流程已完成！"
log_success "=============================================="
log_info "生成的报告及结果位于:"
log_info "- Markdown报告: output/reports/PWV数据分析综合报告.md"
log_info "- Word报告: output/reports/PWV数据分析综合报告.docx"
log_info "- HTML报告: output/reports/PWV数据分析综合报告.html"
log_info "- Excel数据表: output/tables/"
log_info "- 图表: output/figures/ 和 output/image/"
log_info "=============================================="

echo "=============================================="
echo "✅ PWV数据分析流程已完成！"
echo "=============================================="
echo "生成的报告及结果位于:"
echo "- Markdown报告: output/reports/PWV数据分析综合报告.md"
echo "- Word报告: output/reports/PWV数据分析综合报告.docx"
echo "- HTML报告: output/reports/PWV数据分析综合报告.html"
echo "- Excel数据表: output/tables/"
echo "- 图表: output/figures/ 和 output/image/"
echo "=============================================="
echo "详细日志已保存至: $LOG_FILE"

exit 0 