#!/bin/bash

# PWV数据分析工具安装脚本
echo "=============================================="
echo "PWV数据分析工具 - 安装脚本"
echo "基于小米可穿戴设备进行动脉硬化筛查与脑卒中风险预测研究"
echo "=============================================="

# 使用pyenv的Python 3.12.9
echo "切换到Python 3.12.9..."
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
pyenv shell 3.12.9

python_version=$(python --version 2>&1)
if [[ $? -ne 0 ]]; then
    echo "❌ 未找到Python 3.12.9，请先用pyenv安装Python 3.12.9"
    exit 1
fi
echo "✅ 使用Python: $python_version"

# 检查pip
pip_version=$(pip --version 2>&1)
if [[ $? -ne 0 ]]; then
    echo "❌ 未找到pip，请先安装pip"
    exit 1
fi
echo "✅ 检测到pip: $pip_version"

# 阶段1：设置虚拟环境
setup_venv() {
    # 检查是否已存在虚拟环境
    if [ -d "venv" ] && [ -f "venv/bin/activate" ]; then
        echo "检测到已存在的虚拟环境。您想要:"
        echo "1) 使用已有环境"
        echo "2) 创建新环境（会删除现有环境）"
        echo "3) 跳过使用虚拟环境"
        read -p "请选择 [1/2/3]: " venv_choice
        
        case "$venv_choice" in
            1)
                echo "使用已有环境..."
                source venv/bin/activate
                if [[ $? -ne 0 ]]; then
                    echo "❌ 虚拟环境激活失败"
                    return 1
                fi
                use_venv=true
                ;;
            2)
                echo "创建新环境..."
                rm -rf venv
                python -m venv venv
                if [[ $? -ne 0 ]]; then
                    echo "❌ 虚拟环境创建失败"
                    return 1
                fi
                source venv/bin/activate
                if [[ $? -ne 0 ]]; then
                    echo "❌ 虚拟环境激活失败"
                    return 1
                fi
                use_venv=true
                ;;
            3)
                echo "跳过虚拟环境使用，将使用系统Python"
                use_venv=false
                ;;
            *)
                echo "无效选择，默认使用已有环境..."
                source venv/bin/activate
                if [[ $? -ne 0 ]]; then
                    echo "❌ 虚拟环境激活失败"
                    return 1
                fi
                use_venv=true
                ;;
        esac
    else
        echo "未检测到虚拟环境。您想要:"
        echo "1) 创建新环境"
        echo "2) 跳过使用虚拟环境"
        read -p "请选择 [1/2]: " venv_choice
        
        case "$venv_choice" in
            2)
                echo "跳过虚拟环境使用，将使用系统Python"
                use_venv=false
                ;;
            *)
                echo "创建新环境..."
                python -m venv venv
                if [[ $? -ne 0 ]]; then
                    echo "❌ 虚拟环境创建失败"
                    return 1
                fi
                source venv/bin/activate
                if [[ $? -ne 0 ]]; then
                    echo "❌ 虚拟环境激活失败"
                    return 1
                fi
                use_venv=true
                ;;
        esac
    fi
    
    if [ "$use_venv" = true ]; then
        echo "✅ 虚拟环境已激活"
    fi
    
    return 0
}

# 阶段2：安装依赖
install_dependencies() {
    # 安装依赖
    echo "正在安装所需依赖..."
    echo "注意: 这个过程可能需要一些时间，请耐心等待..."
    
    pip install -e .
    if [[ $? -ne 0 ]]; then
        echo "❌ 依赖安装失败"
        return 1
    fi
    
    echo "✅ 依赖安装成功"
    
    # 尝试安装SHAP库（可选）
    read -p "是否要安装SHAP库用于高级模型解释? [Y/n]: " install_shap
    
    if [[ "$install_shap" == "" || "$install_shap" == "y" || "$install_shap" == "Y" ]]; then
        echo "正在安装SHAP库（用于模型解释）..."
        pip install shap || echo "⚠️ SHAP安装失败，但这不会影响基本功能。某些高级模型解释功能可能不可用。"
    else
        echo "跳过SHAP安装。"
    fi
    
    return 0
}

# 阶段3：创建目录结构
setup_directories() {
    # 创建输出目录
    echo "正在创建输出目录..."
    mkdir -p output/figures/{distribution,boxplots,correlation,gender_comparison,age_analysis,regression}
    mkdir -p output/tables output/reports
    
    # 创建符号链接
    echo "正在创建符号链接，以便报告生成器可以找到图像..."
    ln -sf figures output/image
    
    echo "✅ 输出目录已创建"
    
    return 0
}

# 阶段4：运行分析脚本
run_analysis() {
    echo "开始运行PWV数据分析流程..."
    echo "注意: 这个过程可能需要几分钟时间，请耐心等待..."
    
    # 确保虚拟环境被激活
    if [ "$use_venv" = true ] && [[ ! "$VIRTUAL_ENV" == *"venv"* ]]; then
        echo "激活虚拟环境..."
        source venv/bin/activate
    fi
    
    # 运行分析脚本
    python scripts/run_pwv_analysis.py
    if [[ $? -ne 0 ]]; then
        echo "❌ 分析脚本运行失败"
        return 1
    fi
    
    echo "✅ PWV数据分析流程已完成"
    return 0
}

# 主流程
echo "PWV数据分析工具安装将分4个阶段进行:"
echo "1. 设置虚拟环境"
echo "2. 安装依赖"
echo "3. 创建目录结构"
echo "4. 运行分析脚本（可选）"
echo ""

# 声明全局变量
use_venv=false

# 阶段1
echo "====== 阶段1: 设置虚拟环境 ======"
setup_venv
if [[ $? -ne 0 ]]; then
    echo "❌ 虚拟环境设置失败，安装中止"
    exit 1
fi

# 确认继续
read -p "虚拟环境已设置完成，是否继续安装依赖? [Y/n]: " continue_install
if [[ "$continue_install" == "n" || "$continue_install" == "N" ]]; then
    echo "安装已暂停。要继续安装，请重新运行脚本。"
    exit 0
fi

# 阶段2
echo "====== 阶段2: 安装依赖 ======"
install_dependencies
if [[ $? -ne 0 ]]; then
    echo "❌ 依赖安装失败，安装中止"
    exit 1
fi

# 确认继续
read -p "依赖已安装完成，是否继续创建目录结构? [Y/n]: " continue_dirs
if [[ "$continue_dirs" == "n" || "$continue_dirs" == "N" ]]; then
    echo "安装已暂停。要继续安装，请重新运行脚本并选择跳过前两个阶段。"
    exit 0
fi

# 阶段3
echo "====== 阶段3: 创建目录结构 ======"
setup_directories
if [[ $? -ne 0 ]]; then
    echo "❌ 目录结构创建失败，安装中止"
    exit 1
fi

# 完成安装
echo ""
echo "=============================================="
echo "✅ PWV数据分析工具安装完成！"
echo "=============================================="
echo ""

# 询问是否要立即运行分析
read -p "是否立即运行PWV数据分析? [Y/n]: " run_now
if [[ "$run_now" == "" || "$run_now" == "y" || "$run_now" == "Y" ]]; then
    echo ""
    echo "====== 阶段4: 运行数据分析 ======"
    run_analysis
    analysis_status=$?
    echo ""
else
    echo "跳过数据分析运行。"
    analysis_status=0
fi

# 显示使用方法
echo "=============================================="
echo "使用方法:"
if [ "$use_venv" = true ]; then
    echo "1. 每次使用前需要先激活虚拟环境:"
    echo "   source venv/bin/activate"
    echo ""
fi
echo "2. 运行分析脚本:"
echo "   python scripts/run_pwv_analysis.py"
echo ""
echo "3. 查看生成的报告:"
echo "   - Markdown报告: output/reports/PWV数据分析综合报告.md"
echo "   - Word报告: output/reports/PWV数据分析综合报告.docx"
echo "   - HTML报告: output/reports/PWV数据分析综合报告.html"
echo "   - Excel数据表: output/tables/"
echo "   - 图表: output/figures/ 和 output/image/"
echo ""
echo "如有问题，请联系项目负责人。"
echo "==============================================" 

# 如果最后一个步骤失败，返回非零退出码
if [[ $analysis_status -ne 0 ]]; then
    exit 1
fi
exit 0 