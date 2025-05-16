#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
风险图片重命名脚本：
重命名风险子目录中的图片文件，添加风险类型前缀，避免与主目录中的文件冲突
"""

import os
import sys
import glob
import shutil
import re
from pathlib import Path

def rename_risk_images(base_dir="output"):
    """重命名风险子目录中的图片文件"""
    print("开始重命名风险子目录中的图片文件...")
    
    image_dir = os.path.join(base_dir, "image")
    risk_dir = os.path.join(image_dir, "风险预测")
    
    if not os.path.exists(risk_dir):
        print(f"风险预测目录不存在: {risk_dir}")
        return False
    
    # 风险子目录列表
    risk_dirs = {
        "高血压风险": "hypertension_risk",
        "PWV超标风险": "pwv_risk",
        "高综合风险": "comprehensive_risk"
    }
    
    renamed_files = []
    
    # 处理每个风险子目录
    for risk_name, risk_prefix in risk_dirs.items():
        risk_path = os.path.join(risk_dir, risk_name)
        if not os.path.exists(risk_path):
            print(f"风险子目录不存在: {risk_path}")
            continue
        
        print(f"处理风险子目录: {risk_name}")
        
        # 获取子目录中的所有图片文件
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.gif', '*.svg']:
            image_files.extend(glob.glob(os.path.join(risk_path, ext)))
        
        print(f"找到 {len(image_files)} 个图片文件")
        
        # 重命名每个文件
        for img_file in image_files:
            filename = os.path.basename(img_file)
            new_filename = f"{risk_prefix}_{filename}"
            new_path = os.path.join(risk_path, new_filename)
            
            # 重命名文件
            os.rename(img_file, new_path)
            print(f"已重命名: {filename} -> {new_filename}")
            renamed_files.append((img_file, new_path))
    
    print(f"已重命名 {len(renamed_files)} 个文件")
    return renamed_files

def update_markdown_references(renamed_files, base_dir="output"):
    """更新Markdown文件中的图片引用"""
    print("\n更新Markdown报告中的图片引用...")
    
    reports_dir = os.path.join(base_dir, "reports")
    if not os.path.exists(reports_dir):
        print(f"报告目录不存在: {reports_dir}")
        return False
    
    # 查找所有Markdown文件
    markdown_files = glob.glob(os.path.join(reports_dir, "*.md"))
    if not markdown_files:
        print("未找到Markdown报告文件")
        return False
    
    # 创建旧文件名到新文件名的映射
    filename_map = {}
    for old_path, new_path in renamed_files:
        old_filename = os.path.basename(old_path)
        new_filename = os.path.basename(new_path)
        filename_map[old_filename] = new_filename
    
    # 更新每个Markdown文件
    for md_file in markdown_files:
        print(f"处理Markdown文件: {md_file}")
        
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 更新图片引用
        for old_filename, new_filename in filename_map.items():
            # 处理可能的引用模式
            risk_dirs = ["高血压风险", "PWV超标风险", "高综合风险"]
            for risk_dir in risk_dirs:
                # 使用函数进行替换，而不是反向引用
                pattern = f'!\\[(.*?)\\]\\(image/风险预测/{risk_dir}/{re.escape(old_filename)}\\)'
                
                def replace_func(match):
                    alt_text = match.group(1)
                    return f'![{alt_text}](image/风险预测/{risk_dir}/{new_filename})'
                
                content = re.sub(pattern, replace_func, content)
        
        # 保存更新后的内容
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"已更新Markdown文件: {md_file}")
    
    return True

def main():
    """主函数"""
    print("开始执行风险图片重命名...")
    
    # 项目根目录
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    os.chdir(project_root)
    print(f"工作目录: {os.getcwd()}")
    
    # 1. 重命名风险子目录中的图片
    renamed_files = rename_risk_images()
    
    # 2. 更新Markdown引用
    if renamed_files:
        update_markdown_references(renamed_files)
    
    print("\n重命名完成!")

if __name__ == "__main__":
    main() 