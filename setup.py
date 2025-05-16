#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="pwv_analysis",
    version="2.0.0",
    description="PWV数据分析工具 - 基于小米可穿戴设备进行动脉硬化筛查与脑卒中风险预测研究",
    author="人民医院小米项目组",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scipy>=1.7.0",
        "scikit-learn>=0.24.0",
        "statsmodels>=0.12.0",
        "openpyxl>=3.0.0",
        "python-docx>=0.8.0",
        "markdown>=3.0.0",
        "xgboost>=1.4.0",
        "plotly>=5.0.0",
        "kaleido>=0.2.0",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "run-pwv-analysis=scripts.run_pwv_analysis:main_orchestrator",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
) 