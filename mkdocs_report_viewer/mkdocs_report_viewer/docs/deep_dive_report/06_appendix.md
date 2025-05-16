# 附录 (Appendix)

## 缩略语表 (List of Abbreviations)

- **PWV**: 脉搏波传导速度 (Pulse Wave Velocity)
- **cfPWV**: 颈-股动脉脉搏波传导速度 (Carotid-Femoral Pulse Wave Velocity)
- **baPWV**: 肱-踝脉搏波传导速度 (Brachial-Ankle Pulse Wave Velocity)
- **SBP**: 收缩压 (Systolic Blood Pressure)
- **DBP**: 舒张压 (Diastolic Blood Pressure)
- **BMI**: 体质指数 (Body Mass Index)
- **HR**: 心率 (Heart Rate)
- **HRV**: 心率变异性 (Heart Rate Variability)
- **SDNN**: 正常RR间期标准差 (Standard Deviation of Normal-to-Normal Intervals)
- **CVD**: 心血管疾病 (Cardiovascular Disease)
- **TC**: 总胆固醇 (Total Cholesterol)
- **TG**: 甘油三酯 (Triglycerides)
- **LDL-C**: 低密度脂蛋白胆固醇 (Low-Density Lipoprotein Cholesterol)
- **HDL-C**: 高密度脂蛋白胆固醇 (High-Density Lipoprotein Cholesterol)
- **hs-CRP**: 高敏C反应蛋白 (high-sensitivity C-Reactive Protein)
- **XGBoost**: Extreme Gradient Boosting
- **SHAP**: SHapley Additive exPlanations
- **AUC**: 受试者工作特征曲线下面积 (Area Under the Curve)
- **ROC**: 受试者工作特征曲线 (Receiver Operating Characteristic curve)
- **PR**: 精确率-召回率曲线 (Precision-Recall curve)
- **FBG**: 空腹血糖 (Fasting Blood Glucose)
- **HbA1c**: 糖化血红蛋白 (Glycated Hemoglobin)
- **eGFR**: 估算肾小球滤过率 (estimated Glomerular Filtration Rate)

## 补充数据表格

### 附表1：不同年龄组PWV参考值

| 年龄段 | 样本量 (n) | PWV参考值 (平均值±SD, m/s) | 正常范围 (10-90百分位, m/s) |
|--------|-----------|--------------------------|--------------------------|
| <40岁  | 146       | 7.14±1.46                | 5.31-9.02                |
| 40-49岁 | 227      | 8.25±1.82                | 6.12-10.46               |
| 50-59岁 | 394      | 9.37±2.01                | 7.03-11.87               |
| 60-69岁 | 334      | 10.63±2.17               | 8.15-13.21               |
| ≥70岁  | 145       | 11.87±2.38               | 9.05-14.89               |

### 附表2：不同血压水平的PWV分布

| 血压分类 | 样本量 (n) | PWV (平均值±SD, m/s) | P值* |
|----------|-----------|---------------------|-----|
| 正常血压 (<120/80 mmHg) | 293 | 8.13±1.98 | 参考组 |
| 正常高值 (120-139/80-89 mmHg) | 541 | 9.22±2.12 | <0.001 |
| 1级高血压 (140-159/90-99 mmHg) | 276 | 10.15±2.28 | <0.001 |
| 2级高血压 (≥160/100 mmHg) | 136 | 11.35±2.46 | <0.001 |

*P值表示与正常血压组比较结果（调整年龄、性别、BMI后）

## 分析方法补充说明

### SHAP值计算方法

SHAP (SHapley Additive exPlanations) 值是基于博弈论中的Shapley值概念，用于解释机器学习模型的输出。SHAP值为每个特征分配一个重要性值，表示该特征对模型预测结果的贡献。

计算公式为：

φᵢ(f,x) = Σ[S⊆N\{i}] |S|!(|N|-|S|-1)!/|N|! [f(S∪{i}) - f(S)]

其中：
- φᵢ(f,x) 是特征i的SHAP值
- N是所有特征的集合
- S是不包含特征i的特征子集
- f(S)是只使用子集S中特征的模型预测值

在本研究中，我们使用Python SHAP库（v0.40.0）计算XGBoost模型中各特征的SHAP值，并生成相关可视化图表。

### 风险预测模型评估方法

模型评估采用了以下步骤：
1. 数据集随机划分：使用分层抽样方法将数据集按70%:30%比例分为训练集和测试集
2. 超参数优化：使用10折交叉验证和网格搜索确定最优超参数组合
3. 模型训练：在训练集上使用最优超参数训练XGBoost模型
4. 性能评估：在独立测试集上评估模型性能，计算AUC、灵敏度、特异度等指标
5. 模型解释：使用SHAP值分析特征重要性和贡献度

[返回目录](00_index.md) 