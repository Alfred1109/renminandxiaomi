# 1. 引言 (Introduction)

![研究背景图](https://via.placeholder.com/800x250/4051b5/ffffff?text=%E7%A0%94%E7%A9%B6%E8%83%8C%E6%99%AF%E4%B8%8E%E6%84%8F%E4%B9%89)

## 心血管疾病的全球负担

心脑血管疾病(CVD)是全球范围内的主要死亡和致残原因。根据世界卫生组织(WHO)的最新数据，CVD每年导致约1790万人死亡，占全球死亡总数的32%。在中国，CVD已成为首要死因，年死亡人数超过350万，且发病率呈现年轻化趋势。随着人口老龄化和生活方式西化，心血管疾病负担预计将进一步加重，给医疗系统和社会经济带来巨大压力。

> "预防胜于治疗，早期识别高风险人群并采取干预措施是降低心血管疾病负担的关键策略。" —— 世界卫生组织《全球心血管疾病预防与控制战略》

## 动脉僵硬度与脉搏波传导速度

动脉僵硬度是心血管疾病的早期标志，其增加先于临床症状和器质性血管病变的出现。脉搏波传导速度(PWV)作为评估动脉僵硬度的"黄金标准"，已被欧洲高血压学会和欧洲心脏病学会推荐为评估靶器官损伤的重要指标。

PWV测量原理基于脉搏波在动脉中传播的速度，计算公式为：

$$PWV = \frac{距离(D)}{传导时间(Δt)}$$

其中，距离D通常为两个测量点之间的血管长度，传导时间Δt为脉搏波从近端到远端的传播时间。

### PWV的临床意义

多项大型前瞻性研究证实，PWV是心血管事件和全因死亡的独立预测因子，其预测价值优于传统风险因素。具体而言：

1. **心血管事件预测**：PWV每增加1m/s，心血管事件风险增加14%，心血管死亡风险增加15%
2. **器官损伤评估**：PWV增加与心脏、肾脏和脑血管损伤密切相关
3. **治疗效果监测**：PWV可用于评估生活方式干预和药物治疗对动脉僵硬度的改善效果
4. **亚临床动脉粥样硬化识别**：在常规检查未见异常的人群中，PWV可识别早期血管功能改变

<figure markdown>
  ![PWV与心血管风险关系图](https://via.placeholder.com/600x400/ffffff/333333?text=PWV%E4%B8%8E%E5%BF%83%E8%A1%80%E7%AE%A1%E9%A3%8E%E9%99%A9%E5%85%B3%E7%B3%BB%E5%9B%BE)
  <figcaption>图1. PWV与心血管事件风险的关系曲线</figcaption>
</figure>

## 传统PWV测量的局限性

尽管PWV具有重要的临床价值，但传统测量方法存在多项局限性：

| 测量方法 | 优势 | 局限性 |
|---------|------|--------|
| 颈-股动脉PWV | 金标准，证据最充分 | 设备昂贵，操作复杂，需专业人员 |
| 肱-踝PWV | 操作相对简便 | 受外周血管影响大，标准化不足 |
| 超声测量法 | 可同时评估血管结构 | 依赖操作者经验，难以大规模应用 |
| MRI测量法 | 准确度高，可视化好 | 成本高，不适合筛查和随访 |

这些局限性导致PWV测量主要局限于医院和专业研究机构，难以在社区和家庭环境中广泛应用，制约了其在心血管疾病一级预防中的价值发挥。

## 可穿戴设备与PWV监测的新机遇

随着传感器技术、信号处理算法和人工智能的快速发展，可穿戴设备为PWV的便携化、连续化监测提供了新的技术路径。小米健康科技研究院近期开发的可穿戴设备采用光电容积脉搏波描记法(PPG)和加速度传感器融合技术，可实现无创、便捷的PWV测量。

### 技术优势

- **便携性**：集成于日常佩戴的智能手表，无需额外设备
- **易用性**：自动采集和分析，无需专业培训
- **连续性**：支持定期和长期监测，捕捉动态变化
- **成本效益**：相比传统设备，大幅降低测量成本
- **数据整合**：可与其他健康数据(如心率、活动量)联合分析

<figure markdown>
  ![可穿戴设备PWV测量原理图](https://via.placeholder.com/600x300/ffffff/333333?text=%E5%8F%AF%E7%A9%BF%E6%88%B4%E8%AE%BE%E5%A4%87PWV%E6%B5%8B%E9%87%8F%E5%8E%9F%E7%90%86)
  <figcaption>图2. 基于可穿戴设备的PWV测量原理示意图</figcaption>
</figure>

## 研究意义与创新点

本研究旨在探索基于小米可穿戴设备的PWV测量在心血管风险评估中的应用价值，具有以下创新点和意义：

### 创新点

1. **技术创新**：首次大规模验证基于消费级可穿戴设备的PWV测量技术
2. **应用创新**：将PWV监测从医疗机构拓展至社区和家庭场景
3. **模型创新**：整合多源数据构建心血管风险预测模型
4. **方法创新**：采用机器学习方法挖掘PWV与其他风险因素的复杂关系

### 研究意义

1. **临床意义**：为心血管风险分层提供新的客观指标
2. **预防医学意义**：推动心血管疾病一级预防策略向前移
3. **公共卫生意义**：为大规模人群心血管风险筛查提供可行方案
4. **技术发展意义**：促进可穿戴健康监测技术的临床转化

## 研究目标

本研究的具体目标包括：

1. 评估基于小米可穿戴设备测量的PWV与传统设备测量结果的一致性
2. 分析PWV与人口学特征、生活方式和传统心血管风险因素的关系
3. 探究不同年龄、性别和基础疾病状态下的PWV参考值和变异规律
4. 构建整合PWV的心血管风险预测模型，并评估其预测性能
5. 提出基于PWV监测的心血管健康管理策略和实施路径

通过实现这些目标，本研究将为心血管疾病的早期预防和精准管理提供科学依据和技术支持，推动"以预防为中心"的心血管健康管理范式转变。

---

**参考文献**:

1. Vlachopoulos C, et al. Prediction of cardiovascular events and all-cause mortality with arterial stiffness: a systematic review and meta-analysis. J Am Coll Cardiol. 2010;55(13):1318-1327.
2. Townsend RR, et al. Recommendations for Improving and Standardizing Vascular Research on Arterial Stiffness: A Scientific Statement From the American Heart Association. Hypertension. 2015;66(3):698-722.
3. Zhong Q, et al. Arterial stiffness and cardiovascular risk stratification in hypertensive population: a systematic review and meta-analysis. Eur J Prev Cardiol. 2020;27(16):1775-1785.
4. WHO Global Status Report on Noncommunicable Diseases 2023. Geneva: World Health Organization; 2023.
5. China Cardiovascular Disease Report 2022. National Center for Cardiovascular Diseases, China; 2023.

[返回目录](00_index.md) 