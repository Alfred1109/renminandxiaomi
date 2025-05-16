# 8. 技术附录 (Technical Appendix)

本技术附录详细介绍了研究中使用的关键算法、数据处理方法和技术实现细节，方便技术人员复现或进一步开发。

## 8.1 PWV计算方法详解

### 8.1.1 PPG信号预处理

原始PPG信号的处理流程包括以下步骤：

1. **带通滤波**：应用0.5Hz-8Hz的带通滤波器，去除基线漂移和高频噪声
    ```python
    from scipy import signal
    
    def bandpass_filter(data, lowcut=0.5, highcut=8.0, fs=100, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        return signal.filtfilt(b, a, data)
    ```

2. **峰值检测**：使用自适应阈值算法识别PPG波峰和波谷
    ```python
    def detect_peaks(ppg_signal, fs=100):
        # 计算信号导数
        derivative = np.gradient(ppg_signal)
        
        # 使用最小距离和高度阈值识别峰值
        min_distance = int(0.5 * fs)  # 最小心跳间隔(0.5秒)
        peaks, _ = signal.find_peaks(ppg_signal, 
                                     height=0.4*np.max(ppg_signal),
                                     distance=min_distance)
        
        return peaks
    ```

3. **异常心搏检测**：识别并排除不规则心搏
    ```python
    def remove_irregular_heartbeats(peaks, ppg_signal, fs=100):
        intervals = np.diff(peaks)
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        # 排除间隔异常的心搏
        valid_indices = []
        for i in range(len(intervals)):
            if abs(intervals[i] - mean_interval) < 2 * std_interval:
                valid_indices.append(i)
                valid_indices.append(i+1)
        
        valid_indices = list(set(valid_indices))
        valid_indices.sort()
        
        return peaks[valid_indices]
    ```

### 8.1.2 PTT (Pulse Transit Time) 计算

PTT计算基于两个不同位置的脉搏波到达时间差：

```python
def calculate_ptt(proximal_peaks, distal_peaks, fs=100):
    # 匹配最近的峰值对
    ptt_values = []
    for prox_peak in proximal_peaks:
        # 寻找距离最近的远端峰值
        candidates = distal_peaks[distal_peaks > prox_peak]
        if len(candidates) > 0:
            nearest_distal = candidates[0]
            ptt = (nearest_distal - prox_peak) / fs  # 转换为秒
            
            # 验证PTT在合理范围内 (通常30-200ms)
            if 0.03 <= ptt <= 0.2:
                ptt_values.append(ptt)
    
    return np.array(ptt_values)
```

### 8.1.3 PWV计算公式

PWV计算基于测量距离和PTT：

```python
def calculate_pwv(distance_meters, ptt_seconds):
    """
    计算脉搏波传导速度
    
    参数:
        distance_meters: 两个测量点之间的距离(米)
        ptt_seconds: 脉搏传导时间(秒)
    
    返回:
        pwv: 脉搏波传导速度(米/秒)
    """
    return distance_meters / ptt_seconds
```

## 8.2 特征工程详细流程

### 8.2.1 时域特征提取

从PPG信号中提取的主要时域特征包括：

```python
def extract_time_domain_features(ppg_signal, peaks, fs=100):
    # 计算心率
    rr_intervals = np.diff(peaks) / fs  # 转换为秒
    heart_rate = 60 / np.mean(rr_intervals)  # 转换为bpm
    
    # 计算SDNN (心率变异性指标)
    sdnn = np.std(rr_intervals)
    
    # 计算RMSSD (相邻RR间期差值的均方根)
    rmssd = np.sqrt(np.mean(np.diff(rr_intervals)**2))
    
    # 脉搏波幅度变异性
    amplitude = ppg_signal[peaks]
    amplitude_variation = np.std(amplitude) / np.mean(amplitude)
    
    # 波形宽度 (半高宽)
    width_50 = calculate_width_at_half_maximum(ppg_signal, peaks, fs)
    
    return {
        'heart_rate': heart_rate,
        'sdnn': sdnn,
        'rmssd': rmssd,
        'amplitude_variation': amplitude_variation,
        'width_50': width_50
    }
```

### 8.2.2 频域特征提取

频域分析使用快速傅里叶变换(FFT)：

```python
def extract_frequency_domain_features(rr_intervals):
    # 重采样以获得均匀间隔的时间序列
    fs_interp = 4  # Hz
    t_interp = np.cumsum(rr_intervals)
    t_interp = np.insert(t_interp, 0, 0)
    
    # 创建均匀时间序列
    t_uniform = np.arange(0, t_interp[-1], 1/fs_interp)
    
    # 插值
    rr_interpolated = np.interp(t_uniform, t_interp[:-1], rr_intervals)
    
    # 应用汉宁窗并计算FFT
    windowed = rr_interpolated * signal.windows.hann(len(rr_interpolated))
    fft = np.abs(np.fft.rfft(windowed))
    
    # 计算频率分量
    freqs = np.fft.rfftfreq(len(windowed), 1/fs_interp)
    
    # 计算频段功率
    vlf_power = np.sum(fft[(freqs >= 0.0033) & (freqs < 0.04)]**2)
    lf_power = np.sum(fft[(freqs >= 0.04) & (freqs < 0.15)]**2)
    hf_power = np.sum(fft[(freqs >= 0.15) & (freqs < 0.4)]**2)
    total_power = vlf_power + lf_power + hf_power
    
    # 归一化功率
    lf_normalized = lf_power / (lf_power + hf_power)
    hf_normalized = hf_power / (lf_power + hf_power)
    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0
    
    return {
        'vlf_power': vlf_power,
        'lf_power': lf_power,
        'hf_power': hf_power,
        'total_power': total_power,
        'lf_normalized': lf_normalized,
        'hf_normalized': hf_normalized,
        'lf_hf_ratio': lf_hf_ratio
    }
```

### 8.2.3 波形特征提取

从PWV波形中提取的关键特征：

```python
def extract_waveform_features(ppg_signal, peaks, fs=100):
    features = {}
    
    # 提取每个心动周期并归一化
    cycles = []
    for i in range(len(peaks)-1):
        start, end = peaks[i], peaks[i+1]
        if end - start > 0.3*fs and end - start < 1.5*fs:  # 筛选合理周期
            cycle = ppg_signal[start:end]
            # 重采样到固定长度
            cycle_resampled = signal.resample(cycle, 100)
            # 归一化
            cycle_normalized = (cycle_resampled - np.min(cycle_resampled)) / \
                              (np.max(cycle_resampled) - np.min(cycle_resampled))
            cycles.append(cycle_normalized)
    
    if len(cycles) < 5:
        return features  # 返回空特征集
    
    # 计算平均波形
    average_cycle = np.mean(cycles, axis=0)
    
    # 特征提取
    # 1. 收缩期上升时间
    rise_time_idx = np.argmax(average_cycle)
    rise_time = rise_time_idx / 100
    features['rise_time'] = rise_time
    
    # 2. 舒张期时间比例
    features['diastolic_ratio'] = 1 - rise_time
    
    # 3. 反射波指数 (通常在原始波峰后的次级波峰)
    first_derivative = np.gradient(average_cycle)
    inflection_points = np.where(np.diff(np.sign(first_derivative)))[0]
    potential_dicrotic_points = [p for p in inflection_points if p > rise_time_idx]
    
    if potential_dicrotic_points:
        dicrotic_idx = potential_dicrotic_points[0]
        features['reflection_index'] = average_cycle[dicrotic_idx] / average_cycle[rise_time_idx]
    else:
        features['reflection_index'] = 0
    
    # 4. 波形面积
    features['area_under_curve'] = np.trapz(average_cycle)
    
    # 5. 波形二阶导数特征
    second_derivative = np.gradient(first_derivative)
    features['acceleration_slope'] = np.mean(np.abs(second_derivative[:rise_time_idx]))
    
    return features
```

## 8.3 XGBoost模型构建细节

### 8.3.1 超参数优化

使用网格搜索和交叉验证优化XGBoost模型参数：

```python
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

def optimize_xgboost_hyperparameters(X_train, y_train):
    # 定义参数网格
    param_grid = {
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2],
        'min_child_weight': [1, 3, 5]
    }
    
    # 初始化XGBoost模型
    model = xgb.XGBClassifier(objective='binary:logistic', 
                             eval_metric='auc',
                             use_label_encoder=False,
                             random_state=42)
    
    # 设置网格搜索
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        verbose=1,
        n_jobs=-1
    )
    
    # 执行搜索
    grid_search.fit(X_train, y_train)
    
    # 输出最佳参数
    print("Best parameters:", grid_search.best_params_)
    print("Best AUC score:", grid_search.best_score_)
    
    return grid_search.best_estimator_, grid_search.best_params_
```

### 8.3.2 模型训练与评估代码

完整的模型训练、验证和测试流程：

```python
def train_and_evaluate_model(X_train, y_train, X_test, y_test, best_params):
    # 初始化模型
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        use_label_encoder=False,
        random_state=42,
        **best_params
    )
    
    # 训练模型
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=True,
        early_stopping_rounds=20
    )
    
    # 预测测试集
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # 计算评估指标
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import confusion_matrix, classification_report
    
    results = {
        'auc': roc_auc_score(y_test, y_pred_proba),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred)
    }
    
    print("AUC Score:", results['auc'])
    print("Accuracy:", results['accuracy'])
    print("Precision:", results['precision'])
    print("Recall:", results['recall'])
    print("F1 Score:", results['f1'])
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])
    print("\nClassification Report:")
    print(results['classification_report'])
    
    return model, results
```

### 8.3.3 SHAP值计算与可视化

使用SHAP库解释模型预测：

```python
import shap

def explain_model_with_shap(model, X_train, X_test):
    # 创建SHAP解释器
    explainer = shap.TreeExplainer(model)
    
    # 计算SHAP值
    shap_values = explainer.shap_values(X_test)
    
    # 汇总图 - 特征重要性
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar")
    plt.savefig("output/figures/shap_importance.png", dpi=300, bbox_inches='tight')
    
    # 详细图 - 特征影响图
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test)
    plt.savefig("output/figures/shap_summary.png", dpi=300, bbox_inches='tight')
    
    # 依赖图 - 主要特征
    top_features = np.argsort(np.abs(shap_values).mean(0))[-5:]  # 取前5个特征
    for feature_idx in top_features:
        feature_name = X_test.columns[feature_idx]
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(feature_idx, shap_values, X_test, 
                             feature_names=X_test.columns)
        plt.savefig(f"output/figures/shap_dependence_{feature_name}.png", 
                    dpi=300, bbox_inches='tight')
    
    # 力图 - 个例解释
    sample_idx = 0  # 选择一个样本展示
    plt.figure(figsize=(12, 4))
    shap.force_plot(explainer.expected_value, 
                    shap_values[sample_idx,:], 
                    X_test.iloc[sample_idx,:], 
                    matplotlib=True,
                    show=False)
    plt.savefig("output/figures/shap_force_plot.png", dpi=300, bbox_inches='tight')
    
    return shap_values
```

## 8.4 数据预处理完整流程

数据清洗、归一化和特征选择的完整流程：

```python
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectFromModel
import pandas as pd
import numpy as np

def preprocess_data(data, target_column, test_size=0.3, random_state=42):
    # 分离特征和目标变量
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # 处理缺失值
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # 检测并处理异常值
    X_clean = handle_outliers(X_imputed)
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_clean), columns=X_clean.columns)
    
    # 特征选择
    selector = SelectFromModel(xgb.XGBClassifier(n_estimators=100, random_state=random_state))
    selector.fit(X_scaled, y)
    X_selected = selector.transform(X_scaled)
    selected_features = X_scaled.columns[selector.get_support()]
    X_selected = pd.DataFrame(X_selected, columns=selected_features)
    
    # 训练测试集划分
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, scaler, selected_features

def handle_outliers(df, method='iqr', threshold=1.5):
    df_clean = df.copy()
    
    if method == 'iqr':
        for column in df.columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # 将异常值替换为边界值
            df_clean[column] = np.where(df_clean[column] < lower_bound, lower_bound, df_clean[column])
            df_clean[column] = np.where(df_clean[column] > upper_bound, upper_bound, df_clean[column])
    
    elif method == 'zscore':
        from scipy import stats
        z_scores = stats.zscore(df)
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < threshold).all(axis=1)
        df_clean = df[filtered_entries]
    
    return df_clean
```

## 8.5 客户端-服务器通信协议

用于小米设备与分析服务器之间数据传输的协议详细设计：

```json
// 设备发送至服务器的数据格式示例
{
  "device_id": "MI-BAND-12345",
  "user_id": "USER1234",
  "timestamp": 1620000000,
  "data_type": "pwv_measurement",
  "data": {
    "raw_ppg": [...],        // 原始PPG数据（可选）
    "sampling_rate": 100,    // 采样率(Hz)
    "pwv_value": 9.8,        // 计算得到的PWV值(m/s)
    "ptt_values": [...],     // 多次PTT测量值(ms)
    "quality_score": 0.92,   // 信号质量评分(0-1)
    "measurement_conditions": {
      "posture": "sitting",  // 测量姿势
      "arm_position": "heart_level", // 手臂位置
      "activity_level": "resting"    // 活动状态
    },
    "device_meta": {
      "firmware_version": "2.1.0",
      "battery_level": 78,
      "sensor_type": "PPG-IR-3.0"
    }
  }
}

// 服务器响应格式示例
{
  "status": "success",
  "timestamp": 1620000010,
  "message": "Data received successfully",
  "analysis_results": {
    "pwv_percentile": 65,      // 同龄人群百分位
    "trend": "stable",         // 变化趋势
    "risk_category": "moderate", // 风险分类
    "recommendation": "Continue monitoring, consider lifestyle modifications"
  },
  "next_measurements": {
    "recommended_time": 1620086400, // 下次建议测量时间
    "special_instructions": "Measure after 5 minutes of rest" 
  }
}
```

## 8.6 实验协议与数据收集标准

### 8.6.1 实验室测量标准流程

为确保数据质量和一致性，所有PWV测量遵循以下标准流程：

1. **准备阶段**：
   - 受试者休息10分钟，保持安静坐姿
   - 测量环境温度维持在22-24°C
   - 禁止受试者测量前2小时内摄入咖啡因、酒精或大量食物
   - 记录受试者最近药物使用情况

2. **设备佩戴标准**：
   - 腕部设备紧贴皮肤，位于腕骨上方2cm处
   - 设备传感器对准桡动脉位置
   - 佩戴松紧度：可插入一根手指但不松动

3. **测量过程**：
   - 手臂保持心脏水平
   - 每次测量持续3分钟
   - 连续进行3次测量，取平均值
   - 测量间隔1分钟

4. **数据验证标准**：
   - 信号质量指数>0.8
   - 三次测量结果变异系数<10%
   - 异常值自动检测与排除

### 8.6.2 家庭测量指导规范

为确保家庭自测数据质量，我们提供以下标准指导：

1. **测量时间选择**：
   - 优先晨起后、晚间睡前测量
   - 避免剧烈运动后30分钟内测量
   - 保持每日固定时间段测量

2. **环境要求**：
   - 安静、温度适宜的环境
   - 测量时避免说话、走动
   - 避免强电磁干扰环境

3. **测量姿势**：
   - 保持坐姿，背部有支撑
   - 双脚平放于地面
   - 手臂放置于桌面，保持心脏水平

4. **记录相关因素**：
   - 近期睡眠质量
   - 情绪状态
   - 当天药物使用情况
   - 特殊饮食情况

[返回目录](00_index.md) 