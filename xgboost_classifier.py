"""
LiM – 使用 NetMatrix 表格表示进行轻量级网络流量分类（XGBoost）

流程概览：
1) 载入由 `pcap_to_netmatrix.py` 生成的 NetMatrix CSV（示例：`cstnet-tls1.3_5_packets.csv`）
2) 将所有特征列作为输入，将 `label` 列编码为数值标签
3) 对数值特征做标准化（均值为 0，方差为 1）
4) 划分训练/测试集，训练 `XGBClassifier`
5) 评估精度并输出分类报告（再把预测结果解码回原始类别名）

特征说明（以 5 个数据包为例）：每个会话行包含 15 个特征列 + 1 个标签列：
- 对每个包依次记录：`ip_total_len`, `ip_ttl`, `inter_arrival_time`
- 列名格式：`p<N>_<feature>`，例如 `p1_ip_total_len`, `p1_ip_ttl`, `p1_inter_arrival_time` ... 直到 `p5_*`
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score

# 读入 NetMatrix 数据表（可替换为你自己的输出 CSV）
df = pd.read_csv('./cstnet-tls1.3_5_packets.csv')

# 定义特征列
feature_columns = [col for col in df.columns if col != 'label']
# 定义标签列
label_column = 'label'

# 划分特征与标签
X = df[feature_columns]
y = df[label_column]

# 将字符串类别编码为整数标签，以便训练模型
# 字符编码器类
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 本数据集中所有输入特征均为数值型
numerical_features = feature_columns 

# 数值特征标准化类，有助于加速收敛与稳定性
numerical_transformer = StandardScaler()

# 数据预处理：标准化数值特征
X_numerical = numerical_transformer.fit_transform(X[numerical_features])

# 组合预处理后的特征
X_preprocessed = np.hstack([X_numerical])

# 划分训练/测试集（20% 用作测试，固定随机种子）
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y_encoded, test_size=0.2, random_state=42)

# 训练 XGBoost 分类器
# `eval_metric='mlogloss'`：多分类对数损失，这个作为损失函数
xgb_classifier = xgb.XGBClassifier(eval_metric='mlogloss')

# 装入训练数据，训练 XGgBoost 机器学习模型
xgb_classifier.fit(X_train, y_train)

# 在测试集上推理，得到模型预测结果
y_pred = xgb_classifier.predict(X_test)

# 将整数型的预测结果y_pred 与 真实标签 y_test 进行解码,变成字符串，便于阅读报告
y_pred_decoded = label_encoder.inverse_transform(y_pred)
y_test_decoded = label_encoder.inverse_transform(y_test)

# 评估指标：准确率 + 分类报告（含精确率、召回率、F1）
print("Accuracy:", accuracy_score(y_test_decoded, y_pred_decoded))
print("Classification Report:\n", classification_report(y_test_decoded, y_pred_decoded))