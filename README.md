# 项目本地环境配置
## 命令如下：
```
# 创建一个新的conda环境
conda create -n lim-env python=3.10

# 激活环境
conda activate lim-env

# 安装依赖
# 进入项目所在目录，如 cd E:\code\python\LiM
pip install -r requirements.txt

# 快速测试
# 项目已提供预处理好的数据文件：cstnet-tls1.3_5_packets.csv
# 直接运行分类器查看结果
(lim-env) E:\code\python\LiM>python xgboost_classifier.py
Accuracy: 0.9534883720930233
Classification Report:
                 precision    recall  f1-score   support

     apple.com       0.96      0.91      0.93        99
     cisco.com       1.00      0.99      0.99        99
cloudflare.com       0.98      1.00      0.99        88
  facebook.com       0.84      0.95      0.90        85
    github.com       0.96      0.91      0.94        82
    icloud.com       0.96      0.97      0.96        95
   netflix.com       0.95      0.94      0.95        86
      nike.com       0.95      0.96      0.96       101
    nvidia.com       0.98      0.93      0.95        85
     yahoo.com       0.96      0.96      0.96        83

      accuracy                           0.95       903
     macro avg       0.95      0.95      0.95       903
  weighted avg       0.95      0.95      0.95       903

```

