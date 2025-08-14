# 🧬 蛋白质内在无序区域预测模型项目说明

📌 项目概述

本项目基于ESM-2蛋白质语言模型，构建了一个能够预测蛋白质内在无序区域(IDRs)的深度学习模型。内在无序区域在蛋白质功能中扮演关键角色，准确预测这些区域对于理解蛋白质功能、疾病机制和药物设计具有重要意义。

🧠 模型架构

本项目采用ESM-2（Evolutionary Scale Modeling）作为基础模型：
• 模型名称: facebook/esm2_t30_150M_UR50D

• 参数规模: 1.5亿参数

• 任务类型: Token分类任务（每个氨基酸残基的二分类）

• 输出层: 二元分类层（0: 非结合位点, 1: 结合位点）

⚙️ 数据处理流程

1. 数据加载:
   • 从PKL文件加载蛋白质序列和标签

   • 转换为Hugging Face Dataset格式

2. 序列分词:
   • 使用ESM-2专用分词器

   • 最大序列长度: 2048

   • 添加特殊token: [CLS], [SEP]

3. 标签对齐:
   • 将原始序列标签与分词后的token对齐

   • 特殊token标签设为-100（训练时忽略）

   • 确保每个氨基酸残基对应正确的标签

🚀 训练策略

采用5折交叉验证策略提高模型鲁棒性：
• 折数: 5

• 每折训练轮数: 6

• 批次大小: 4（梯度累积步数: 3）

• 优化器: AdamW

• 学习率: 2e-5

• 评估指标: F1分数（主要评估指标）

训练参数细节

per_device_train_batch_size=4,
gradient_accumulation_steps=3,
per_device_eval_batch_size=8,
eval_steps=100,
save_steps=100,
num_train_epochs=6,
learning_rate=2e-5,
weight_decay=0.01,
fp16=True,  # 混合精度训练


📊 评估指标

使用加权平均的F1分数作为主要评估指标：
• 精确率(Precision): TP/(TP+FP)

• 召回率(Recall): TP/(TP+FN)

• F1分数: 2×Precision×Recall/(Precision+Recall)

🧪 模型推理

实现预测函数predict_idrs:
1. 对输入序列进行分词
2. 模型推理获取预测结果
3. 过滤特殊token的预测
4. 返回每个残基的预测标签(0或1)

📤 结果提交

实现generate_submission_file函数生成符合比赛要求的提交文件：
• 格式: CSV

• 表头: proteinID,sequence,IDRs

• 内容: 每行一个蛋白质的预测结果

📂 项目结构


protein_idr_prediction/

├── data/

│   └── WSAA_data_public.pkl        # 原始数据集

├── finetuned_model/                 # 训练好的模型

│   └── esm2-150M-L3000/

│       └── cross_valid/

│           ├── fold_1/              # 第一折模型

│           ├── fold_2/              # 第二折模型

│           ├── ...                 

│           └── fold_5/              # 第五折模型

├── results/                         # 结果文件

│   └── submit.csv                   # 提交文件

├── protein_idr_prediction.ipynb      # Jupyter Notebook

└── requirements.txt                 # 依赖库


🛠️ 使用说明

1. 环境配置:
   pip install torch transformers datasets scikit-learn tqdm
   

2. 数据准备:
   • 将WSAA_data_public.pkl放在项目目录的data/文件夹下

3. 训练模型:
   • 运行Notebook中的"5折交叉验证训练"部分

4. 模型推理:
   • 使用predict_idrs函数预测单个序列

   • 使用generate_submission_file生成批量预测结果

5. 结果提交:
   • 将生成的submit.csv提交至比赛平台

🏆 比赛成绩

• 初赛排名: 11

• 初赛分数: 0.8070

• 复赛排名: 17

• 复赛分数: 0.8832

📈 性能优化建议

1. 模型架构:
   • 尝试更大的ESM模型变体（如esm2_t33_650M）

   • 添加CRF层优化序列标注一致性

2. 训练策略:
   • 增加训练轮数（10-20轮）

   • 使用学习率预热和衰减策略

   • 尝试不同的优化器（如LAMB）

3. 数据处理:
   • 实现数据增强（序列截断、滑动窗口）

   • 添加蛋白质家族信息作为额外特征

4. 集成学习:
   • 将5折模型的预测结果进行集成

   • 结合不同ESM模型的预测结果

本项目提供了一个强大的基线模型，通过进一步优化和调整，有望在蛋白质内在无序区域预测任务中取得更好的性能。
