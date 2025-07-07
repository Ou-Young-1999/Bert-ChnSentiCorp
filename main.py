# -*- coding: utf-8 -*-
"""
BERT中文文本分类：以ChnSentiCorp_htl_all为例
功能：数据加载、划分、分词、模型构建、训练、验证、测试、可视化
"""
# 设置 HuggingFace 镜像（国内建议）
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


# ========== 1. 数据加载 ==========
df = pd.read_csv('./ChnSentiCorp_htl_all.csv', encoding='utf-8')
print("数据预览：")
print(df.head())

# ========== 2. 数据预处理 ==========
# 有 'review' 和 'label' 列
df = df[['review', 'label']].dropna().drop_duplicates()  # 去除空缺行和重复行
df['label'] = df['label'].astype(int)  # 标签转为整数

# ========== 3. 数据集划分 ==========
# stratify：确保分割后的数据集按 label 列的类别比例分层（避免类别不平衡）
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
valid_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)
print(f"训练集：{len(train_df)}，验证集：{len(valid_df)}，测试集：{len(test_df)}")
label_counts = train_df['label'].value_counts()
print(f'训练集比例：{label_counts}')
label_counts = valid_df['label'].value_counts()
print(f'验证集比例：{label_counts}')
label_counts = test_df['label'].value_counts()
print(f'测试集比例：{label_counts}')

# ========== 4. Tokenizer准备 ==========
MODEL_NAME = './bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
text = "我喜欢学习人工智能"
tokens = tokenizer.tokenize(text)
print(tokens)

encoding = tokenizer.encode_plus(
    text,
    max_length=16,
    truncation=True,
    padding='max_length',
    return_tensors='pt'
)
print("input_ids:", encoding['input_ids'])
print("token_type_ids:", encoding['token_type_ids'])
print("attention_mask:", encoding['attention_mask'])

# ========== 5. 自定义Dataset ==========
class SentimentDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.texts = df['review'].tolist()
        self.labels = df['label'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer( # 默认调用tokenizer.encode_plus
            self.texts[idx],
            max_length=self.max_length,
            truncation=True, # 截断
            padding='max_length',
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

train_dataset = SentimentDataset(train_df, tokenizer)
valid_dataset = SentimentDataset(valid_df, tokenizer)
test_dataset = SentimentDataset(test_df, tokenizer)
for i in range(3):  # 输出前三条
    print(test_dataset[i])

# ========== 6. 加载BERT模型 ==========
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# ========== 7. 训练参数 ==========
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    logging_dir='./logs',
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

# ========== 8. 评估函数 ==========
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc
    }

# ========== 9. Trainer训练 ==========
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

# ========== 10. 验证集评估 ==========
print("验证集评估：")
valid_preds_output = trainer.predict(valid_dataset)
valid_preds = np.argmax(valid_preds_output.predictions, axis=1)
print(classification_report(valid_df['label'], valid_preds, digits=4))

# ========== 11. 测试集评估 ==========
print("测试集评估：")
test_preds_output = trainer.predict(test_dataset)
test_preds = np.argmax(test_preds_output.predictions, axis=1)
print(classification_report(test_df['label'], test_preds, digits=4))

"""
【注意事项】
1. 若GPU内存不足，适当调小batch_size或max_length。
2. 若数据集字段名不同，请相应修改。
3. 需要提前安装 transformers、torch、pandas、sklearn、matplotlib。
4. 若在国内，建议加上 HF_ENDPOINT 换源行。
"""