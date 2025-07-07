import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def predict_sentiment(texts, model, tokenizer, max_length=128, device='cpu'):
    """
    输入texts（str或list），返回情感预测标签和概率
    """
    if isinstance(texts, str):
        texts = [texts]
    model.eval()
    model.to(device)
    encodings = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

    return preds.cpu().numpy(), probs.cpu().numpy()

if __name__ == "__main__":
    from transformers import BertForSequenceClassification, BertTokenizer

    # 已经得到best_model_path
    best_model_path = './results/checkpoint-1554'
    model = BertForSequenceClassification.from_pretrained(best_model_path)
    tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')

    df = pd.read_csv('./ChnSentiCorp_htl_all.csv', encoding='utf-8')
    df = df[['review', 'label']].dropna().drop_duplicates()  # 去除空缺行和重复行
    df['label'] = df['label'].astype(int)  # 标签转为整数
    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    valid_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)
    texts = test_df['review'].tolist()
    labels = test_df['label'].tolist()

    preds = []
    probs = []

    for text in tqdm(texts):
        pred, prob = predict_sentiment(text, model, tokenizer)
        preds.append(pred[0])
        probs.append(prob[0][1])

    # 创建一个DataFrame
    data = pd.DataFrame({
        'Predictions': preds,
        'Probabilities': probs,
        'True_Labels': labels
    })
    # 将DataFrame保存到CSV文件
    csv_file_path = './predictions.csv'
    data.to_csv(csv_file_path, index=False)
    print(f'Data has been saved to {csv_file_path}')

    # 输出混淆矩阵
    cm = confusion_matrix(labels, preds)
    print('Confusion matrix:')
    print(cm)

    # 使用Seaborn库来绘制混淆矩阵
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',xticklabels=['Negative', 'Positive'],yticklabels=['Negative', 'Positive'])
    # 设置图表标题和坐标轴标签
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    confusion_matrix_path = './confusion_matrix.png'
    plt.savefig(confusion_matrix_path)
    print(f'Confusion matrix has been saved to {confusion_matrix_path}')
    plt.show()

    # 输出分类报告
    print('Classification report:')
    print(classification_report(labels, preds))

    # 计算并输出AUC
    auc = roc_auc_score(labels, probs)
    print('AUC:', auc)

    # 可视化ROC曲线
    fpr, tpr, thresholds = roc_curve(labels, probs)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    auc_path = './auc.png'
    plt.savefig(auc_path)
    print(f'AUC has been saved to {auc_path}')
    plt.show()